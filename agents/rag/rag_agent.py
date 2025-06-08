"""
RAG (Retrieval Augmented Generation) Agent

This module implements a RAG agent that:
1. Takes a user question
2. Searches Azure AI Search index for relevant documents
3. Sends the context and question to GPT-4 for generating an answer

The agent follows Azure best practices:
- Uses managed identity when possible
- Implements proper error handling and logging
- Uses connection pooling and retry logic
- Includes monitoring and observability
"""

import os
import logging
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient
from azure.monitor.opentelemetry import configure_azure_monitor
from opentelemetry import trace
from opentelemetry.instrumentation.openai import OpenAIInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from openai import AzureOpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Get tracer for OpenTelemetry spans
tracer = trace.get_tracer(__name__)



@dataclass
class RAGResponse:
    """Response from the RAG agent."""
    answer: str
    sources: List[Dict[str, Any]]
    query: str
    retrieval_time: float
    generation_time: float
    total_time: float


@dataclass
class RetrievedDocument:
    """Document retrieved from search."""
    docid: str
    content: str
    score: float
    metadata: Dict[str, Any]


class RAGAgent:
    """
    RAG (Retrieval Augmented Generation) Agent

    This agent implements the RAG pattern by:
    1. Retrieving relevant documents from Azure AI Search
    2. Augmenting the user query with retrieved context
    3. Generating responses using GPT-4 with the enhanced context

    Features:
    - Hybrid search (keyword + semantic)
    - Configurable retrieval parameters
    - Error handling and retry logic
    - Performance monitoring
    - Context window management
    """

    def __init__(
        self,
        search_service_name: str,
        search_index_name: str,
        azure_openai_endpoint: str,
        azure_openai_api_version: str = "2024-12-01-preview",
        chat_model: str = "gpt-4.1",
        embedding_model: str = "text-embedding-3-small",
        max_context_length: int = 20000,
        top_k_documents: int = 3
    ):
        """
        Initialize the RAG agent.

        Args:
            search_service_name: Name of Azure AI Search service
            search_index_name: Name of the search index
            azure_openai_endpoint: Azure OpenAI endpoint (optional, can use env var)
            azure_openai_api_version: API version for Azure OpenAI
            chat_model: Name of the chat completion model (e.g., gpt-4, gpt-4o)
            embedding_model: Name of the embedding model for semantic search
            max_context_length: Maximum context length for the LLM
            top_k_documents: Number of documents to retrieve
        """
        self.search_service_name = search_service_name
        self.search_index_name = search_index_name
        self.chat_model = chat_model
        self.embedding_model = embedding_model
        self.max_context_length = max_context_length
        self.top_k_documents = top_k_documents

        #for more details see:
        #https://learn.microsoft.com/en-us/python/api/overview/azure/ai-projects-readme?view=azure-python-preview
        self.project_client = AIProjectClient(
            subscription_id=os.environ["AZURE_SUBSCRIPTION_ID"],
            resource_group_name=os.environ["AZURE_RESOURCE_GROUP"],
            project_name=os.environ["AZURE_PROJECT_NAME"],
            credential=DefaultAzureCredential(),
            endpoint=os.environ["AZURE_PROJECT_ENDPOINT"],
            )

        connection_string = self.project_client.telemetry.get_connection_string()
        logger.info(
            f"Application Insights connection string: {connection_string}")
        

        if not connection_string:
            logger.error("Application Insights is not enabled. Enable by going to Tracing in your Azure AI Foundry project.")
            exit()

        configure_azure_monitor(connection_string=connection_string) #enable telemetry collection
        
        # Initialize auto-instrumentation for OpenAI and requests
        # This automatically traces all OpenAI API calls and HTTP requests
        logger.info("Setting up auto-instrumentation for OpenAI and requests...")
        OpenAIInstrumentor().instrument()
        RequestsInstrumentor().instrument()
        logger.info("✅ Auto-instrumentation enabled for OpenAI and requests")
        
        # Initialize search client with proper authentication
        self._init_search_client()

        # Initialize Azure OpenAI client
        self._init_openai_client(
            azure_openai_endpoint, azure_openai_api_version)

        logger.info(
            f"RAG Agent initialized with chat model: {chat_model}, embedding model: {embedding_model}")

    def _init_search_client(self):
        """Initialize Azure AI Search client with authentication."""
        try:
            # Try API key first (for development), then managed identity
            api_key = os.getenv("AZURE_SEARCH_API_KEY")
            if api_key:
                credential = AzureKeyCredential(api_key)
                logger.warning(
                    "Using API key authentication for Azure Search - consider managed identity for production")
            else:
                credential = DefaultAzureCredential()
                logger.info(
                    "Using managed identity for Azure Search authentication")

            self.search_client = SearchClient(
                endpoint=f"https://{self.search_service_name}.search.windows.net",
                index_name=self.search_index_name,
                credential=credential
            )

        except Exception as e:
            logger.error(f"Failed to initialize search client: {str(e)}")
            raise

    def _init_openai_client(self, azure_openai_endpoint: Optional[str], api_version: str):
        """Initialize Azure OpenAI client with authentication."""
        try:
            endpoint = azure_openai_endpoint or os.getenv(
                "AZURE_OPENAI_ENDPOINT")
            if not endpoint:
                raise ValueError(
                    "Azure OpenAI endpoint must be provided via parameter or AZURE_OPENAI_ENDPOINT environment variable")

            # Try API key first (for development), then managed identity
            api_key = os.getenv("AZURE_OPENAI_API_KEY")
            if api_key:
                self.openai_client = AzureOpenAI(
                    azure_endpoint=endpoint,
                    api_key=api_key,
                    api_version=api_version
                )
                logger.warning(
                    "Using API key authentication for Azure OpenAI - consider managed identity for production")
            else:
                # Use managed identity
                credential = DefaultAzureCredential()
                self.openai_client = AzureOpenAI(
                    azure_endpoint=endpoint,
                    azure_ad_token_provider=credential,
                    api_version=api_version
                )
                logger.info(
                    "Using managed identity for Azure OpenAI authentication")

        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {str(e)}")
            raise

    def _generate_embedding(self, text: str, max_retries: int = 3) -> List[float]:
        """
        Generate embedding for text using Azure OpenAI.
        Auto-instrumentation will automatically trace the OpenAI API call.

        Args:
            text: Text to generate embedding for
            max_retries: Maximum retry attempts

        Returns:
            Embedding vector as list of floats
        """
        with tracer.start_as_current_span("rag_agent.generate_embedding") as span:
            span.set_attribute("embedding.model", self.embedding_model)
            span.set_attribute("embedding.input_length", len(text))
            span.set_attribute("embedding.max_retries", max_retries)
            
            for attempt in range(max_retries):
                try:
                    # OpenAI call will be automatically traced by auto-instrumentation
                    response = self.openai_client.embeddings.create(
                        model=self.embedding_model,
                        input=text
                    )
                    
                    embedding = response.data[0].embedding
                    span.set_attribute("embedding.output_dimension", len(embedding))
                    span.set_attribute("embedding.attempts_used", attempt + 1)
                    return embedding

                except Exception as e:
                    wait_time = 2 ** attempt  # Exponential backoff
                    logger.warning(
                        f"Embedding generation attempt {attempt + 1} failed: {str(e)}")
                    
                    span.record_exception(e)
                    span.set_attribute("embedding.error", str(e))

                    if attempt < max_retries - 1:
                        logger.info(f"Retrying in {wait_time} seconds...")
                        time.sleep(wait_time)
                    else:
                        logger.error("All embedding generation attempts failed")
                        span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                        raise

    def retrieve_documents(self, query: str, use_hybrid: bool = True) -> List[RetrievedDocument]:
        """
        Retrieve relevant documents from Azure AI Search.

        Args:
            query: User query
            use_hybrid: Whether to use hybrid search (keyword + semantic)

        Returns:
            List of retrieved documents with scores
        """
        with tracer.start_as_current_span("rag_agent.retrieve_documents") as span:
            span.set_attribute("search.query", query)
            span.set_attribute("search.use_hybrid", use_hybrid)
            span.set_attribute("search.top_k", self.top_k_documents)
            span.set_attribute("search.index_name", self.search_index_name)
            
            start_time = time.time()

            try:
                if use_hybrid:
                    # Generate embedding for semantic search
                    query_embedding = self._generate_embedding(query)

                    # Perform hybrid search (keyword + semantic)
                    with tracer.start_as_current_span("azure_search.hybrid_search") as search_span:
                        search_span.set_attribute("search.type", "hybrid")
                        search_span.set_attribute("search.vector_dimension", len(query_embedding))
                        
                        results = self.search_client.search(
                            search_text=query,
                            vector_queries=[{
                                "kind": "vector",
                                "vector": query_embedding,
                                "k_nearest_neighbors": self.top_k_documents * 3,  # Get more candidates
                                "fields": "page_vector"
                            }],
                            top=self.top_k_documents,
                            include_total_count=True
                        )
                else:
                    # Pure keyword search
                    with tracer.start_as_current_span("azure_search.keyword_search") as search_span:
                        search_span.set_attribute("search.type", "keyword")
                        
                        results = self.search_client.search(
                            search_text=query,
                            top=self.top_k_documents,
                            include_total_count=True
                        )

                # Process results
                retrieved_docs = []
                for result in results:
                    doc = RetrievedDocument(
                        docid=result.get("docid", ""),
                        content=result.get("page", ""),
                        score=result.get("@search.score", 0.0),
                        metadata={"source_type": "azure_search"}
                    )
                    retrieved_docs.append(doc)

                retrieval_time = time.time() - start_time
                search_type = "hybrid" if use_hybrid else "keyword"
                
                # Add telemetry attributes
                span.set_attribute("search.documents_retrieved", len(retrieved_docs))
                span.set_attribute("search.retrieval_time", retrieval_time)
                span.set_attribute("search.type_used", search_type)
                
                logger.info(
                    f"Retrieved {len(retrieved_docs)} documents using {search_type} search in {retrieval_time:.3f}s")

                return retrieved_docs

            except Exception as e:
                logger.error(f"Document retrieval failed: {str(e)}")
                span.record_exception(e)
                span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                return []

    def _prepare_context(self, documents: List[RetrievedDocument]) -> str:
        """
        Prepare context string from retrieved documents.

        Args:
            documents: List of retrieved documents

        Returns:
            Formatted context string
        """
        if not documents:
            return "No relevant documents found."

        context_parts = []
        current_length = 0

        for i, doc in enumerate(documents, 1):
            # Format document with source information
            doc_text = f"Document {i} (ID: {doc.docid}, Score: {doc.score:.3f}):\n{doc.content}\n"

            # Check if adding this document would exceed context limit
            if current_length + len(doc_text) > self.max_context_length:
                logger.warning(
                    f"Context limit reached, truncating at {i-1} documents")
                break

            context_parts.append(doc_text)
            current_length += len(doc_text)

        return "\n".join(context_parts)

    def _create_prompt(self, query: str, context: str) -> str:
        """
        Create prompt for the LLM with retrieved context.

        Args:
            query: User query
            context: Retrieved document context

        Returns:
            Formatted prompt string
        """
        prompt = f"""Context from retrieved documents:
{context}

User Question: {query}"""
        return prompt

    def generate_response(
        self,
        query: str,
        context: str,
        max_tokens: int = 1000,
        temperature: float = 0.1
    ) -> Tuple[str, float]:
        """
        Generate response using GPT-4 with retrieved context.
        Auto-instrumentation will automatically trace the OpenAI API call.

        Args:
            query: User query
            context: Retrieved document context
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature

        Returns:
            Tuple of (response_text, generation_time)
        """
        with tracer.start_as_current_span("rag_agent.generate_response") as span:
            span.set_attribute("generation.model", self.chat_model)
            span.set_attribute("generation.max_tokens", max_tokens)
            span.set_attribute("generation.temperature", temperature)
            span.set_attribute("generation.query_length", len(query))
            span.set_attribute("generation.context_length", len(context))
            
            start_time = time.time()

            try:
                prompt = self._create_prompt(query, context)
                span.set_attribute("generation.prompt_length", len(prompt))

                # OpenAI call will be automatically traced by auto-instrumentation
                response = self.openai_client.chat.completions.create(
                    model=self.chat_model,
                    messages=[
                        {"role": "system", "content": """You are an AI assistant helping users find information from a document collection. 
Use the provided context to answer the user's question accurately and comprehensively.

Instructions:
1. Base your answer primarily on the provided context
2. If the context doesn't contain enough information, clearly state this limitation
3. Cite specific documents when referencing information (e.g., "According to Document 1...")
4. Provide a helpful and detailed response
5. If you're unsure about something, express that uncertainty"""},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=0.9
                )

                answer = response.choices[0].message.content
                generation_time = time.time() - start_time
                
                # Add telemetry attributes for our custom metrics
                span.set_attribute("generation.response_length", len(answer))
                span.set_attribute("generation.time", generation_time)
                if response.usage:
                    span.set_attribute("generation.completion_tokens", response.usage.completion_tokens)
                    span.set_attribute("generation.prompt_tokens", response.usage.prompt_tokens)
                    span.set_attribute("generation.total_tokens", response.usage.total_tokens)

                logger.info(
                    f"Generated response in {generation_time:.3f}s using {self.chat_model}")
                return answer, generation_time

            except Exception as e:
                logger.error(f"Response generation failed: {str(e)}")
                span.record_exception(e)
                span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                return f"I apologize, but I encountered an error while generating a response: {str(e)}", 0.0

    def ask(self, query: str, use_hybrid_search: bool = True) -> RAGResponse:
        """
        Main method to ask a question using the RAG pipeline.

        Args:
            query: User question
            use_hybrid_search: Whether to use hybrid search

        Returns:
            RAGResponse with answer, sources, and timing information
        """
        with tracer.start_as_current_span("rag_agent.ask") as span:
            span.set_attribute("rag.query", query)
            span.set_attribute("rag.use_hybrid_search", use_hybrid_search)
            span.set_attribute("rag.top_k_documents", self.top_k_documents)
            span.set_attribute("rag.chat_model", self.chat_model)
            span.set_attribute("rag.embedding_model", self.embedding_model)
            
            start_time = time.time()

            logger.info(f"Processing query: '{query}'")

            try:
                # Step 1: Retrieve relevant documents
                retrieval_start = time.time()
                documents = self.retrieve_documents(
                    query, use_hybrid=use_hybrid_search)
                retrieval_time = time.time() - retrieval_start

                # Step 2: Prepare context from retrieved documents
                context = self._prepare_context(documents)
                span.set_attribute("rag.context_length", len(context))

                # Step 3: Generate response using LLM
                answer, generation_time = self.generate_response(query, context)

                # Prepare sources information
                sources = []
                for doc in documents:
                    sources.append({
                        "docid": doc.docid,
                        "content_preview": doc.content[:200] + "..." if len(doc.content) > 200 else doc.content,
                        "score": doc.score,
                        "metadata": doc.metadata
                    })

                total_time = time.time() - start_time
                
                # Add final telemetry attributes
                span.set_attribute("rag.documents_used", len(documents))
                span.set_attribute("rag.retrieval_time", retrieval_time)
                span.set_attribute("rag.generation_time", generation_time)
                span.set_attribute("rag.total_time", total_time)
                span.set_attribute("rag.answer_length", len(answer))

                logger.info(
                    f"RAG pipeline completed in {total_time:.3f}s (retrieval: {retrieval_time:.3f}s, generation: {generation_time:.3f}s)")

                return RAGResponse(
                    answer=answer,
                    sources=sources,
                    query=query,
                    retrieval_time=retrieval_time,
                    generation_time=generation_time,
                    total_time=total_time
                )

            except Exception as e:
                logger.error(f"RAG pipeline failed: {str(e)}")
                span.record_exception(e)
                span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                return RAGResponse(
                    answer=f"I apologize, but I encountered an error while processing your question: {str(e)}",
                    sources=[],
                    query=query,
                    retrieval_time=0.0,
                    generation_time=0.0,
                    total_time=time.time() - start_time
                )


def main():
    """
    Demo function to test the RAG agent.
    """
    # Configuration from environment variables
    search_service_name = os.getenv(
        "AZURE_SEARCH_SERVICE_NAME", "your-search-service")
    search_index_name = os.getenv(
        "AZURE_SEARCH_INDEX_NAME", "ai-foundry-workshop-index-v1")
    azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    chat_model = os.getenv("AZURE_OPENAI_CHAT_MODEL", "gpt-4.1")

    # Validate configuration
    if search_service_name == "your-search-service":
        print("❌ Error: Please set AZURE_SEARCH_SERVICE_NAME environment variable")
        return

    if not azure_openai_endpoint:
        print("❌ Error: Please set AZURE_OPENAI_ENDPOINT environment variable")
        return

    try:
        # Initialize RAG agent
        print("🤖 Initializing RAG Agent...")
        agent = RAGAgent(
            search_service_name=search_service_name,
            search_index_name=search_index_name,
            azure_openai_endpoint=azure_openai_endpoint,
            chat_model=chat_model,
            top_k_documents=3
        )
        print(f"✅ RAG Agent initialized with model: {chat_model}")

        # Test queries
        test_queries = [
            "What are the key capabilities of Azure AI services?",
            "How does GPT-4 perform compared to previous models?",
            "What are the features of Document Intelligence?",
            "Explain the architecture of modern language models"
        ]

        for query in test_queries:
            print(f"\n" + "="*80)
            print(f"🔍 Query: {query}")
            print("="*80)

            # Get response from RAG agent
            response = agent.ask(query)

            # Display results
            print(f"\n📝 Answer:")
            print(response.answer)

            print(f"\n📊 Performance:")
            print(f"   • Retrieval: {response.retrieval_time:.3f}s")
            print(f"   • Generation: {response.generation_time:.3f}s")
            print(f"   • Total: {response.total_time:.3f}s")

            print(f"\n📚 Sources ({len(response.sources)} documents):")
            for i, source in enumerate(response.sources, 1):
                print(
                    f"   {i}. {source['docid']} (score: {source['score']:.3f})")
                print(f"      Preview: {source['content_preview']}")

        print(f"\n🎉 RAG Agent demo completed!")

    except Exception as e:
        logger.error(f"Demo failed: {str(e)}")
        print(f"❌ Error: {str(e)}")


if __name__ == "__main__":
    main()
