"""
RAG (Retrieval Augmented Generation) Agent using Azure AI Agent Service

This module implements a RAG agent using Azure AI Agent Service SDK that:
1. Creates an agent with access to Azure AI Search tools
2. Takes user questions and retrieves relevant documents
3. Generates responses using the agent service with retrieved context

The agent uses Azure AI Agent Service features:
- Built-in tool integration for Azure AI Search
- Managed conversation threads
- Auto-instrumentation and tracing
- Error handling and retry logic

Run this agent from project root with:
    python -m agents.rag.rag_agent_asrv
"""

import os
import logging
import time
import traceback
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from dotenv import load_dotenv

from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import ConnectionType
from azure.ai.agents.models import (
    ThreadMessage,
    MessageRole,
    RunStatus,
    AzureAISearchTool,
    AzureAISearchQueryType,
    ListSortOrder
)
from azure.identity import DefaultAzureCredential
from azure.monitor.opentelemetry import configure_azure_monitor
from opentelemetry import trace
from opentelemetry.instrumentation.openai import OpenAIInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor

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
class RAGRequest:
    """Request structure for RAG queries."""
    query: str
    max_documents: Optional[int] = 3
    search_type: Optional[str] = "hybrid"  # hybrid, semantic, or keyword


@dataclass
class RAGResponse:
    """Response from the RAG agent using Agent Service."""
    answer: str
    query: str
    thread_id: str
    run_id: str
    total_time: float
    sources: List[Dict[str, Any]]


class RAGAgentService:
    """RAG Agent implementation using Azure AI Agent Service SDK."""

    def __init__(
        self,
        project_endpoint: Optional[str] = None,
        search_index_name: Optional[str] = None,
        chat_model: str = "gpt-4.1",
    ):
        """
        Initialize the RAG Agent using Azure AI Agent Service.

        Args:
            project_endpoint: Azure AI Foundry project endpoint
            search_index_name: Name of the search index
            chat_model: Name of the chat completion model
        """
        self.project_endpoint = project_endpoint or os.getenv("PROJECT_ENDPOINT")
        self.search_index_name = search_index_name or os.getenv("AZURE_SEARCH_INDEX_NAME")
        self.chat_model = chat_model
        
        # Validate required configuration
        if not self.project_endpoint:
            raise ValueError("Azure AI Project endpoint is required (PROJECT_ENDPOINT)")
        if not self.search_index_name:
            raise ValueError("Azure Search index name is required (AZURE_SEARCH_INDEX_NAME)")

        # Initialize Azure AI Project client
        self.credential = DefaultAzureCredential()
        self.project_client = AIProjectClient(
            endpoint=self.project_endpoint,
            credential=self.credential
        )
        
        # Initialize monitoring and tracing
        self._init_monitoring()
        
        self.agent = None
        self._setup_agent()

    def _init_monitoring(self):
        """Initialize Azure Monitor and auto-instrumentation."""
        connection_string = self.project_client.telemetry.get_connection_string()
        logger.info(f"Application Insights connection string: {connection_string}")

        if not connection_string:
            logger.error(
                "Application Insights is not enabled. Enable by going to Tracing in your Azure AI Foundry project.")
            exit()

        # Enable telemetry collection
        configure_azure_monitor(connection_string=connection_string)

        # Initialize auto-instrumentation for OpenAI and requests
        logger.info("Setting up auto-instrumentation for OpenAI and requests...")
        OpenAIInstrumentor().instrument()
        RequestsInstrumentor().instrument()
        logger.info("✅ Auto-instrumentation enabled for OpenAI and requests")

    def _setup_agent(self):
        """Set up the RAG agent with Azure AI Search tool."""
        try:
            with tracer.start_as_current_span("create_rag_agent_span") as span:
                span.set_attribute("chat_model", self.chat_model)
                span.set_attribute("search_index", self.search_index_name)
                span.set_attribute("project_endpoint", self.project_endpoint)
                
                # Get the default Azure AI Search connection
                search_connection_id = self.project_client.connections.get_default(ConnectionType.AZURE_AI_SEARCH).id
                logger.info(f"Using Azure AI Search connection: {search_connection_id}")
                
                # Create Azure AI Search tool with proper configuration
                ai_search_tool = AzureAISearchTool(
                    index_connection_id=search_connection_id,
                    index_name=self.search_index_name,
                    query_type=AzureAISearchQueryType.VECTOR_SIMPLE_HYBRID,  # Use hybrid search for best results
                    top_k=3,  # Default number of documents to retrieve
                    filter=""  # No filter by default
                )
                
                # Define agent instructions for RAG behavior
                instructions = """You are a helpful AI assistant that answers questions using information from a document collection.

Your capabilities:
1. Search for relevant documents using Azure AI Search when users ask questions
2. Analyze the retrieved content to provide accurate, comprehensive answers
3. Cite specific sources when referencing information
4. Clearly indicate when information is not available in the search results

Instructions for answering questions:
1. Always search for relevant documents first using the Azure AI Search tool
2. Base your answers primarily on the retrieved document content, don't use your own knowledge
3. When referencing information, cite the document sources (e.g., "According to document X...")
4. If the search results don't contain sufficient information, clearly state this limitation
5. Provide helpful and detailed responses when possible
6. Express uncertainty when you're not sure about something based on the available documents

Search parameters to use:
- Use the configured search index for document retrieval
- Retrieve multiple relevant documents to provide comprehensive answers
- Prefer hybrid search for best results combining keyword and semantic matching"""

                # Create the RAG agent with Azure AI Search tool
                self.agent = self.project_client.agents.create_agent(
                    model=self.chat_model,
                    name="RAGAgent001",
                    description="AI agent for retrieval-augmented generation using Azure AI Search",
                    instructions=instructions,
                    tools=ai_search_tool.definitions,
                    tool_resources=ai_search_tool.resources
                )
                
                logger.info(f"RAG Agent created with ID: {self.agent.id}")
                span.set_attribute("agent_id", self.agent.id)
                span.set_attribute("search_connection_id", search_connection_id)

        except Exception as e:
            logger.error(f"Failed to setup RAG agent: {e}")
            logger.error(f"Full exception trace:\n{traceback.format_exc()}")
            raise

    def ask(self, query: str, max_documents: int = 3) -> RAGResponse:
        """
        Ask a question using the RAG agent service.

        Args:
            query: User question
            max_documents: Maximum number of documents to retrieve

        Returns:
            RAGResponse with answer and metadata
        """
        try:
            logger.info(f"Processing query: '{query}'")
            
            with tracer.start_as_current_span("rag_agent_service.ask") as main_span:
                main_span.set_attribute("rag.query", query)
                main_span.set_attribute("rag.max_documents", max_documents)
                main_span.set_attribute("rag.chat_model", self.chat_model)
                main_span.set_attribute("rag.search_index", self.search_index_name)
                
                start_time = time.time()
                
                with tracer.start_as_current_span("thread_creation_span") as thread_span:
                    # Create a thread for this conversation
                    thread = self.project_client.agents.threads.create()
                    thread_span.set_attribute("thread_id", thread.id)
                    logger.info(f"Created thread: {thread.id}")

                with tracer.start_as_current_span("rag_query_span") as query_span:
                
                    # Send the message to the agent
                    message = self.project_client.agents.messages.create(
                        thread_id=thread.id,
                        role=MessageRole.USER,
                        content=query
                    )
                    query_span.set_attribute("user_message", query)
                    
                    # Run the agent
                    run = self.project_client.agents.runs.create_and_process(
                        thread_id=thread.id,
                        agent_id=self.agent.id
                    )
                    query_span.set_attribute("agent_id", self.agent.id)
                    query_span.set_attribute("run_id", run.id)
                    
                    # Wait for completion
                    completed_run = self._wait_for_run_completion(thread.id, run.id)
                    
                    total_time = time.time() - start_time
                    
                    if completed_run.status == RunStatus.COMPLETED:
                        query_span.set_attribute("run_status", "COMPLETED")
                        
                        # Set tool call attributes for observability
                        self._set_tool_call_attributes(query_span, thread.id, run.id)
                        
                        # Retrieve the agent's response
                        messages = self.project_client.agents.messages.list(thread_id=thread.id)
                        
                        # Process the response
                        answer, sources = self._process_agent_response(messages)
                        
                        query_span.set_attribute("answer_length", len(answer))
                        query_span.set_attribute("sources_count", len(sources))
                        
                        logger.info(f"RAG query completed in {total_time:.3f}s")
                        
                        return RAGResponse(
                            answer=answer,
                            query=query,
                            thread_id=thread.id,
                            run_id=run.id,
                            total_time=total_time,
                            sources=sources
                        )
                    else:
                        query_span.set_attribute("run_status", "FAILED")
                        error_msg = f"Agent run failed with status: {completed_run.status}"
                        if hasattr(completed_run, 'last_error') and completed_run.last_error:
                            error_msg += f"\nLast error: {completed_run.last_error}"
                        logger.error(error_msg)
                        raise RuntimeError(error_msg)

        except Exception as e:
            logger.error(f"Error during RAG query: {e}")
            logger.error(f"Full exception trace:\n{traceback.format_exc()}")
            raise

    def _wait_for_run_completion(self, thread_id: str, run_id: str, timeout: int = 300) -> Any:
        """
        Wait for the agent run to complete.

        Args:
            thread_id: Thread identifier
            run_id: Run identifier
            timeout: Maximum wait time in seconds

        Returns:
            Completed run object
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            run = self.project_client.agents.runs.get(thread_id=thread_id, run_id=run_id)

            if run.status in [RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CANCELLED]:
                return run

            logger.info(f"Run status: {run.status}. Waiting...")
            time.sleep(5)

        raise TimeoutError(f"Run did not complete within {timeout} seconds")

    def _set_tool_call_attributes(self, span, thread_id: str, run_id: str):
        """Set OpenTelemetry attributes with tool call information."""
        try:
            run_steps = self.project_client.agents.run_steps.list(thread_id=thread_id, run_id=run_id)
            tool_call_count = 0
            
            for step in run_steps:
                step_details = step.step_details if hasattr(step, 'step_details') else {}
                tool_calls = step_details.tool_calls if hasattr(step_details, 'tool_calls') else []
                
                for call in tool_calls:
                    tool_call_count += 1
                    span.set_attribute(f"tool_call_{tool_call_count}.type", getattr(call, 'type', 'unknown'))
                    span.set_attribute(f"tool_call_{tool_call_count}.id", getattr(call, 'id', 'unknown'))
                    
                    if hasattr(call, 'azure_ai_search'):
                        azure_search = call.azure_ai_search
                        if hasattr(azure_search, 'input'):
                            span.set_attribute(f"tool_call_{tool_call_count}.search_input", str(azure_search.input))
                        if hasattr(azure_search, 'output'):
                            span.set_attribute(f"tool_call_{tool_call_count}.search_output", str(azure_search.output))
            
            span.set_attribute("tool_calls_total", tool_call_count)
            
        except Exception as e:
            span.set_attribute("tool_calls_error", str(e))

    def _process_agent_response(self, messages: List[ThreadMessage]) -> tuple[str, List[Dict[str, Any]]]:
        """
        Process the agent's response to extract answer and sources.

        Args:
            messages: Thread messages from the agent

        Returns:
            Tuple of (answer_text, sources_list)
        """
        try:
            # Get the latest assistant message
            assistant_messages = [
                msg for msg in messages if msg.role == MessageRole.AGENT
            ]
            
            if not assistant_messages:
                raise ValueError("No assistant response found")

            latest_response = assistant_messages[-1]
            
            # Extract text content and handle citations
            answer = ""
            sources = []
            
            if hasattr(latest_response, 'content') and latest_response.content:
                for content_item in latest_response.content:
                    if hasattr(content_item, 'text') and hasattr(content_item.text, 'value'):
                        answer += content_item.text.value
            
            # Process URL citations if available (following the example pattern)
            if hasattr(latest_response, 'url_citation_annotations') and latest_response.url_citation_annotations:
                placeholder_annotations = {
                    annotation.text: f" [see {annotation.url_citation.title}] ({annotation.url_citation.url})"
                    for annotation in latest_response.url_citation_annotations
                }
                
                # Replace citation placeholders in the answer
                for placeholder, citation in placeholder_annotations.items():
                    answer = answer.replace(placeholder, citation)
                
                # Extract source information
                for annotation in latest_response.url_citation_annotations:
                    sources.append({
                        "title": annotation.url_citation.title,
                        "url": annotation.url_citation.url,
                        "placeholder": annotation.text
                    })
            
            return answer, sources

        except Exception as e:
            logger.error(f"Error processing agent response: {e}")
            raise

    def cleanup(self):
        """Clean up resources."""
        try:
            if self.agent:
                self.project_client.agents.delete_agent(self.agent.id)
                logger.info("RAG Agent cleaned up successfully")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")


def main():
    """
    Demo function to test the RAG Agent Service.
    """
    try:
        # Initialize the RAG agent
        print("🤖 Initializing RAG Agent Service...")
        agent = RAGAgentService()
        print(f"✅ RAG Agent Service initialized")

        # Test queries
        test_queries = [
            "What's Document Intelligence layout model?",
            "What's Document Intelligence read model and what is diffrence between it and layout model?",
        ]

        for query in test_queries:
            print(f"\n" + "="*80)
            print(f"🔍 Query: {query}")
            print("="*80)

            # Get response from RAG agent service
            response = agent.ask(query, max_documents=3)

            # Display results
            print(f"\n📝 Answer:")
            print(response.answer)

            print(f"\n📊 Performance:")
            print(f"   • Total time: {response.total_time:.3f}s")
            print(f"   • Thread ID: {response.thread_id}")
            print(f"   • Run ID: {response.run_id}")

            if response.sources:
                print(f"\n📚 Sources ({len(response.sources)} documents):")
                for i, source in enumerate(response.sources, 1):
                    print(f"   {i}. {source}")

        print(f"\n🎉 RAG Agent Service demo completed!")

    except Exception as e:
        logger.error(f"Demo failed: {str(e)}")
        print(f"❌ Error: {str(e)}")
    finally:
        if 'agent' in locals():
            agent.cleanup()


if __name__ == "__main__":
    main()
