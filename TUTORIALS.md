# Tutorials and Walkthroughs

This document provides step-by-step tutorials for working with the Azure AI Foundry Workshop. Each tutorial includes detailed explanations, code examples, and expected outcomes.

## Table of Contents

- [Tutorial 1: Setting Up Your First RAG Pipeline](#tutorial-1-setting-up-your-first-rag-pipeline)
- [Tutorial 2: Document Processing and Indexing](#tutorial-2-document-processing-and-indexing)
- [Tutorial 3: Building and Testing a RAG Agent](#tutorial-3-building-and-testing-a-rag-agent)
- [Tutorial 4: Evaluation and Performance Optimization](#tutorial-4-evaluation-and-performance-optimization)
- [Tutorial 5: Advanced Tracing and Observability](#tutorial-5-advanced-tracing-and-observability)
- [Tutorial 6: Custom Evaluators and Metrics](#tutorial-6-custom-evaluators-and-metrics)
- [Tutorial 7: Production Deployment and Monitoring](#tutorial-7-production-deployment-and-monitoring)

## Tutorial 1: Setting Up Your First RAG Pipeline

### Prerequisites
- Azure subscription with AI services access
- Python 3.12+ installed
- Basic familiarity with Azure services

### Learning Objectives
- Understand RAG architecture components
- Set up Azure AI services
- Configure the development environment
- Run your first RAG query

### Step 1: Create Azure Resources

1. **Login to Azure**:
   ```bash
   az login
   az account set --subscription "your-subscription-id"
   ```

2. **Create Resource Group**:
   ```bash
   az group create --name rg-rag-tutorial --location eastus2
   ```

3. **Create Azure AI Foundry Hub**:
   - Navigate to [Azure Portal](https://portal.azure.com)
   - Search for "Azure AI Foundry"
   - Click "Create" and fill in:
     - Subscription: Your subscription
     - Resource group: `rg-rag-tutorial`
     - Name: `aih-rag-tutorial`
     - Region: `East US 2`
   - Click "Review + Create"

4. **Create AI Search Service**:
   ```bash
   az search service create \
     --name search-rag-tutorial \
     --resource-group rg-rag-tutorial \
     --sku Standard \
     --location eastus2
   ```

5. **Create OpenAI Service and Deploy Models**:
   ```bash
   # Create OpenAI service
   az cognitiveservices account create \
     --name openai-rag-tutorial \
     --resource-group rg-rag-tutorial \
     --kind OpenAI \
     --sku S0 \
     --location eastus2

   # Deploy GPT-4o model
   az cognitiveservices account deployment create \
     --name openai-rag-tutorial \
     --resource-group rg-rag-tutorial \
     --deployment-name gpt-4o \
     --model-name gpt-4o \
     --model-version "2024-08-06" \
     --model-format OpenAI \
     --sku-capacity 10

   # Deploy embedding model
   az cognitiveservices account deployment create \
     --name openai-rag-tutorial \
     --resource-group rg-rag-tutorial \
     --deployment-name text-embedding-3-small \
     --model-name text-embedding-3-small \
     --model-version "1" \
     --model-format OpenAI \
     --sku-capacity 10
   ```

6. **Create Document Intelligence Service**:
   ```bash
   az cognitiveservices account create \
     --name docintel-rag-tutorial \
     --resource-group rg-rag-tutorial \
     --kind FormRecognizer \
     --sku S0 \
     --location eastus2
   ```

### Step 2: Set Up Development Environment

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/vladfeigin/aifoundry-workshop.git
   cd aifoundry-workshop
   ```

2. **Install Dependencies**:
   ```bash
   pip install azure-ai-documentintelligence azure-search-documents azure-identity python-dotenv rich openai azure-ai-projects azure-monitor-opentelemetry opentelemetry-sdk azure-ai-evaluation
   ```

3. **Configure Environment Variables**:
   ```bash
   cp .env.template .env
   ```

   Edit `.env` file with your service details:
   ```env
   # Get these values from Azure Portal
   AZURE_SEARCH_SERVICE_NAME=search-rag-tutorial
   AZURE_SEARCH_INDEX_NAME=rag-tutorial-index
   AZURE_OPENAI_ENDPOINT=https://openai-rag-tutorial.openai.azure.com/
   AZURE_DOCINTEL_ENDPOINT=https://docintel-rag-tutorial.cognitiveservices.azure.com/
   PROJECT_ENDPOINT=https://your-project.services.ai.azure.com/api/projects/your-project

   # Get API keys from Azure Portal > Your Service > Keys and Endpoint
   AZURE_SEARCH_API_KEY=your-search-api-key
   AZURE_OPENAI_API_KEY=your-openai-api-key
   AZURE_DOCINTEL_KEY=your-docintel-api-key

   # Model configurations
   AZURE_OPENAI_CHAT_MODEL=gpt-4o
   AZURE_OPENAI_EMBEDDING_MODEL=text-embedding-3-small
   AZURE_OPENAI_API_VERSION=2024-12-01-preview
   ```

### Step 3: Create Your First Search Index

1. **Run the Index Creation Script**:
   ```bash
   python -m aisearch.create_search_index \
     --search-service search-rag-tutorial \
     --index-name rag-tutorial-index \
     --use-api-key
   ```

   **Expected Output**:
   ```
   Creating search index 'rag-tutorial-index' on service 'search-rag-tutorial'...
   Index created successfully!
   Index schema:
   - docid (String, key=True)
   - page (String, searchable=True)
   - page_vector (Collection[Single], searchable=True, dimensions=1536)
   ```

2. **Verify Index Creation**:
   - Go to Azure Portal > AI Search Service > Indexes
   - You should see `rag-tutorial-index` listed

### Step 4: Process Your First Document

1. **Convert a PDF to Markdown**:
   ```bash
   python -m docintel.pdf-2-md \
     ./docintel/data/GPT-4-Technical-Report.pdf \
     ./docintel/data/GPT-4-Technical-Report.md
   ```

   **Expected Output**:
   ```
   Processing document: ./docintel/data/GPT-4-Technical-Report.pdf
   Converting PDF to Markdown using Azure Document Intelligence...
   ✓ Document processed successfully
   ✓ 12 pages extracted
   ✓ 3 tables converted
   Output saved to: ./docintel/data/GPT-4-Technical-Report.md
   ```

2. **Ingest Documents into Search Index**:
   ```bash
   python -m aisearch.ingest_documents \
     --search-service search-rag-tutorial \
     --index-name rag-tutorial-index
   ```

   **Expected Output**:
   ```
   Processing documents from: ./docintel/data
   Found 2 markdown files to process
   
   Processing: GPT-4-Technical-Report.md
   ✓ Split into 15 chunks
   ✓ Generated embeddings
   ✓ Uploaded to search index
   
   Processing: document-intelligence-4.md  
   ✓ Split into 8 chunks
   ✓ Generated embeddings
   ✓ Uploaded to search index
   
   Total documents indexed: 23
   ```

### Step 5: Test Your RAG Agent

1. **Run the RAG Agent**:
   ```bash
   python -m agents.rag.rag_agent
   ```

   **Expected Output**:
   ```
   Initializing RAG Agent...
   ✓ Azure AI Foundry project client created
   ✓ Agent client initialized
   ✓ Search connection configured
   
   Testing RAG Agent with sample queries...
   
   Query: What is GPT-4?
   
   Retrieved Documents:
   1. GPT-4-Technical-Report_page_1_abc123 (Score: 0.87)
      Preview: GPT-4 Technical Report Abstract We report the development of GPT-4...
   
   2. GPT-4-Technical-Report_page_2_def456 (Score: 0.82)
      Preview: GPT-4 is a large-scale, multimodal model which can accept image and text inputs...
   
   Agent Response:
   GPT-4 is a large-scale, multimodal model developed by OpenAI that can accept both image and text inputs and produce text outputs. It represents a significant advancement in artificial intelligence, demonstrating human-level performance on various professional and academic benchmarks.
   
   Sources: 2 documents
   Response time: 2.3 seconds
   Thread ID: thread_abc123
   ```

**Congratulations!** You've successfully set up your first RAG pipeline. The system can now:
- Convert PDFs to searchable text
- Index documents with semantic search capabilities
- Answer questions using retrieved context

---

## Tutorial 2: Document Processing and Indexing

### Learning Objectives
- Master document processing workflows
- Understand indexing strategies
- Optimize search performance
- Handle different document types

### Step 1: Understanding Document Processing Pipeline

The document processing pipeline follows these stages:

```
PDF Input → Structure Extraction → Markdown Conversion → Chunking → Embedding → Indexing
```

Let's walk through each stage:

### Step 2: Advanced PDF Processing

1. **Process Multiple Document Types**:
   ```bash
   # Create a test documents directory
   mkdir test-documents
   
   # Download sample documents (or use your own)
   curl -o test-documents/sample1.pdf "https://example.com/document1.pdf"
   curl -o test-documents/sample2.pdf "https://example.com/document2.pdf"
   ```

2. **Batch Process Documents**:
   ```python
   # batch_process.py
   import os
   import subprocess
   from pathlib import Path
   
   def batch_process_pdfs(input_dir: str, output_dir: str):
       """Process all PDFs in a directory"""
       input_path = Path(input_dir)
       output_path = Path(output_dir)
       output_path.mkdir(exist_ok=True)
       
       for pdf_file in input_path.glob("*.pdf"):
           output_file = output_path / f"{pdf_file.stem}.md"
           
           print(f"Processing: {pdf_file.name}")
           result = subprocess.run([
               "python", "-m", "docintel.pdf-2-md",
               str(pdf_file),
               str(output_file)
           ], capture_output=True, text=True)
           
           if result.returncode == 0:
               print(f"✓ Converted: {output_file.name}")
           else:
               print(f"✗ Failed: {pdf_file.name} - {result.stderr}")
   
   if __name__ == "__main__":
       batch_process_pdfs("test-documents", "processed-documents")
   ```

   Run the batch processor:
   ```bash
   python batch_process.py
   ```

### Step 3: Understanding Document Chunking

1. **Examine Chunking Strategy**:
   ```python
   # chunking_demo.py
   import re
   from typing import List
   
   def analyze_document_structure(markdown_file: str):
       """Analyze how a document gets chunked"""
       with open(markdown_file, 'r', encoding='utf-8') as f:
           content = f.read()
       
       # Find page delimiters
       page_pattern = r'<!---- Page (\d+) -+>'
       pages = re.split(page_pattern, content)
       
       print(f"Document: {markdown_file}")
       print(f"Total pages found: {(len(pages) - 1) // 2}")
       
       # Analyze page content
       for i in range(1, len(pages), 2):
           page_num = pages[i]
           page_content = pages[i + 1] if i + 1 < len(pages) else ""
           
           print(f"\nPage {page_num}:")
           print(f"  Content length: {len(page_content)} characters")
           print(f"  Lines: {len(page_content.splitlines())}")
           print(f"  Tables: {page_content.count('|')}")
           print(f"  Preview: {page_content[:100]}...")
   
   # Analyze a processed document
   analyze_document_structure("./docintel/data/GPT-4-Technical-Report.md")
   ```

2. **Custom Chunking Strategy**:
   ```python
   # custom_chunking.py
   from typing import List, Dict
   import hashlib
   
   class DocumentChunker:
       def __init__(self, max_chunk_size: int = 8000, overlap: int = 200):
           self.max_chunk_size = max_chunk_size
           self.overlap = overlap
       
       def chunk_by_paragraphs(self, content: str, filename: str) -> List[Dict]:
           """Chunk document by paragraphs with overlap"""
           paragraphs = content.split('\n\n')
           chunks = []
           current_chunk = ""
           chunk_num = 1
           
           for paragraph in paragraphs:
               if len(current_chunk) + len(paragraph) > self.max_chunk_size:
                   if current_chunk:
                       chunks.append(self._create_chunk(
                           current_chunk, filename, chunk_num
                       ))
                       chunk_num += 1
                       
                       # Add overlap from end of current chunk
                       overlap_text = current_chunk[-self.overlap:]
                       current_chunk = overlap_text + paragraph
                   else:
                       current_chunk = paragraph
               else:
                   current_chunk += "\n\n" + paragraph
           
           # Add final chunk
           if current_chunk:
               chunks.append(self._create_chunk(
                   current_chunk, filename, chunk_num
               ))
           
           return chunks
       
       def _create_chunk(self, content: str, filename: str, chunk_num: int) -> Dict:
           """Create a chunk document for indexing"""
           content_hash = hashlib.md5(content.encode()).hexdigest()[:8]
           doc_id = f"{filename}_chunk_{chunk_num}_{content_hash}"
           
           return {
               "docid": doc_id,
               "page": content.strip(),
               "filename": filename,
               "chunk_number": chunk_num,
               "content_length": len(content)
           }
   
   # Test the chunker
   chunker = DocumentChunker(max_chunk_size=2000, overlap=100)
   with open("./docintel/data/GPT-4-Technical-Report.md", 'r') as f:
       content = f.read()
   
   chunks = chunker.chunk_by_paragraphs(content, "GPT-4-Technical-Report")
   print(f"Created {len(chunks)} chunks")
   for i, chunk in enumerate(chunks[:3]):  # Show first 3 chunks
       print(f"\nChunk {i+1}:")
       print(f"  ID: {chunk['docid']}")
       print(f"  Length: {chunk['content_length']} chars")
       print(f"  Preview: {chunk['page'][:100]}...")
   ```

### Step 4: Optimizing Search Index

1. **Create Index with Custom Configuration**:
   ```python
   # optimized_index.py
   from azure.search.documents.indexes import SearchIndexClient
   from azure.search.documents.indexes.models import *
   from azure.identity import DefaultAzureCredential
   
   def create_optimized_index(service_name: str, index_name: str):
       """Create an optimized search index"""
       endpoint = f"https://{service_name}.search.windows.net"
       credential = DefaultAzureCredential()
       
       client = SearchIndexClient(endpoint=endpoint, credential=credential)
       
       # Define fields with custom analyzers
       fields = [
           SimpleField(name="docid", type=SearchFieldDataType.String, key=True),
           SearchableField(
               name="page", 
               type=SearchFieldDataType.String,
               analyzer_name="en.microsoft"  # Use Microsoft English analyzer
           ),
           SearchableField(
               name="title",
               type=SearchFieldDataType.String,
               analyzer_name="keyword"  # Exact match for titles
           ),
           VectorSearchField(
               name="page_vector",
               type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
               searchable=True,
               vector_search_dimensions=1536,
               vector_search_profile_name="default-vector-profile"
           ),
           SimpleField(name="filename", type=SearchFieldDataType.String, filterable=True),
           SimpleField(name="page_number", type=SearchFieldDataType.Int32, filterable=True),
           SimpleField(name="content_length", type=SearchFieldDataType.Int32, filterable=True)
       ]
       
       # Configure vector search
       vector_search = VectorSearch(
           profiles=[
               VectorSearchProfile(
                   name="default-vector-profile",
                   algorithm_configuration_name="default-hnsw"
               )
           ],
           algorithms=[
               HnswAlgorithmConfiguration(
                   name="default-hnsw",
                   parameters=HnswParameters(
                       m=8,  # Increased connections for better recall
                       ef_construction=400,
                       ef_search=500,
                       metric=VectorSearchAlgorithmMetric.COSINE
                   )
               )
           ]
       )
       
       # Create index
       index = SearchIndex(
           name=index_name,
           fields=fields,
           vector_search=vector_search,
           scoring_profiles=[
               ScoringProfile(
                   name="content-boost",
                   text_weights=TextWeights(weights={"page": 2.0, "title": 3.0}),
                   functions=[
                       FreshnessScoringFunction(
                           field_name="last_modified",
                           boost=1.5,
                           parameters=FreshnessScoringParameters(boosting_duration="P30D")
                       )
                   ]
               )
           ]
       )
       
       result = client.create_or_update_index(index)
       print(f"Optimized index '{index_name}' created successfully")
       return result
   
   # Create the optimized index
   create_optimized_index("search-rag-tutorial", "optimized-rag-index")
   ```

### Step 5: Testing Search Performance

1. **Search Performance Benchmark**:
   ```python
   # search_benchmark.py
   import time
   import asyncio
   from azure.search.documents import SearchClient
   from azure.identity import DefaultAzureCredential
   
   class SearchBenchmark:
       def __init__(self, service_name: str, index_name: str):
           endpoint = f"https://{service_name}.search.windows.net"
           self.client = SearchClient(
               endpoint=endpoint,
               index_name=index_name,
               credential=DefaultAzureCredential()
           )
       
       def benchmark_search_types(self, query: str, iterations: int = 10):
           """Benchmark different search types"""
           results = {}
           
           # 1. Keyword search
           times = []
           for _ in range(iterations):
               start = time.time()
               results_list = list(self.client.search(
                   search_text=query,
                   top=5
               ))
               times.append(time.time() - start)
           
           results["keyword"] = {
               "avg_time": sum(times) / len(times),
               "result_count": len(results_list)
           }
           
           # 2. Semantic search (if available)
           times = []
           for _ in range(iterations):
               start = time.time()
               results_list = list(self.client.search(
                   search_text=query,
                   query_type="semantic",
                   top=5
               ))
               times.append(time.time() - start)
           
           results["semantic"] = {
               "avg_time": sum(times) / len(times),
               "result_count": len(results_list)
           }
           
           return results
   
   # Run benchmark
   benchmark = SearchBenchmark("search-rag-tutorial", "rag-tutorial-index")
   test_queries = [
       "What is GPT-4?",
       "machine learning capabilities",
       "document intelligence features",
       "Azure AI services"
   ]
   
   for query in test_queries:
       print(f"\nBenchmarking query: '{query}'")
       results = benchmark.benchmark_search_types(query)
       
       for search_type, metrics in results.items():
           print(f"  {search_type.capitalize()} search:")
           print(f"    Average time: {metrics['avg_time']:.3f}s")
           print(f"    Result count: {metrics['result_count']}")
   ```

**Key Takeaways**:
- Document chunking strategy affects search quality
- Index configuration impacts performance
- Different search types have different use cases
- Regular performance monitoring is essential

---

## Tutorial 3: Building and Testing a RAG Agent

### Learning Objectives
- Build a custom RAG agent from scratch
- Implement conversation memory
- Add tool integration
- Test agent responses

### Step 1: Understanding RAG Agent Architecture

The RAG agent follows this workflow:

```
User Query → Query Processing → Document Retrieval → Context Formation → LLM Generation → Response
```

### Step 2: Build a Custom RAG Agent

1. **Create Basic RAG Agent**:
   ```python
   # custom_rag_agent.py
   import os
   import time
   from typing import List, Dict, Any, Optional
   from dataclasses import dataclass
   from azure.search.documents import SearchClient
   from azure.identity import DefaultAzureCredential
   from openai import AzureOpenAI
   
   @dataclass
   class RAGResponse:
       answer: str
       sources: List[Dict[str, Any]]
       response_time: float
       confidence_score: Optional[float] = None
   
   class CustomRAGAgent:
       def __init__(
           self,
           search_service: str,
           search_index: str,
           openai_endpoint: str,
           openai_key: str,
           chat_model: str = "gpt-4o",
           embedding_model: str = "text-embedding-3-small"
       ):
           # Initialize search client
           search_endpoint = f"https://{search_service}.search.windows.net"
           self.search_client = SearchClient(
               endpoint=search_endpoint,
               index_name=search_index,
               credential=DefaultAzureCredential()
           )
           
           # Initialize OpenAI client
           self.openai_client = AzureOpenAI(
               azure_endpoint=openai_endpoint,
               api_key=openai_key,
               api_version="2024-12-01-preview"
           )
           
           self.chat_model = chat_model
           self.embedding_model = embedding_model
           self.conversation_history = []
       
       def search_documents(self, query: str, top_k: int = 5) -> List[Dict]:
           """Search for relevant documents"""
           # Generate query embedding
           embedding_response = self.openai_client.embeddings.create(
               input=query,
               model=self.embedding_model
           )
           query_vector = embedding_response.data[0].embedding
           
           # Perform hybrid search
           results = self.search_client.search(
               search_text=query,
               vector_queries=[{
                   "kind": "vector",
                   "vector": query_vector,
                   "fields": "page_vector",
                   "k": top_k
               }],
               top=top_k,
               select=["docid", "page"]
           )
           
           return [
               {
                   "docid": doc["docid"],
                   "content": doc["page"],
                   "score": doc["@search.score"]
               }
               for doc in results
           ]
       
       def format_context(self, documents: List[Dict]) -> str:
           """Format retrieved documents into context"""
           context_parts = []
           for i, doc in enumerate(documents, 1):
               context_parts.append(f"Document {i} (ID: {doc['docid']}):\n{doc['content']}")
           
           return "\n\n".join(context_parts)
       
       def generate_response(self, query: str, context: str) -> str:
           """Generate response using LLM"""
           system_prompt = """You are a helpful AI assistant that answers questions based on the provided context.
           
   Rules:
   1. Answer only based on the information in the context
   2. If the context doesn't contain enough information, say so
   3. Cite specific document IDs when referencing information
   4. Be concise but comprehensive
   5. If asked about something not in the context, politely decline
   
   Context:
   {context}"""
           
           messages = [
               {"role": "system", "content": system_prompt.format(context=context)},
               {"role": "user", "content": query}
           ]
           
           # Add conversation history
           for msg in self.conversation_history[-10:]:  # Last 10 messages
               messages.insert(-1, msg)
           
           response = self.openai_client.chat.completions.create(
               model=self.chat_model,
               messages=messages,
               temperature=0.1,
               max_tokens=1000
           )
           
           return response.choices[0].message.content
       
       def ask(self, query: str) -> RAGResponse:
           """Process a query and return response"""
           start_time = time.time()
           
           # Step 1: Search for relevant documents
           documents = self.search_documents(query)
           
           # Step 2: Format context
           context = self.format_context(documents)
           
           # Step 3: Generate response
           answer = self.generate_response(query, context)
           
           # Step 4: Update conversation history
           self.conversation_history.extend([
               {"role": "user", "content": query},
               {"role": "assistant", "content": answer}
           ])
           
           response_time = time.time() - start_time
           
           return RAGResponse(
               answer=answer,
               sources=[{"docid": doc["docid"], "score": doc["score"]} for doc in documents],
               response_time=response_time
           )
   ```

2. **Test the Custom Agent**:
   ```python
   # test_custom_agent.py
   from custom_rag_agent import CustomRAGAgent
   import os
   from dotenv import load_dotenv
   
   load_dotenv()
   
   # Initialize agent
   agent = CustomRAGAgent(
       search_service=os.getenv("AZURE_SEARCH_SERVICE_NAME"),
       search_index=os.getenv("AZURE_SEARCH_INDEX_NAME"),
       openai_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
       openai_key=os.getenv("AZURE_OPENAI_API_KEY")
   )
   
   # Test queries
   test_queries = [
       "What is GPT-4?",
       "How does document intelligence work?",
       "What are the capabilities of Azure AI services?"
   ]
   
   print("Testing Custom RAG Agent")
   print("=" * 50)
   
   for query in test_queries:
       print(f"\nQuery: {query}")
       response = agent.ask(query)
       
       print(f"Answer: {response.answer}")
       print(f"Sources: {len(response.sources)} documents")
       print(f"Response time: {response.response_time:.2f}s")
       print("-" * 30)
   ```

### Step 3: Add Advanced Features

1. **Conversation Memory**:
   ```python
   # conversation_memory.py
   from typing import List, Dict, Any
   import json
   from datetime import datetime
   
   class ConversationMemory:
       def __init__(self, max_history: int = 20):
           self.max_history = max_history
           self.conversations = []
           self.context_summary = ""
       
       def add_interaction(self, query: str, response: str, sources: List[Dict]):
           """Add a new interaction to memory"""
           interaction = {
               "timestamp": datetime.now().isoformat(),
               "query": query,
               "response": response,
               "sources": sources
           }
           
           self.conversations.append(interaction)
           
           # Keep only recent conversations
           if len(self.conversations) > self.max_history:
               self.conversations = self.conversations[-self.max_history:]
       
       def get_conversation_context(self) -> str:
           """Get formatted conversation context"""
           if not self.conversations:
               return ""
           
           context_parts = ["Previous conversation:"]
           for conv in self.conversations[-5:]:  # Last 5 interactions
               context_parts.append(f"User: {conv['query']}")
               context_parts.append(f"Assistant: {conv['response'][:200]}...")
           
           return "\n".join(context_parts)
       
       def save_to_file(self, filename: str):
           """Save conversation history to file"""
           with open(filename, 'w') as f:
               json.dump(self.conversations, f, indent=2)
       
       def load_from_file(self, filename: str):
           """Load conversation history from file"""
           try:
               with open(filename, 'r') as f:
                   self.conversations = json.load(f)
           except FileNotFoundError:
               self.conversations = []
   ```

2. **Tool Integration**:
   ```python
   # rag_tools.py
   import re
   import json
   from typing import Dict, Any, List
   from azure.search.documents import SearchClient
   
   class RAGTools:
       def __init__(self, search_client: SearchClient):
           self.search_client = search_client
       
       def search_by_filter(self, query: str, filename: str = None, page_range: tuple = None) -> List[Dict]:
           """Search with filters"""
           filter_conditions = []
           
           if filename:
               filter_conditions.append(f"filename eq '{filename}'")
           
           if page_range:
               start, end = page_range
               filter_conditions.append(f"page_number ge {start} and page_number le {end}")
           
           filter_str = " and ".join(filter_conditions) if filter_conditions else None
           
           results = self.search_client.search(
               search_text=query,
               filter=filter_str,
               top=10
           )
           
           return list(results)
       
       def get_document_summary(self, docid: str) -> Dict[str, Any]:
           """Get summary information about a document"""
           results = self.search_client.search(
               search_text="*",
               filter=f"docid eq '{docid}'",
               select=["docid", "page", "filename", "page_number"]
           )
           
           doc = next(results, None)
           if doc:
               return {
                   "docid": doc["docid"],
                   "content_preview": doc["page"][:200] + "...",
                   "filename": doc.get("filename", "Unknown"),
                   "page_number": doc.get("page_number", "Unknown"),
                   "word_count": len(doc["page"].split())
               }
           return {}
       
       def extract_entities(self, text: str) -> List[str]:
           """Extract named entities from text"""
           # Simple regex-based entity extraction
           entities = []
           
           # Extract potential organization names (capitalized words)
           orgs = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
           entities.extend([org for org in orgs if len(org.split()) <= 3])
           
           # Extract numbers and percentages
           numbers = re.findall(r'\d+(?:\.\d+)?%?', text)
           entities.extend(numbers)
           
           return list(set(entities))
   ```

### Step 4: Interactive Testing Interface

1. **Create Interactive Chat Interface**:
   ```python
   # interactive_chat.py
   import os
   from custom_rag_agent import CustomRAGAgent
   from conversation_memory import ConversationMemory
   from rag_tools import RAGTools
   from dotenv import load_dotenv
   
   load_dotenv()
   
   class InteractiveRAGChat:
       def __init__(self):
           self.agent = CustomRAGAgent(
               search_service=os.getenv("AZURE_SEARCH_SERVICE_NAME"),
               search_index=os.getenv("AZURE_SEARCH_INDEX_NAME"),
               openai_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
               openai_key=os.getenv("AZURE_OPENAI_API_KEY")
           )
           self.memory = ConversationMemory()
           self.tools = RAGTools(self.agent.search_client)
       
       def print_welcome(self):
           print("🤖 Interactive RAG Agent Chat")
           print("=" * 50)
           print("Commands:")
           print("  /help - Show this help")
           print("  /history - Show conversation history")
           print("  /search <query> - Search documents")
           print("  /info <docid> - Get document info")
           print("  /quit - Exit chat")
           print("=" * 50)
       
       def handle_command(self, command: str) -> bool:
           """Handle special commands. Returns True if command was handled."""
           if command.startswith("/help"):
               self.print_welcome()
               return True
           
           elif command.startswith("/history"):
               print("\n📜 Conversation History:")
               for i, conv in enumerate(self.memory.conversations[-5:], 1):
                   print(f"{i}. Q: {conv['query'][:50]}...")
                   print(f"   A: {conv['response'][:50]}...")
               return True
           
           elif command.startswith("/search"):
               query = command[8:].strip()
               results = self.tools.search_by_filter(query)
               print(f"\n🔍 Search Results for '{query}':")
               for i, result in enumerate(results[:3], 1):
                   print(f"{i}. {result['docid']} (Score: {result['@search.score']:.2f})")
                   print(f"   Preview: {result['page'][:100]}...")
               return True
           
           elif command.startswith("/info"):
               docid = command[6:].strip()
               info = self.tools.get_document_summary(docid)
               if info:
                   print(f"\n📄 Document Info:")
                   for key, value in info.items():
                       print(f"   {key}: {value}")
               else:
                   print(f"Document '{docid}' not found.")
               return True
           
           elif command.startswith("/quit"):
               print("👋 Goodbye!")
               return True
           
           return False
       
       def run(self):
           """Run the interactive chat"""
           self.print_welcome()
           
           while True:
               try:
                   user_input = input("\n💬 You: ").strip()
                   
                   if not user_input:
                       continue
                   
                   # Handle commands
                   if user_input.startswith("/"):
                       if self.handle_command(user_input):
                           if user_input.startswith("/quit"):
                               break
                           continue
                   
                   # Process regular query
                   print("🤔 Thinking...")
                   response = self.agent.ask(user_input)
                   
                   print(f"\n🤖 Assistant: {response.answer}")
                   print(f"\n📚 Sources: {len(response.sources)} documents")
                   print(f"⏱️  Response time: {response.response_time:.2f}s")
                   
                   # Save to memory
                   self.memory.add_interaction(
                       user_input,
                       response.answer,
                       response.sources
                   )
                   
               except KeyboardInterrupt:
                   print("\n👋 Goodbye!")
                   break
               except Exception as e:
                   print(f"❌ Error: {e}")
   
   if __name__ == "__main__":
       chat = InteractiveRAGChat()
       chat.run()
   ```

2. **Run the Interactive Chat**:
   ```bash
   python interactive_chat.py
   ```

   **Example Session**:
   ```
   🤖 Interactive RAG Agent Chat
   ==================================================
   Commands:
     /help - Show this help
     /history - Show conversation history
     /search <query> - Search documents
     /info <docid> - Get document info
     /quit - Exit chat
   ==================================================
   
   💬 You: What is GPT-4?
   🤔 Thinking...
   
   🤖 Assistant: GPT-4 is a large-scale, multimodal model developed by OpenAI that can accept both image and text inputs and produce text outputs...
   
   📚 Sources: 3 documents
   ⏱️  Response time: 2.35s
   
   💬 You: /search machine learning
   
   🔍 Search Results for 'machine learning':
   1. GPT-4-Technical-Report_page_3_abc123 (Score: 0.89)
      Preview: Machine learning capabilities of GPT-4 include advanced reasoning...
   2. document-intelligence-4_page_1_def456 (Score: 0.76)
      Preview: Document intelligence uses machine learning to extract...
   
   💬 You: Tell me more about the machine learning capabilities
   🤔 Thinking...
   
   🤖 Assistant: Based on the documents, GPT-4's machine learning capabilities include...
   ```

**Key Features Implemented**:
- Custom RAG agent with conversation memory
- Interactive chat interface with commands
- Document search and filtering tools
- Real-time response generation
- Performance monitoring

---

## Tutorial 4: Evaluation and Performance Optimization

### Learning Objectives
- Implement comprehensive evaluation metrics
- Create custom evaluation datasets
- Optimize RAG performance
- Monitor system metrics

### Step 1: Understanding RAG Evaluation

RAG evaluation focuses on four key metrics:
- **Groundedness**: Are responses supported by the context?
- **Relevance**: Are responses relevant to the query?
- **Completeness**: Do responses fully address the query?
- **Intent Resolution**: Do responses resolve the user's intent?

### Step 2: Create Custom Evaluation Dataset

1. **Generate Evaluation Dataset**:
   ```python
   # create_eval_dataset.py
   import json
   import random
   from typing import List, Dict
   from azure.search.documents import SearchClient
   from azure.identity import DefaultAzureCredential
   
   class EvaluationDatasetGenerator:
       def __init__(self, search_service: str, search_index: str):
           endpoint = f"https://{search_service}.search.windows.net"
           self.search_client = SearchClient(
               endpoint=endpoint,
               index_name=search_index,
               credential=DefaultAzureCredential()
           )
       
       def generate_questions_from_documents(self, num_questions: int = 20) -> List[Dict]:
           """Generate evaluation questions from indexed documents"""
           
           # Get all documents
           all_docs = list(self.search_client.search("*", top=50))
           
           # Question templates
           question_templates = [
               "What is {topic}?",
               "How does {topic} work?",
               "What are the key features of {topic}?",
               "Explain {topic} in detail.",
               "What are the benefits of {topic}?",
               "How can {topic} be used?",
               "What problems does {topic} solve?",
               "Compare {topic} with alternatives."
           ]
           
           evaluation_data = []
           
           for i in range(num_questions):
               # Select random document
               doc = random.choice(all_docs)
               content = doc["page"]
               
               # Extract potential topics (simple approach)
               topics = self._extract_topics(content)
               if not topics:
                   continue
               
               topic = random.choice(topics)
               template = random.choice(question_templates)
               question = template.format(topic=topic)
               
               # Create ground truth from document content
               ground_truth = self._create_ground_truth(content, topic)
               
               evaluation_data.append({
                   "query": question,
                   "context": content,
                   "ground_truth": ground_truth,
                   "source_docid": doc["docid"]
               })
           
           return evaluation_data
       
       def _extract_topics(self, content: str) -> List[str]:
           """Extract potential topics from content"""
           import re
           
           # Simple topic extraction - look for noun phrases
           # In practice, you'd use more sophisticated NLP
           sentences = content.split('. ')
           topics = []
           
           for sentence in sentences[:3]:  # First few sentences
               # Extract capitalized phrases
               phrases = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', sentence)
               topics.extend(phrases)
           
           # Filter and clean topics
           filtered_topics = []
           for topic in topics:
               if 2 <= len(topic.split()) <= 4 and len(topic) > 3:
                   filtered_topics.append(topic)
           
           return list(set(filtered_topics))[:5]  # Return unique topics
       
       def _create_ground_truth(self, content: str, topic: str) -> str:
           """Create ground truth answer from content"""
           sentences = content.split('. ')
           relevant_sentences = []
           
           for sentence in sentences:
               if topic.lower() in sentence.lower():
                   relevant_sentences.append(sentence.strip())
           
           if relevant_sentences:
               return '. '.join(relevant_sentences[:2]) + '.'
           else:
               return content[:200] + '...'
       
       def save_dataset(self, dataset: List[Dict], filename: str):
           """Save dataset to JSONL file"""
           with open(filename, 'w') as f:
               for item in dataset:
                   f.write(json.dumps(item) + '\n')
           print(f"Saved {len(dataset)} evaluation items to {filename}")
   
   # Generate evaluation dataset
   generator = EvaluationDatasetGenerator(
       "search-rag-tutorial",
       "rag-tutorial-index"
   )
   
   dataset = generator.generate_questions_from_documents(25)
   generator.save_dataset(dataset, "custom_evaluation_dataset.jsonl")
   ```

2. **Manual Evaluation Dataset**:
   ```python
   # manual_eval_dataset.py
   import json
   
   # Create high-quality manual evaluation dataset
   manual_evaluation_data = [
       {
           "query": "What is GPT-4 and what makes it different from previous models?",
           "context": "GPT-4 is a large-scale, multimodal model which can accept image and text inputs and produce text outputs. While less capable than humans in many real-world scenarios, GPT-4 exhibits human-level performance on various professional and academic benchmarks, including passing a simulated bar exam with a score around the top 10% of test takers.",
           "ground_truth": "GPT-4 is a large-scale, multimodal model that can process both image and text inputs to generate text outputs. It differs from previous models by achieving human-level performance on professional and academic benchmarks, including scoring in the top 10% on bar exam simulations."
       },
       {
           "query": "How does Azure Document Intelligence process PDF documents?",
           "context": "Azure Document Intelligence uses advanced machine learning models to extract text, key-value pairs, tables, and structures from documents. The service can analyze document layout, recognize printed and handwritten text, and extract semantic meaning from various document types including forms, invoices, and receipts.",
           "ground_truth": "Azure Document Intelligence processes PDF documents by using advanced machine learning models to extract text, key-value pairs, tables, and document structures. It can analyze layout, recognize both printed and handwritten text, and extract semantic meaning from various document types."
       },
       # Add more evaluation items...
   ]
   
   # Save manual dataset
   with open("manual_evaluation_dataset.jsonl", 'w') as f:
       for item in manual_evaluation_data:
           f.write(json.dumps(item) + '\n')
   
   print(f"Created manual evaluation dataset with {len(manual_evaluation_data)} items")
   ```

### Step 3: Implement Custom Evaluators

1. **Create Custom Evaluator Framework**:
   ```python
   # custom_evaluators.py
   import re
   import asyncio
   from typing import Dict, List, Any
   from azure.ai.evaluation import GroundednessEvaluator, RelevanceEvaluator
   from openai import AzureOpenAI
   import numpy as np
   
   class CustomEvaluatorFramework:
       def __init__(self, openai_client: AzureOpenAI, model: str = "gpt-4o"):
           self.openai_client = openai_client
           self.model = model
           
           # Initialize Azure AI evaluators
           self.groundedness_evaluator = GroundednessEvaluator(
               model_config={"model": model, "api_version": "2024-12-01-preview"}
           )
           self.relevance_evaluator = RelevanceEvaluator(
               model_config={"model": model, "api_version": "2024-12-01-preview"}
           )
       
       async def evaluate_citation_accuracy(self, response: str, sources: List[Dict]) -> float:
           """Evaluate if citations in response are accurate"""
           # Extract citation patterns (e.g., "Document 1", "ID: abc123")
           citation_pattern = r'(?:Document\s+\d+|ID:\s*\w+)'
           citations = re.findall(citation_pattern, response, re.IGNORECASE)
           
           if not citations:
               return 0.5  # No citations provided
           
           # Check if cited documents exist in sources
           source_ids = [src.get('docid', '') for src in sources]
           
           accurate_citations = 0
           for citation in citations:
               # Simple matching - in practice, use more sophisticated matching
               for source_id in source_ids:
                   if source_id in citation or any(part in citation for part in source_id.split('_')):
                       accurate_citations += 1
                       break
           
           return accurate_citations / len(citations) if citations else 0.0
       
       async def evaluate_response_completeness(
           self, query: str, response: str, ground_truth: str
       ) -> float:
           """Evaluate how complete the response is compared to ground truth"""
           
           prompt = f"""
           Evaluate how completely the response answers the query compared to the ground truth.
           
           Query: {query}
           
           Ground Truth: {ground_truth}
           
           Response: {response}
           
           Rate the completeness on a scale of 0.0 to 1.0 where:
           - 1.0: Response covers all aspects of the ground truth
           - 0.8: Response covers most aspects with minor omissions
           - 0.6: Response covers key aspects but misses some details
           - 0.4: Response covers some aspects but significant gaps
           - 0.2: Response minimally addresses the query
           - 0.0: Response doesn't address the query
           
           Return only the numeric score.
           """
           
           try:
               result = self.openai_client.chat.completions.create(
                   model=self.model,
                   messages=[{"role": "user", "content": prompt}],
                   temperature=0.0,
                   max_tokens=10
               )
               
               score_text = result.choices[0].message.content.strip()
               return float(re.findall(r'0\.\d+|1\.0|0|1', score_text)[0])
           except:
               return 0.5  # Default score if evaluation fails
       
       async def evaluate_factual_consistency(
           self, response: str, context: str
       ) -> float:
           """Evaluate factual consistency between response and context"""
           
           prompt = f"""
           Check if the response contains any factual inconsistencies with the provided context.
           
           Context: {context}
           
           Response: {response}
           
           Rate factual consistency on a scale of 0.0 to 1.0 where:
           - 1.0: All facts in response are consistent with context
           - 0.8: Minor inconsistencies that don't affect main meaning
           - 0.6: Some inconsistencies but generally aligned
           - 0.4: Significant inconsistencies
           - 0.2: Major factual errors
           - 0.0: Completely inconsistent with context
           
           Return only the numeric score.
           """
           
           try:
               result = self.openai_client.chat.completions.create(
                   model=self.model,
                   messages=[{"role": "user", "content": prompt}],
                   temperature=0.0,
                   max_tokens=10
               )
               
               score_text = result.choices[0].message.content.strip()
               return float(re.findall(r'0\.\d+|1\.0|0|1', score_text)[0])
           except:
               return 0.5
       
       async def comprehensive_evaluation(
           self, query: str, response: str, context: str, 
           ground_truth: str, sources: List[Dict]
       ) -> Dict[str, float]:
           """Run comprehensive evaluation"""
           
           # Run all evaluations concurrently
           tasks = [
               self.groundedness_evaluator(
                   query=query, response=response, context=context
               ),
               self.relevance_evaluator(
                   query=query, response=response, context=context
               ),
               self.evaluate_citation_accuracy(response, sources),
               self.evaluate_response_completeness(query, response, ground_truth),
               self.evaluate_factual_consistency(response, context)
           ]
           
           results = await asyncio.gather(*tasks)
           
           return {
               "groundedness": results[0].score if hasattr(results[0], 'score') else results[0],
               "relevance": results[1].score if hasattr(results[1], 'score') else results[1],
               "citation_accuracy": results[2],
               "completeness": results[3],
               "factual_consistency": results[4],
               "overall_score": np.mean([r.score if hasattr(r, 'score') else r for r in results])
           }
   ```

### Step 4: Run Comprehensive Evaluation

1. **Evaluation Runner**:
   ```python
   # run_comprehensive_evaluation.py
   import json
   import asyncio
   import time
   from typing import List, Dict
   from custom_rag_agent import CustomRAGAgent
   from custom_evaluators import CustomEvaluatorFramework
   from openai import AzureOpenAI
   import os
   from dotenv import load_dotenv
   
   load_dotenv()
   
   class ComprehensiveEvaluationRunner:
       def __init__(self):
           self.agent = CustomRAGAgent(
               search_service=os.getenv("AZURE_SEARCH_SERVICE_NAME"),
               search_index=os.getenv("AZURE_SEARCH_INDEX_NAME"),
               openai_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
               openai_key=os.getenv("AZURE_OPENAI_API_KEY")
           )
           
           self.evaluator = CustomEvaluatorFramework(
               openai_client=self.agent.openai_client
           )
           
           self.results = []
       
       def load_evaluation_dataset(self, filename: str) -> List[Dict]:
           """Load evaluation dataset from JSONL file"""
           dataset = []
           with open(filename, 'r') as f:
               for line in f:
                   dataset.append(json.loads(line.strip()))
           return dataset
       
       async def evaluate_single_query(self, eval_item: Dict) -> Dict:
           """Evaluate a single query"""
           query = eval_item["query"]
           expected_context = eval_item["context"]
           ground_truth = eval_item["ground_truth"]
           
           print(f"Evaluating: {query[:50]}...")
           
           # Get agent response
           start_time = time.time()
           response = self.agent.ask(query)
           response_time = time.time() - start_time
           
           # Run comprehensive evaluation
           eval_scores = await self.evaluator.comprehensive_evaluation(
               query=query,
               response=response.answer,
               context=expected_context,  # Use expected context for evaluation
               ground_truth=ground_truth,
               sources=response.sources
           )
           
           return {
               "query": query,
               "response": response.answer,
               "ground_truth": ground_truth,
               "sources": response.sources,
               "response_time": response_time,
               "evaluation_scores": eval_scores
           }
       
       async def run_evaluation(self, dataset_file: str, output_file: str):
           """Run evaluation on entire dataset"""
           dataset = self.load_evaluation_dataset(dataset_file)
           
           print(f"Running evaluation on {len(dataset)} queries...")
           
           # Run evaluations with limited concurrency
           semaphore = asyncio.Semaphore(3)  # Limit to 3 concurrent evaluations
           
           async def evaluate_with_semaphore(eval_item):
               async with semaphore:
                   return await self.evaluate_single_query(eval_item)
           
           tasks = [evaluate_with_semaphore(item) for item in dataset]
           results = await asyncio.gather(*tasks)
           
           # Calculate aggregate metrics
           aggregate_scores = self.calculate_aggregate_metrics(results)
           
           # Save results
           output_data = {
               "evaluation_summary": aggregate_scores,
               "individual_results": results,
               "dataset_size": len(dataset),
               "timestamp": time.time()
           }
           
           with open(output_file, 'w') as f:
               json.dump(output_data, f, indent=2)
           
           print(f"Evaluation completed. Results saved to {output_file}")
           return aggregate_scores
       
       def calculate_aggregate_metrics(self, results: List[Dict]) -> Dict:
           """Calculate aggregate evaluation metrics"""
           import numpy as np
           
           metrics = {}
           score_types = [
               "groundedness", "relevance", "citation_accuracy", 
               "completeness", "factual_consistency", "overall_score"
           ]
           
           for score_type in score_types:
               scores = [r["evaluation_scores"][score_type] for r in results]
               metrics[score_type] = {
                   "mean": np.mean(scores),
                   "std": np.std(scores),
                   "min": np.min(scores),
                   "max": np.max(scores),
                   "median": np.median(scores)
               }
           
           # Response time metrics
           response_times = [r["response_time"] for r in results]
           metrics["response_time"] = {
               "mean": np.mean(response_times),
               "std": np.std(response_times),
               "min": np.min(response_times),
               "max": np.max(response_times),
               "median": np.median(response_times)
           }
           
           return metrics
       
       def print_evaluation_summary(self, aggregate_scores: Dict):
           """Print evaluation summary"""
           print("\n" + "="*60)
           print("EVALUATION SUMMARY")
           print("="*60)
           
           for metric, stats in aggregate_scores.items():
               if metric == "response_time":
                   print(f"\n{metric.upper()}:")
                   print(f"  Average: {stats['mean']:.3f}s")
                   print(f"  Median:  {stats['median']:.3f}s")
                   print(f"  Range:   {stats['min']:.3f}s - {stats['max']:.3f}s")
               else:
                   print(f"\n{metric.upper()}:")
                   print(f"  Average: {stats['mean']:.3f}")
                   print(f"  Median:  {stats['median']:.3f}")
                   print(f"  Range:   {stats['min']:.3f} - {stats['max']:.3f}")
   
   # Run the evaluation
   async def main():
       runner = ComprehensiveEvaluationRunner()
       
       # Run on custom dataset
       aggregate_scores = await runner.run_evaluation(
           "custom_evaluation_dataset.jsonl",
           "comprehensive_evaluation_results.json"
       )
       
       runner.print_evaluation_summary(aggregate_scores)
   
   if __name__ == "__main__":
       asyncio.run(main())
   ```

**Expected Output**:
```
Running evaluation on 25 queries...
Evaluating: What is GPT-4 and what makes it different...
Evaluating: How does Azure Document Intelligence...
...

============================================================
EVALUATION SUMMARY
============================================================

GROUNDEDNESS:
  Average: 0.847
  Median:  0.850
  Range:   0.600 - 1.000

RELEVANCE:
  Average: 0.823
  Median:  0.820
  Range:   0.650 - 0.950

CITATION_ACCURACY:
  Average: 0.780
  Median:  0.800
  Range:   0.500 - 1.000

COMPLETENESS:
  Average: 0.756
  Median:  0.750
  Range:   0.400 - 0.950

FACTUAL_CONSISTENCY:
  Average: 0.892
  Median:  0.900
  Range:   0.700 - 1.000

OVERALL_SCORE:
  Average: 0.820
  Median:  0.824
  Range:   0.650 - 0.940

RESPONSE_TIME:
  Average: 2.456s
  Median:  2.300s
  Range:   1.200s - 4.100s
```

This comprehensive evaluation framework provides deep insights into RAG performance across multiple dimensions, helping identify areas for improvement and optimization.