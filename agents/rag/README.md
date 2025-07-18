# RAG (Retrieval Augmented Generation) Agent

This directory contains a RAG agent implementation using the **Azure AI Foundry SDK** and **Azure AI Agent Service**.

**`rag_agent.py`** - RAG implementation using **Azure AI Agent Service SDK** ⭐

## Overview

The RAG agent (`rag_agent.py`) implements a comprehensive RAG workflow using the latest Azure AI services:

### Azure AI Agent Service RAG Agent (`rag_agent.py`) ⭐

- Uses **Azure AI Agent Service SDK** from Azure AI Foundry for managed agent orchestration
- Built-in **Azure AI Search tool integration** with connection-based configuration
- Built-in session **management**
- Enhanced **tool call observability** with OpenTelemetry
- **Connection-based configuration** using Azure AI Foundry project connections

## Features

### 🔍 **Advanced Search Capabilities**

- **Azure AI Search Tool Integration**: Native tool integration for search operations
- **Hybrid Search**: Combines keyword and semantic vector search for optimal results
- **Connection-based Configuration**: Uses Azure AI Foundry project connections

### 🤖 **Azure AI Agent Service Features**

- **Managed Agent Orchestration**: Built-in conversation thread management
- **Azure AI Search Tool Integration**: Native search tool with connection-based configuration
- **Enhanced Observability**: Tool call tracking via OpenTelemetry attributes (not verbose logging)
- **Citation Handling**: Automatic source citation and URL annotation processing
- **Connection-based Setup**: Uses Azure AI Foundry project connections for seamless integration

### 📊 **Azure AI Foundry Tracing & Observability**

- **Auto-Instrumentation**: Automatic tracing of OpenAI API calls and HTTP requests
- **Custom Spans**: Business logic tracing for RAG operations
- **Performance Metrics**: Detailed timing and tool call tracking
- **Error Tracking**: Exception capture and error analysis
- **Azure Monitor Integration**: Seamless integration with Application Insights

## Tracing Implementation

### Overview

The RAG agent implements comprehensive tracing using **Azure Monitor OpenTelemetry** and **auto-instrumentation** to provide deep observability into the RAG pipeline. This enables monitoring, debugging, and optimization of the system in production.

### Tracing Architecture

1. **Auto-Instrumentation**: Automatically captures detailed traces for:

   - All OpenAI API calls (embeddings, chat completions)
   - HTTP requests to Azure services
   - Request/response details, token usage, and timing
2. **Custom Business Logic Spans**: Manual spans for RAG-specific operations:

   - Document retrieval workflow
   - Context preparation and formatting
   - Response generation pipeline
   - Error handling and retry logic

### What Gets Traced

#### 🤖 **OpenAI Operations (Auto-Instrumented)**

- **Embedding Generation**: Model name, input length, output dimensions, token usage
- **Chat Completions**: Model name, prompt length, response length, temperature, max tokens
- **API Performance**: Request/response times, rate limiting, errors

#### 🔍 **Search Operations (Custom Spans)**

- **Document Retrieval**: Query text, search type (hybrid/keyword), number of results
- **Vector Search**: Embedding dimensions, similarity scores, retrieval time
- **Context Processing**: Document count, content length, truncation details

#### 📊 **Business Metrics (Custom Spans)**

- **End-to-End Pipeline**: Total RAG execution time
- **Component Performance**: Retrieval vs generation timing breakdown
- **Quality Metrics**: Search scores, document relevance, response confidence

### Tracing Configuration

#### Environment Setup

The tracing is automatically configured when you set up your Azure AI Foundry project:

```bash
# Required for Azure AI Foundry integration
PROJECT_ENDPOINT=https://your-project.region.api.azureml.ms
```

#### Code Integration

The RAG agent automatically sets up tracing in the constructor:

```python
# Auto-instrumentation setup (in RAGAgent.__init__)
configure_azure_monitor(connection_string=connection_string)

# Enable auto-instrumentation for OpenAI and HTTP requests
OpenAIInstrumentor().instrument()
RequestsInstrumentor().instrument()
```

### Viewing Traces in Azure AI Foundry

#### **Access the Tracing Portal**

- Navigate to your Azure AI Foundry project
- Go to **"Tracing"** in the left navigation
- View real-time traces and historical data

### Custom Span Examples

#### Document Retrieval Span

```python
with tracer.start_as_current_span("rag_agent.retrieve_documents") as span:
    span.set_attribute("search.query", query)
    span.set_attribute("search.use_hybrid", use_hybrid)
    span.set_attribute("search.top_k", self.top_k_documents)
    span.set_attribute("search.documents_retrieved", len(retrieved_docs))
    span.set_attribute("search.retrieval_time", retrieval_time)
```

#### Response Generation Span

```python
with tracer.start_as_current_span("rag_agent.generate_response") as span:
    span.set_attribute("generation.model", self.chat_model)
    span.set_attribute("generation.max_tokens", max_tokens)
    span.set_attribute("generation.temperature", temperature)
    span.set_attribute("generation.query_length", len(query))
    span.set_attribute("generation.context_length", len(context))
```

## Quick Start

### 1. Environment Setup

Create a `.env` file in the project root or set environment variables:

```bash
# Required
PROJECT_ENDPOINT=https://your-project.region.api.azureml.ms
AZURE_SEARCH_INDEX_NAME=ai-foundry-workshop-index-v1
AZURE_TRACING_GEN_AI_CONTENT_RECORDING_ENABLED="true"
AZURE_DOCINTEL_ENDPOINT=https://<docintel service name>.cognitiveservices.azure.com/
AZURE_DOCINTEL_KEY

```

### 2. Run the Demo

```bash
cd agents/rag
python rag_agent.py
```

This will run a demonstration with several test queries and show:

- Retrieved documents with scores
- Generated responses
- Performance metrics (retrieval and generation times)

### 3. Use Programmatically

```python
from rag_agent import RAGAgentService

# Initialize the agent
agent = RAGAgentService(
    project_endpoint=<AI Foundry Project endpoint>,
    search_index_name="ai-foundry-workshop-index-v1",
    chat_model="gpt-4.1"
)

# Ask a question
response = agent.ask("What are the key capabilities of Azure AI services?")

# Access the results
print(f"Answer: {response.answer}")
print(f"Sources: {len(response.sources)} documents")
print(f"Total time: {response.total_time:.3f}s")
print(f"Thread ID: {response.thread_id}")
print(f"Run ID: {response.run_id}")

# Access individual sources
for i, source in enumerate(response.sources, 1):
    print(f"Source {i}: {source}")
```

### Debug Mode

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.INFO)
```

## Dependencies

The RAG agent requires the following Azure SDK packages:

- `azure-search-documents`: For Azure AI Search integration
- `azure-identity`: For managed identity authentication
- `openai`: For Azure OpenAI integration
- `python-dotenv`: For environment variable management

## Related Components

- **Search Index Creation**: See `../../aisearch/create_search_index.py`
- **Document Processing**: See `../../aisearch/ingest_documents.py`
- **Document Intelligence**: See `../../docintel/pdf-2-md.py`
