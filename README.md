# Azure AI Foundry Workshop

This repository contains code and resources for a comprehensive Azure AI Foundry workshop focusing on building intelligent document processing, search solutions and AI Agents using Azure AI services and Azure AI Foundry.

## Workshop Goals

- **Explain** the core mechanics of large-language models (LLMs) and retrieval-augmented generation (RAG) so that participants can articulate when and why to combine the two in enterprise solutions
- **Apply** Azure AI Services — Document Intelligence, AI Search, and Azure AI Foundry — to solve a realistic extraction-and-chat scenario during guided labs
- **Build** and **evaluate** multi-agent solutions in **Azure AI Foundry**, including model catalog selection, data indexing, agent orchestration via MCP, and built-in observability dashboards
- **Implement** responsible-AI controls (content filters and prompt shields) that satisfy Microsoft's safety baseline for generative AI workloads
- **Design** a reference-grade architecture for agentic applications, balancing accuracy, latency, cost, and governance

## Workshop Agenda

### 1. Introduction to LLMs (45 minutes)

- LLMs Introduction
- Prompt Engineering techniques
- Retrieval-Augmented Generation (RAG)

### 2. Azure AI Services Overview (2.5 hours)

- **Document Intelligence Service**
- **AI Search Service**
- **Azure OpenAI Models overview**
- **Demo**: OpenAI Service + hands-on (in portal and studio)

### 3. Azure AI Foundry (3 hours)

- Model Catalog
- AI Agents Intro
- Agent Service
- AI Agents Observability
- AI Agents Evaluations
- Playground
- **Demo + Labs**

### 4. Azure Responsible AI and Security (30 minutes)

- Content Filters
- Prompt Protection

### 5. Optional: Common Architectural Patterns for AI-Based Applications (45 minutes)

- Agentic Applications Architecture
- Best Practices and Design Patterns

## Project Structure

```
aifoundry-workshop/
├── agents/                      # AI Agents implementation
│   ├── rag/                     # RAG (Retrieval-Augmented Generation) Agent
│   │   ├── rag_agent.py         # Main RAG agent with Azure AI Foundry tracing
│   │   └── README.md            # RAG agent documentation and setup
│   └── evaluations/             # Agent evaluation framework
│       ├── rag/                 # RAG agent evaluation module
│       │   ├── rag_agent_eval.py           # Comprehensive evaluation module
│       │   ├── rag_agent_eval_in_foundry.py # Cloud-based evaluation
│       │   └── README.md        # Evaluation documentation
│       └── data/                # Evaluation datasets
│           ├── single-turn-eval-ds.jsonl   # Evaluation dataset
│           └── output/          # Evaluation results
│               ├── single-turn-eval-ds-agent-output.jsonl
│               └── evaluation_results.json
├── aisearch/                    # Azure AI Search implementation
│   ├── create_search_index.py   # Creates search index with vector fields
│   ├── ingest_documents.py       # Processes documents and generates embeddings
│   └── README.md                # AI Search setup instructions
├── docintel/                    # Document Intelligence utilities
│   ├── pdf-2-md.py             # PDF to Markdown conversion with Azure Document Intelligence
│   └── data/                    # Sample documents and processed outputs
│       ├── *.pdf                # Source PDF files
│       └── *.md                 # Converted Markdown files
├── .vscode/                     # VS Code configuration
│   └── launch.json              # Debug configurations for all modules
├── pyproject.toml               # Python dependencies and project configuration
├── uv.lock                      # UV package manager lock file
├── install_deps.sh              # Dependency installation script
└── README.md                    # This file
```

## Key Features

### RAG Agent in Agent Service

- **Intelligent Document Retrieval**: Context-aware document search and retrieval
- **Azure AI Foundry Tracing**: Comprehensive observability and monitoring
- **Auto-instrumentation**: OpenAI and HTTP requests automatically traced
- **Multi-metric Evaluation**: Groundedness, Relevance, Completeness, and Intent Resolution

### Foundry Agent Evaluation Framework

- **Comprehensive Metrics**: Built using Azure AI Evaluation SDK
- **Automated Evaluation**: Batch evaluation with performance monitoring
- **Multiple Evaluators**: Groundedness, Relevance, Response Completeness, Intent Resolution
- **Detailed Reporting**: JSON outputs with statistical analysis

### Azure AI Search Integration

- **Vector Search**: Semantic search using embeddings
- **Hybrid Search**: Combines keyword and vector search
- **Document Indexing**: Automatic processing and indexing of Markdown documents
- **Advanced Retrieval**: Context-aware document chunking and retrieval

### Azure Document Intelligence Service

- **PDF Processing**: Convert PDF documents to Markdown format using Azure Document Intelligence
- **Content Extraction**: Extract and structure document content intelligently
- **Page Segmentation**: Smart document page splitting and organization

## Getting Started

### Prerequisites

- Azure subscription with access to:
  - Azure AI Foundry
  - Azure AI Search
  - Azure Document Intelligence
- Python 3.12+
- UV package manager (recommended) or pip
- Participants should have "Azure AI User" role
- Azure CLI

### Quick Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/vladfeigin/aifoundry-workshop.git
   cd aifoundry-workshop
   ```
2. **Prepare environment**

   - [Development Containers](https://containers.dev/) Option (recommended):

      Install [Docker](https://www.docker.com/products/docker-desktop/)
      
      Install [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) for VS Code and allow to run the project in a container.

   - Local environment option: 
   
      Installing dependencies for Python:

      ```bash
      # Using UV (recommended)
      uv sync

      # Or using pip
      pip install -e .
      ```

      Install [Azure CLI](https://learn.microsoft.com/en-us/cli/azure/install-azure-cli) as well.

3. **Configure environment**

   ```bash
   # Copy and edit the environment template
   cp .env.template .env
   # Edit .env with your Azure specifc values
   ```
4. **Convert PDFs to Markdown (Document Intelligence)**

   ```bash  
   # Convert PDF documents to Markdown format using Azure Document Intelligence
   # This step prepares your PDF documents for indexing in Azure AI Search

   # Example: Convert a sample PDF to Markdown
   python -m docintel.pdf-2-md ./docintel/data/GPT-4-Technical-Report.pdf ./docintel/data/GPT-4-Technical-Report.md

   # Or convert your own PDF file
   python -m docintel.pdf-2-md path/to/your/document.pdf path/to/output/document.md

   # You can also convert from a URL
   python -m docintel.pdf-2-md https://example.com/document.pdf ./output/document.md
   ```

   **Note**: Ensure your `.env` file contains the Document Intelligence credentials:

   ```env
   AZURE_DOCINTEL_ENDPOINT=https://your-doc-intel-service.cognitiveservices.azure.com/
   AZURE_DOCINTEL_KEY=your-doc-intel-api-key
   ```
5. **Set up Azure AI Search**

   ```bash
   # 1. Login to Azure Portal
   az login

   # 2. Create search index
   # Take the AI Search service name from Azure AI Search portal 
   # Run from the project root folder:

   python ./aisearch/create_search_index.py --search-service <service_name> --index-name <index_name>  --use-api-key
   or
   python -m aisearch.create_search_index --search-service <service_name> --index-name <index_name>  --use-api-key

   Check in Azure AI Search portal, an index has been created.

   # 3. Ingest documents documents to the index
   python -m aisearch.ingest_documents --search-service <service_name> --index-name <index_name>
   ```
6. **Run the RAG Agent**

   ```bash
   # Test the RAG agent
   python -m agents.rag.rag_agent
   ```
7. **Evaluate the RAG Agent**

   ```bash
   # Run local evaluation
   python -m agents.evaluations.rag.rag_agent_eval

   # Run evaluation in Azure
   python -m agents.evaluations.rag.rag_agent_eval_azure

   ```

## Environment Configuration

Create a `.env` file in the project root with the following variables:

```env
# Azure AI Search
AZURE_SEARCH_SERVICE_NAME=your-search-service
AZURE_SEARCH_API_KEY=your-search-api-key
AZURE_SEARCH_INDEX_NAME=your-index-name

# Azure OpenAI
AZURE_OPENAI_ENDPOINT=https://your-openai-service.openai.azure.com/
AZURE_OPENAI_API_KEY=your-openai-api-key
AZURE_OPENAI_API_VERSION=2024-12-01-preview
AZURE_OPENAI_EMBEDDING_MODEL=text-embedding-3-small
AZURE_OPENAI_CHAT_MODEL=gpt-4o

# Azure Document Intelligence
AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT=https://your-doc-intel-service.cognitiveservices.azure.com/
AZURE_DOCUMENT_INTELLIGENCE_API_KEY=your-doc-intel-api-key

# Azure Monitor (for tracing)
APPLICATIONINSIGHTS_CONNECTION_STRING=your-app-insights-connection-string
```

## Usage Examples

### RAG Agent Usage

```python
from agents.rag.rag_agent import RAGAgent

# Initialize the RAG agent
agent = RAGAgent(
    search_service_name="your-search-service",
    search_index_name="your-index-name",
    azure_openai_endpoint="https://your-openai-service.openai.azure.com/",
    chat_model="gpt-4o"
)

# Ask a question
response = agent.ask("What is GPT-4?")
print(f"Answer: {response.answer}")
print(f"Sources: {response.sources}")
```

### Agent Evaluation

```python
from agents.evaluations.rag.rag_agent_eval import RAGAgentEvaluator

# Run comprehensive evaluation
evaluator = RAGAgentEvaluator(rag_agent, project_client)
dataset = evaluator.load_evaluation_dataset("path/to/dataset.jsonl")
results = evaluator.evaluate_dataset(dataset)
evaluator.print_evaluation_summary(results)
```

### Creating a Search Index

```python
from aisearch.create_search_index import create_search_index

# Creates an index with vector field support
create_search_index()
```

### Processing Documents

```python
from aisearch.ingest_documents import DocumentProcessor

processor = DocumentProcessor()
processor.process_documents()
```

### Document Intelligence (PDF to Markdown)

```bash
# Convert PDF documents to Markdown using Azure Document Intelligence
# This is typically the first step in the document processing pipeline

# Convert a local PDF file
python -m docintel.pdf-2-md ./docintel/data/GPT-4-Technical-Report.pdf ./docintel/data/GPT-4-Technical-Report.md

# Convert from a URL
python -m docintel.pdf-2-md https://example.com/document.pdf ./output/document.md

# Convert with custom output location
python -m docintel.pdf-2-md /path/to/your/document.pdf /path/to/output/document.md
```

**Features:**

- Smart content extraction from PDF documents
- Table recognition and Markdown table formatting
- Page-by-page processing with clear delineation
- Support for both local files and HTTP URLs
- Rich progress indicators during processing

### Searching Documents

```python
# Semantic search
results = processor.search_documents("your search query")

# Hybrid search
results = processor.hybrid_search("your search query")
```

## Development and Debugging

### VS Code Integration

The project includes comprehensive VS Code debugging configurations:

- **Debug RAG Agent Evaluation Module**: For debugging the evaluation framework
- **Debug RAG Agent**: For debugging the main RAG agent
- **Debug Document Ingestion (aisearch)**: For debugging AI Search integration
- **Debug Create Search Index (aisearch)**: For debugging index creation
- **Debug PDF to Markdown (docintel)**: For debugging document intelligence

### Running Modules

```bash
# Run individual modules
python -m agents.rag.rag_agent
python -m agents.evaluations.rag.rag_agent_eval
python -m aisearch.create_search_index
python -m aisearch.ingest_documents
python -m docintel.pdf-2-md
```

### Evaluation Metrics

The evaluation framework provides comprehensive metrics:

- **Groundedness**: Measures if responses are supported by retrieved context
- **Relevance**: Evaluates response relevance to user queries
- **Response Completeness**: Assesses if responses fully address queries
- **Intent Resolution**: Determines if responses resolve user intent

## Contributing

This workshop is designed for learning purposes. Feel free to:

- Experiment with different models and configurations
- Add new document processing capabilities
- Enhance search functionality
- Improve evaluation metrics and datasets
- Add new agent capabilities
- Enhance observability and tracing
- Improve error handling and logging

## Resources

- [Azure AI Foundry Documentation](https://docs.microsoft.com/azure/ai-foundry/)
- [Azure AI Search Documentation](https://docs.microsoft.com/azure/search/)
- [Azure OpenAI Service Documentation](https://docs.microsoft.com/azure/cognitive-services/openai/)
- [Azure Document Intelligence Documentation](https://docs.microsoft.com/azure/cognitive-services/document-intelligence/)
- [Azure AI Evaluation SDK](https://docs.microsoft.com/azure/ai-foundry/how-to/evaluate-generative-ai-app)
- [Azure Monitor OpenTelemetry](https://docs.microsoft.com/azure/azure-monitor/app/opentelemetry-enable)

## Architecture Overview

The workshop demonstrates a production-ready RAG architecture:

1. **Document Processing**: Azure Document Intelligence converts PDFs to structured Markdown
2. **Indexing**: Azure AI Search creates vector embeddings and indexes documents
3. **Retrieval**: Hybrid search combines semantic and keyword matching
4. **Generation**: Azure OpenAI generates contextual responses
5. **Evaluation**: Multi-metric assessment ensures quality and performance
6. **Observability**: Azure AI Foundry provides comprehensive tracing and monitoring

## License

This project is for educational purposes as part of the Azure AI Foundry workshop.
