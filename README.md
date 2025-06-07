# Azure AI Foundry Workshop

This repository contains code and resources for a comprehensive Azure AI Foundry workshop focusing on building intelligent document processing and search solutions using Azure AI services.

## Workshop Goals

- **Explain** the core mechanics of large-language models (LLMs) and retrieval-augmented generation (RAG) so that participants can articulate when and why to combine the two in enterprise solutions
- **Apply** Azure AI Services — Document Intelligence, AI Search, and Azure OpenAI — to solve a realistic extraction-and-chat scenario during guided labs
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
- AI Agents
- Agent Service
- Data Indexing
- AI Agents Observability
- AI Agents Evaluations
- Playground
- **Demo + Hands-on**: Agent Service (create agent A, B, communicate between them + MCP)

### 4. Azure Responsible AI and Security (30 minutes)
- Content Filters
- Prompt Protection

### 5. Optional: Common Architectural Patterns for AI-Based Applications (45 minutes)
- Agentic Applications Architecture
- Best Practices and Design Patterns

## Project Structure

```
aifoundry-workshop/
├── aisearch/                    # Azure AI Search implementation
│   ├── create_search_index.py   # Creates search index with vector fields
│   ├── document_processor.py    # Processes documents and generates embeddings
│   └── README.md                # Detailed setup instructions
├── docintel/                    # Document Intelligence utilities
│   ├── pdf-2-md.py             # PDF to Markdown conversion
│   └── data/                    # Sample documents
│       ├── *.pdf                # Source PDF files
│       └── *.md                 # Converted Markdown files
├── pyproject.toml               # Python dependencies
└── README.md                    # This file
```

## Key Features

### Azure AI Search Integration
- **Vector Search**: Semantic search using embeddings
- **Hybrid Search**: Combines keyword and vector search
- **Document Indexing**: Automatic processing and indexing of Markdown documents

### Azure AI Foundry Integration
- **Text Embeddings**: Using Azure OpenAI text-embedding-3-small model
- **Document Processing**: Smart page splitting and content extraction
- **Search Capabilities**: Semantic and hybrid search functionality

### Document Intelligence
- **PDF Processing**: Convert PDF documents to Markdown format
- **Content Extraction**: Extract and structure document content
- **Page Segmentation**: Intelligent document page splitting

## Getting Started

### Prerequisites
- Python 3.12+
- Azure subscription with access to:
  - Azure OpenAI Service
  - Azure AI Search
  - Azure Document Intelligence
- UV package manager (recommended) or pip

### Quick Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/vladfeigin/aifoundry-workshop.git
   cd aifoundry-workshop
   ```

2. **Install dependencies**
   ```bash
   # Using UV (recommended)
   uv sync
   
   # Or using pip
   pip install -e .
   ```

3. **Configure environment**
   ```bash
   cp aisearch/.env.template aisearch/.env
   # Edit .env with your Azure credentials
   ```

4. **Run the search index creation**
   ```bash
   cd aisearch
   python create_search_index.py
   ```

5. **Process documents**
   ```bash
   python document_processor.py
   ```

## Environment Configuration

Create a `.env` file in the `aisearch/` directory with the following variables:

```env
# Azure AI Search
AZURE_SEARCH_SERVICE_NAME=your-search-service
AZURE_SEARCH_API_KEY=your-search-api-key
AZURE_SEARCH_INDEX_NAME=your-index-name

# Azure OpenAI
AZURE_OPENAI_ENDPOINT=https://your-openai-service.openai.azure.com/
AZURE_OPENAI_API_KEY=your-openai-api-key
AZURE_OPENAI_API_VERSION=2024-05-01-preview
AZURE_OPENAI_EMBEDDING_MODEL=text-embedding-3-small
```

## Usage Examples

### Creating a Search Index
```python
from aisearch.create_search_index import create_search_index

# Creates an index with vector field support
create_search_index()
```

### Processing Documents
```python
from aisearch.document_processor import DocumentProcessor

processor = DocumentProcessor()
processor.process_documents()
```

### Searching Documents
```python
# Semantic search
results = processor.search_documents("your search query")

# Hybrid search
results = processor.hybrid_search("your search query")
```

## Contributing

This workshop is designed for learning purposes. Feel free to:
- Experiment with different models and configurations
- Add new document processing capabilities
- Enhance search functionality
- Improve error handling and logging

## Resources

- [Azure AI Foundry Documentation](https://docs.microsoft.com/azure/ai-foundry/)
- [Azure AI Search Documentation](https://docs.microsoft.com/azure/search/)
- [Azure OpenAI Service Documentation](https://docs.microsoft.com/azure/cognitive-services/openai/)
- [Azure Document Intelligence Documentation](https://docs.microsoft.com/azure/cognitive-services/document-intelligence/)

## License

This project is for educational purposes as part of the Azure AI Foundry workshop.
