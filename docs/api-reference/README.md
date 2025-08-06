# API Reference

Quick reference for the main classes and functions in the Azure AI Foundry Workshop.

## 📦 Modules Overview

### `agents.rag.rag_agent`

#### `RAGAgentService`
Main RAG agent implementation using Azure AI Agent Service.

```python
class RAGAgentService:
    def __init__(
        self, 
        project_endpoint: str,
        search_index_name: str = "ai-foundry-workshop-index-v1",
        chat_model: str = "gpt-4o"
    )
    
    def ask(self, query: str) -> RAGResponse:
        """Process a query and return RAG response"""
```

**Parameters:**
- `project_endpoint`: Azure AI Foundry project endpoint
- `search_index_name`: Name of the Azure AI Search index
- `chat_model`: Azure OpenAI chat model deployment name

**Returns:** `RAGResponse` object with answer, sources, timing, and metadata

### `aisearch.create_search_index`

#### `create_search_index()`
Creates Azure AI Search index with vector search capabilities.

```python
def create_search_index(
    search_service_name: str,
    index_name: str,
    use_api_key: bool = False
) -> None
```

### `aisearch.ingest_documents`

#### `DocumentProcessor`
Processes and ingests documents into Azure AI Search.

```python
class DocumentProcessor:
    def process_documents(self, docs_path: str = "./docintel/data") -> None
        """Process markdown documents and ingest into search index"""
```

### `docintel.pdf-2-md`

#### `convert_pdf_to_markdown()`
Converts PDF documents to Markdown using Azure Document Intelligence.

```python
def convert_pdf_to_markdown(
    pdf_path: str,
    output_path: str,
    endpoint: str,
    key: str
) -> str
```

### `agents.evaluations.rag.rag_agent_eval`

#### `RAGAgentEvaluator`
Local evaluation framework for RAG agents.

```python
class RAGAgentEvaluator:
    def __init__(
        self, 
        rag_agent: RAGAgentService, 
        project_client: AIProjectClient
    )
    
    def evaluate_dataset(self, dataset: List[Dict]) -> Dict
        """Evaluate RAG agent on dataset"""
```

## 🔧 Configuration Classes

### `RAGResponse`
Response object returned by RAG agents.

```python
@dataclass
class RAGResponse:
    answer: str
    sources: List[Dict[str, Any]]
    total_time: float
    thread_id: str
    run_id: str
```

### Environment Variables

```python
# Required environment variables
AZURE_SEARCH_SERVICE_NAME: str
AZURE_SEARCH_INDEX_NAME: str
PROJECT_ENDPOINT: str
AZURE_OPENAI_ENDPOINT: str
AZURE_OPENAI_API_KEY: str
AZURE_DOCINTEL_ENDPOINT: str
AZURE_DOCINTEL_KEY: str
```

## 📊 Usage Examples

### Basic RAG Query
```python
from agents.rag.rag_agent import RAGAgentService

agent = RAGAgentService(
    project_endpoint="https://your-project.services.ai.azure.com/api/projects/your-project",
    search_index_name="your-index"
)

response = agent.ask("What is GPT-4?")
print(response.answer)
```

### Document Processing
```python
from aisearch.ingest_documents import DocumentProcessor

processor = DocumentProcessor()
processor.process_documents("./my-documents")
```

### Evaluation
```python
from agents.evaluations.rag.rag_agent_eval import RAGAgentEvaluator

evaluator = RAGAgentEvaluator(agent, project_client)
dataset = evaluator.load_evaluation_dataset("eval.jsonl")
results = evaluator.evaluate_dataset(dataset)
```

For detailed usage examples, see [TUTORIALS.md](../TUTORIALS.md).