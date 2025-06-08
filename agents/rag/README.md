# RAG (Retrieval Augmented Generation) Agent

This directory contains a production-ready RAG agent implementation that demonstrates the complete RAG pipeline using Azure AI services.

## Overview

The RAG agent implements the following workflow:

1. **Retrieval**: Searches Azure AI Search index for relevant documents using hybrid search (keyword + semantic)
2. **Augmentation**: Prepares context from retrieved documents with proper formatting
3. **Generation**: Uses Azure OpenAI GPT models to generate responses based on the retrieved context

## Features

### 🔍 **Advanced Search Capabilities**

- **Hybrid Search**: Combines keyword and semantic vector search for optimal results
- **Configurable Retrieval**: Adjustable top-k documents, context length, and search parameters
- **Smart Context Management**: Automatically handles context window limits

### 🛡️ **Azure Best Practices**

- **Managed Identity Support**: Prefers managed identity over API keys for production security
- **Error Handling**: Comprehensive retry logic with exponential backoff
- **Performance Monitoring**: Tracks retrieval and generation times
- **Logging**: Detailed logging for debugging and observability

### 🚀 **Production Ready**

- **Type Safety**: Full type annotations with dataclasses for structured responses
- **Configuration**: Environment-based configuration with sensible defaults
- **Authentication**: Supports both API key (development) and managed identity (production)
- **Observability**: Built-in performance metrics and timing

## Architecture

### System Design

```
User Query
    ↓
[Embedding Generation] ──→ Azure OpenAI (text-embedding-3-small)
    ↓
[Hybrid Search] ──────────→ Azure AI Search (keyword + vector)
    ↓
[Context Preparation] ─────→ Smart truncation and formatting
    ↓
[Response Generation] ─────→ Azure OpenAI (GPT-4.1/GPT-4o)
    ↓
Structured Response
```

### Prompt Architecture

- **System Message**: Contains detailed instructions for how to use context, cite sources, and handle limitations
- **User Message**: Contains only the retrieved document context and user question
- **Clean Separation**: No redundant instructions between system and user prompts

## Quick Start

### 1. Environment Setup

Create a `.env` file in the project root or set environment variables:

```bash
# Required
AZURE_SEARCH_SERVICE_NAME=your-search-service
AZURE_SEARCH_INDEX_NAME=ai-foundry-workshop-index-v1
AZURE_OPENAI_ENDPOINT=https://your-openai.openai.azure.com/

# Optional (for development - use managed identity in production)
AZURE_SEARCH_API_KEY=your-search-api-key
AZURE_OPENAI_API_KEY=your-openai-api-key

# Model configuration (optional)
AZURE_OPENAI_CHAT_MODEL=gpt-4.1
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
from rag_agent import RAGAgent

# Initialize the agent
agent = RAGAgent(
    search_service_name="your-search-service",
    search_index_name="ai-foundry-workshop-index-v1", 
    azure_openai_endpoint="https://your-openai.openai.azure.com/",
    chat_model="gpt-4",
    top_k_documents=3
)

# Ask a question
response = agent.ask("What are the key capabilities of Azure AI services?")

# Access the results
print(f"Answer: {response.answer}")
print(f"Sources: {len(response.sources)} documents")
print(f"Total time: {response.total_time:.3f}s")

# Access individual sources
for i, source in enumerate(response.sources, 1):
    print(f"Source {i}: {source['docid']} (score: {source['score']:.3f})")
```

## Configuration Options

### RAGAgent Parameters

| Parameter                    | Type | Default                  | Description                         |
| ---------------------------- | ---- | ------------------------ | ----------------------------------- |
| `search_service_name`      | str  | Required                 | Azure AI Search service name        |
| `search_index_name`        | str  | Required                 | Name of the search index            |
| `azure_openai_endpoint`    | str  | Required                 | Azure OpenAI endpoint URL           |
| `azure_openai_api_version` | str  | "2024-12-01-preview"     | API version for Azure OpenAI        |
| `chat_model`               | str  | "gpt-4"                  | Chat completion model name          |
| `embedding_model`          | str  | "text-embedding-3-small" | Embedding model for semantic search |
| `max_context_length`       | int  | 20000                    | Maximum context length for the LLM  |
| `top_k_documents`          | int  | 3                        | Number of documents to retrieve     |

### Environment Variables

| Variable                      | Required | Description                  |
| ----------------------------- | -------- | ---------------------------- |
| `AZURE_SEARCH_SERVICE_NAME` | ✅       | Azure AI Search service name |
| `AZURE_SEARCH_INDEX_NAME`   | ✅       | Search index name            |
| `AZURE_OPENAI_ENDPOINT`     | ✅       | Azure OpenAI endpoint        |
| `AZURE_SEARCH_API_KEY`      | ⚠️     | Search API key (dev only)    |
| `AZURE_OPENAI_API_KEY`      | ⚠️     | OpenAI API key (dev only)    |
| `AZURE_OPENAI_CHAT_MODEL`   | ❌       | Override default chat model  |

## Response Format

The RAG agent returns a structured `RAGResponse` object:

```python
@dataclass
class RAGResponse:
    answer: str                    # Generated response
    sources: List[Dict[str, Any]]  # Retrieved documents with metadata
    query: str                     # Original user query
    retrieval_time: float          # Time spent retrieving documents
    generation_time: float         # Time spent generating response
    total_time: float             # Total pipeline time
```

### Source Information

Each source in the `sources` list contains:

- `docid`: Document identifier from the search index
- `content_preview`: First 200 characters of the document content
- `score`: Search relevance score
- `metadata`: Additional metadata about the source

## Search Modes

### Hybrid Search (Default)

Combines keyword and semantic search for best results:

```python
response = agent.ask("your question", use_hybrid_search=True)
```

### Keyword-Only Search

Uses only traditional keyword search:

```python
response = agent.ask("your question", use_hybrid_search=False)
```

## Performance Considerations

### Context Management

- The agent automatically truncates context to fit within the model's context window
- Documents are prioritized by search score
- Context length is configurable via `max_context_length`

### Retry Logic

- Embedding generation includes exponential backoff retry logic
- Search operations are retried on transient failures
- Response generation handles rate limits gracefully

### Performance Monitoring

All operations are timed and logged:

```
RAG pipeline completed in 2.847s (retrieval: 1.234s, generation: 1.613s)
```

## Security Best Practices

### Authentication

1. **Production**: Use Managed Identity (no API keys required)
2. **Development**: Use API keys stored in environment variables
3. **Never**: Hardcode credentials in source code

### Network Security

- All connections use HTTPS
- Support for Azure Private Endpoints
- VNet integration compatible

### Access Control

- Follows principle of least privilege
- Supports Azure RBAC for fine-grained access control
- Search service can be configured with IP restrictions

## Error Handling

The RAG agent includes comprehensive error handling:

- **Connection Errors**: Automatic retry with exponential backoff
- **Authentication Errors**: Clear error messages with guidance
- **Rate Limiting**: Automatic retry with appropriate delays
- **Context Overflow**: Graceful truncation with logging
- **Model Errors**: Fallback responses with error details

## Troubleshooting

### Common Issues

1. **"Failed to initialize search client"**

   - Check `AZURE_SEARCH_SERVICE_NAME` environment variable
   - Verify API key or managed identity permissions
2. **"Failed to initialize OpenAI client"**

   - Check `AZURE_OPENAI_ENDPOINT` environment variable
   - Verify API key or managed identity permissions
3. **"No relevant documents found"**

   - Check if the search index contains documents
   - Verify index name matches `AZURE_SEARCH_INDEX_NAME`
4. **"Context limit reached"**

   - Increase `max_context_length` parameter
   - Reduce `top_k_documents` to retrieve fewer documents

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
- **Document Processing**: See `../../aisearch/document_processor.py`
- **Document Intelligence**: See `../../docintel/pdf-2-md.py`

## Contributing

When modifying the RAG agent, please:

1. Follow Azure development best practices
2. Maintain type annotations
3. Add appropriate error handling
4. Update tests and documentation
5. Test with both API key and managed identity authentication

## License

This project is part of the Azure AI Foundry Workshop and follows the same license terms.
