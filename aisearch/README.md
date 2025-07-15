# Azure AI Search Integration

This directory contains scripts for creating and managing Azure AI Search indexes with vector search capabilities.

## Files

- `create_search_index.py` - Creates search indexes with vector fields for document storage
- `ingest_documents.py` - Processes Markdown documents, generates embeddings, and populates search indexes

## Index Schema

The search index contains three fields:

- `docid` (String, key) - Unique document identifier with page numbers and content hash
- `page` (String, searchable) - Document page content with standard Lucene analyzer
- `page_vector` (Collection[Single]) - 1536-dimension embedding vectors for semantic search

## Vector Search Configuration

Uses HNSW (Hierarchical Navigable Small World) algorithm with:
- `m: 4` - Bi-directional links for new elements
- `efConstruction: 400` - Dynamic candidate list size during construction  
- `efSearch: 500` - Dynamic candidate list size for search
- `metric: cosine` - Distance metric for vector similarity

## Usage

### Create Search Index

```bash
python -m aisearch.create_search_index --search-service <service_name> --index-name <index_name>
```

### Process and Ingest Documents

```bash
python -m aisearch.ingest_documents --search-service <service_name> --index-name <index_name>
```

## Document Processing

The ingest script:

1. Reads Markdown files from `./docintel/data` directory
2. Splits documents using page delimiter pattern: `<!---- Page X ---------------------------------------------------------------------------------------------------------------------------------->` 
3. Falls back to header-based or paragraph-based splitting for files without delimiters
4. Generates embeddings using Azure OpenAI `text-embedding-3-small` model
5. Creates document IDs with format: `{filename}_page_{number}_{hash}`
6. Uploads documents with embeddings to search index

## Search Methods

The ingest script includes test search functionality:

- **Semantic Search** - Uses vector embeddings for similarity matching
- **Hybrid Search** - Combines keyword and vector search with expanded candidate pool
- **Keyword Search** - Traditional full-text search using Lucene analyzer

## Authentication

- **Development**: Uses API keys from environment variables
- **Azure-hosted**: Supports managed identity authentication
- **Required Environment Variables**:
  - `AZURE_SEARCH_API_KEY` (for API key auth)
  - `AZURE_OPENAI_ENDPOINT` 
  - `AZURE_OPENAI_API_KEY`
  - `AZURE_OPENAI_EMBEDDING_MODEL` (optional, defaults to text-embedding-3-small)

## Error Handling

- Exponential backoff retry logic for embedding generation
- Graceful handling of missing files or empty content
- Text truncation for content exceeding model limits (8000 characters)
- Automatic index creation if it doesn't exist

## Dependencies

- `azure-search-documents` - Azure AI Search client
- `azure-identity` - Azure authentication
- `openai` - Azure OpenAI integration
- `python-dotenv` - Environment variable management
