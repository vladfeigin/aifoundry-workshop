# Contributing to Azure AI Foundry Workshop

Thank you for your interest in contributing to the Azure AI Foundry Workshop! This guide will help you get started with development and contribution workflows.

## Table of Contents

- [Development Environment Setup](#development-environment-setup)
- [Project Structure](#project-structure)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Pull Request Process](#pull-request-process)
- [Documentation Guidelines](#documentation-guidelines)
- [Debugging and Development Tools](#debugging-and-development-tools)

## Development Environment Setup

### Prerequisites

- Python 3.12 or higher
- Azure subscription with access to:
  - Azure AI Foundry
  - Azure AI Search
  - Azure Document Intelligence
  - Azure OpenAI Service
- Azure CLI installed and configured
- Docker (for Development Containers)

### Setup Options

#### Option 1: Development Containers (Recommended)

1. **Install Prerequisites**:
   - [Docker Desktop](https://www.docker.com/products/docker-desktop/)
   - [VS Code](https://code.visualstudio.com/)
   - [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)

2. **Open in Container**:
   ```bash
   git clone https://github.com/vladfeigin/aifoundry-workshop.git
   cd aifoundry-workshop
   code .
   # VS Code will prompt to "Reopen in Container"
   ```

#### Option 2: Local Development

1. **Clone Repository**:
   ```bash
   git clone https://github.com/vladfeigin/aifoundry-workshop.git
   cd aifoundry-workshop
   ```

2. **Install Dependencies**:
   ```bash
   # Using UV (preferred)
   curl -LsSf https://astral.sh/uv/install.sh | sh
   uv sync

   # Or using pip
   pip install azure-ai-documentintelligence azure-search-documents azure-identity python-dotenv rich openai azure-ai-projects azure-monitor-opentelemetry opentelemetry-sdk azure-ai-evaluation
   ```

3. **Configure Environment**:
   ```bash
   cp .env.template .env
   # Edit .env with your Azure service credentials
   ```

### Azure Services Setup

1. **Azure CLI Authentication**:
   ```bash
   az login
   az account set --subscription "your-subscription-id"
   ```

2. **Ensure Required Role Assignment**:
   - Your Azure account needs "Azure AI User" role or equivalent
   - Services should have proper RBAC permissions

## Project Structure

```
aifoundry-workshop/
├── agents/                      # AI Agents implementation
│   ├── rag/                     # RAG Agent module
│   │   ├── rag_agent.py         # Main RAG agent implementation
│   │   └── README.md            # Module documentation
│   └── evaluations/             # Evaluation framework
│       ├── rag/                 # RAG evaluation modules
│       └── data/                # Evaluation datasets
├── aisearch/                    # Azure AI Search utilities
│   ├── create_search_index.py   # Index creation
│   ├── ingest_documents.py      # Document processing and ingestion
│   └── README.md                # Module documentation
├── docintel/                    # Document Intelligence utilities
│   ├── pdf-2-md.py             # PDF to Markdown conversion
│   ├── data/                   # Sample documents
│   └── README.md               # Module documentation
├── .devcontainer/              # Development container configuration
├── .vscode/                    # VS Code settings and debug configs
├── docs/                       # Additional documentation
├── pyproject.toml              # Python project configuration
└── README.md                   # Main project documentation
```

## Coding Standards

### Python Style Guide

- Follow [PEP 8](https://pep8.org/) style guidelines
- Use type hints for function parameters and return values
- Maximum line length: 88 characters (Black formatter default)
- Use descriptive variable and function names

### Code Formatting

We use the following tools for code quality:

```bash
# Format code with Black
black .

# Sort imports with isort
isort .

# Lint with flake8
flake8 .

# Type checking with mypy
mypy .
```

### Naming Conventions

- **Classes**: PascalCase (`RAGAgent`, `DocumentProcessor`)
- **Functions/Methods**: snake_case (`create_search_index`, `process_documents`)
- **Variables**: snake_case (`search_results`, `document_content`)
- **Constants**: UPPER_SNAKE_CASE (`DEFAULT_MODEL`, `MAX_RETRIES`)

### Error Handling

- Use specific exception types rather than generic `Exception`
- Include detailed error messages with context
- Implement proper logging for debugging
- Use exponential backoff for retrying API calls

Example:
```python
import logging
from azure.core.exceptions import ServiceRequestError

logger = logging.getLogger(__name__)

try:
    result = azure_service.call_api()
except ServiceRequestError as e:
    logger.error(f"Azure service call failed: {e}")
    raise ServiceRequestError(f"Failed to process request: {e}") from e
```

## Testing Guidelines

### Test Structure

- Unit tests for individual functions and classes
- Integration tests for Azure service interactions
- End-to-end tests for complete workflows

### Test Environment

```bash
# Set test environment variables
export AZURE_TEST_MODE="playback"  # For recorded tests
export AZURE_SKIP_LIVE_TESTS="true"  # Skip live API calls

# Run tests
python -m pytest tests/
```

### Writing Tests

1. **Test Naming**: Use descriptive test names that explain the scenario
   ```python
   def test_rag_agent_returns_relevant_response_for_valid_query():
       pass
   ```

2. **Mocking**: Mock Azure service calls for unit tests
   ```python
   @patch('azure.search.documents.SearchClient')
   def test_search_documents(mock_client):
       mock_client.search.return_value = mock_search_results
       # Test logic here
   ```

3. **Test Data**: Use small, representative test datasets
   - Store test data in `tests/data/` directory
   - Use synthetic data that doesn't contain sensitive information

### Test Categories

- **Unit Tests**: Test individual functions in isolation
- **Integration Tests**: Test interaction with Azure services
- **Evaluation Tests**: Test RAG evaluation metrics
- **End-to-End Tests**: Test complete user workflows

## Pull Request Process

### Before Submitting

1. **Test Your Changes**:
   ```bash
   # Run tests
   python -m pytest

   # Run linting
   flake8 .
   black --check .
   ```

2. **Update Documentation**:
   - Update relevant README files
   - Add docstrings to new functions/classes
   - Update environment variable documentation if needed

3. **Test in Multiple Environments**:
   - Test locally and in development container
   - Verify environment variable configurations
   - Test with minimal required permissions

### Pull Request Guidelines

1. **Branch Naming**:
   - Feature: `feature/add-new-evaluator`
   - Bug fix: `fix/search-indexing-issue`
   - Documentation: `docs/update-setup-guide`

2. **Commit Messages**:
   - Use imperative mood: "Add new evaluator" not "Added new evaluator"
   - Keep first line under 50 characters
   - Include detailed description if needed

3. **PR Description**:
   - Clearly describe the changes made
   - Include any breaking changes
   - Add screenshots for UI changes
   - Reference related issues: "Fixes #123"

4. **Code Review Checklist**:
   - [ ] Code follows style guidelines
   - [ ] Tests pass
   - [ ] Documentation updated
   - [ ] No sensitive data exposed
   - [ ] Error handling implemented
   - [ ] Logging added for debugging

### PR Review Process

1. All PRs require at least one review
2. Address review feedback promptly
3. Ensure CI/CD checks pass
4. Squash commits before merging (if requested)

## Documentation Guidelines

### Documentation Types

1. **API Documentation**: Docstrings for all public functions/classes
2. **Module Documentation**: README.md files for each module
3. **Tutorial Documentation**: Step-by-step guides
4. **Architecture Documentation**: High-level design and flow diagrams

### Docstring Format

Use Google-style docstrings:

```python
def search_documents(query: str, top_k: int = 5) -> List[SearchResult]:
    """Search for documents using hybrid search.
    
    Args:
        query: The search query string.
        top_k: Maximum number of results to return.
        
    Returns:
        List of search results with scores and content.
        
    Raises:
        SearchServiceError: If the search service is unavailable.
        
    Example:
        >>> results = search_documents("What is GPT-4?", top_k=3)
        >>> print(f"Found {len(results)} results")
    """
```

### README Structure

Each module should have a README.md with:

1. **Overview**: Brief description of the module
2. **Features**: Key capabilities and functionalities  
3. **Quick Start**: Minimal example to get started
4. **Usage Examples**: Detailed code examples
5. **Configuration**: Environment variables and settings
6. **Dependencies**: Required packages and services

## Debugging and Development Tools

### VS Code Configuration

The project includes comprehensive VS Code configurations:

- **Debug Configurations**: Pre-configured for all modules
- **Launch Settings**: Easy debugging of individual components
- **Extensions**: Recommended extensions for Python development

### Debug Configurations Available

1. **Debug RAG Agent**: Debug the main RAG agent implementation
2. **Debug Evaluation Module**: Debug evaluation framework
3. **Debug Document Ingestion**: Debug AI Search integration
4. **Debug PDF Processing**: Debug Document Intelligence

### Logging

Enable debug logging during development:

```python
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Module-specific logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
```

### Azure Service Debugging

#### Tracing and Observability

The project uses Azure Monitor OpenTelemetry for tracing:

```python
# Enable detailed tracing
export AZURE_TRACING_GEN_AI_CONTENT_RECORDING_ENABLED=true

# View traces in Azure AI Foundry portal
# Navigate to: Your Project > Tracing
```

#### Common Debug Scenarios

1. **Authentication Issues**:
   ```bash
   az account show  # Verify logged in account
   az account list-locations  # Test API access
   ```

2. **Search Index Issues**:
   ```bash
   # Test search service connectivity
   python -c "from azure.search.documents import SearchClient; print('Search client works')"
   ```

3. **Document Intelligence Issues**:
   ```bash
   # Test document intelligence service
   python -c "from azure.ai.documentintelligence import DocumentIntelligenceClient; print('Doc Intel works')"
   ```

### Performance Profiling

For performance analysis:

```python
import cProfile
import pstats

# Profile a function
profiler = cProfile.Profile()
profiler.enable()

# Your code here
result = rag_agent.ask("What is GPT-4?")

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative').print_stats(10)
```

## Common Development Tasks

### Adding a New Evaluator

1. Create evaluator class in `agents/evaluations/`
2. Implement required interface methods
3. Add unit tests
4. Update evaluation documentation
5. Add usage examples

### Adding New Document Types

1. Extend document processing in `docintel/`
2. Update search indexing in `aisearch/`
3. Test with sample documents
4. Update documentation

### Extending RAG Agent

1. Add new methods to `RAGAgent` class
2. Implement proper error handling and logging
3. Add tracing spans for observability
4. Update agent documentation

## Getting Help

- **Issues**: Create GitHub issues for bugs or feature requests
- **Discussions**: Use GitHub Discussions for questions
- **Documentation**: Check module-specific README files
- **Azure Support**: Use Azure support channels for service-specific issues

## Code of Conduct

- Be respectful and inclusive
- Focus on constructive feedback
- Help others learn and grow
- Follow Azure community guidelines

Thank you for contributing to the Azure AI Foundry Workshop! Your contributions help make this educational resource better for everyone.