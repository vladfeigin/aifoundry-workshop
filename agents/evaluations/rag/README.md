# RAG Agent Evaluation Module

This module provides comprehensive evaluation capabilities for RAG (Retrieval Augmented Generation) agents using Azure AI Evaluation SDK. It implements multiple evaluation metrics and automated testing workflows to assess RAG agent performance across various dimensions.

## 🚀 Quick Start

1. **Set up environment variables** (see Environment Setup below)
2. **Run evaluation**: `python run_evaluation.py`
3. **Check results** in `../data/output/` directory

## 🏗️ Architecture

The evaluation module consists of several key components:

- **RAGAgentEvaluator**: Main evaluation orchestrator
- **Built-in Evaluators**: Azure AI Evaluation SDK metrics (Groundedness, Relevance)
- **Custom Evaluators**: Response Completeness and Intent Resolution
- **Dataset Processing**: JSONL dataset loading and agent response generation
- **Result Analysis**: Comprehensive reporting and analytics

## 📊 Evaluation Metrics

### 1. Groundedness
**Purpose**: Measures if the response is supported by the retrieved context
- **Scale**: 0.0 - 1.0 (higher is better)
- **Implementation**: Azure AI Evaluation SDK GroundednessEvaluator
- **Focus**: Factual accuracy and context alignment

### 2. Relevance
**Purpose**: Evaluates how relevant the response is to the user query
- **Scale**: 0.0 - 1.0 (higher is better)
- **Implementation**: Azure AI Evaluation SDK RelevanceEvaluator
- **Focus**: Query-response relevance and appropriateness

### 3. Response Completeness (Custom)
**Purpose**: Assesses if the response fully addresses all aspects of the query
- **Scale**: 0.0 - 1.0 (higher is better)
- **Implementation**: Custom GPT-4 powered evaluator
- **Focus**: Comprehensiveness and thoroughness

### 4. Intent Resolution (Custom)
**Purpose**: Determines if the response successfully resolves the user's underlying intent
- **Scale**: 0.0 - 1.0 (higher is better)
- **Implementation**: Custom GPT-4 powered evaluator
- **Focus**: User satisfaction and information need fulfillment

## 🗂️ Dataset Format

The evaluation dataset uses JSONL format with the following schema:

```json
{
  "query": "What is GPT-4?",
  "context": "GPT-4 Technical Report - Abstract: We report the development of GPT-4...",
  "ground_truth": "GPT-4 is a large-scale, multimodal model developed by OpenAI..."
}
```

### Fields:
- **query**: User's question or request
- **context**: Reference context that should inform the answer
- **ground_truth**: Expected comprehensive answer for comparison

## 🚀 Quick Start

### Prerequisites

1. **Environment Variables**: Set up required Azure credentials
```bash
export AZURE_SUBSCRIPTION_ID="your-subscription-id"
export AZURE_RESOURCE_GROUP="your-resource-group"
export AZURE_PROJECT_NAME="your-project-name"
export AZURE_PROJECT_ENDPOINT="your-project-endpoint"
export AZURE_SEARCH_SERVICE_NAME="your-search-service"
export AZURE_SEARCH_INDEX_NAME="your-search-index"
export AZURE_OPENAI_ENDPOINT="your-openai-endpoint"
```

2. **Dependencies**: Ensure all packages are installed
```bash
uv sync  # or pip install -r requirements.txt
```

### Basic Usage

```bash
# Run evaluation with default settings
cd agents/evaluations/rag
python run_evaluation.py
```

### Environment Setup

Set up your environment variables in your shell or `.env` file:

```bash
export AZURE_SUBSCRIPTION_ID="your-subscription-id"
export AZURE_RESOURCE_GROUP="your-resource-group"
export AZURE_PROJECT_NAME="your-project-name"
export AZURE_PROJECT_ENDPOINT="your-project-endpoint"
export AZURE_SEARCH_SERVICE_NAME="your-search-service"
export AZURE_SEARCH_INDEX_NAME="your-search-index"
export AZURE_OPENAI_ENDPOINT="your-openai-endpoint"
```

### Programmatic Usage

```python
import asyncio
from rag_agent_eval import RAGAgentEvaluator
from agents.rag.rag_agent import RAGAgent

# Initialize components
rag_agent = RAGAgent(...)
evaluator = RAGAgentEvaluator(rag_agent, project_client)

# Load and process dataset
dataset = evaluator.load_evaluation_dataset("eval-ds.jsonl")
dataset_with_responses = evaluator.single_turn_agent_run(dataset)

# Run evaluation
results = await evaluator.evaluate_dataset(dataset_with_responses)
evaluator.print_evaluation_summary(results)
```

## 📁 File Structure

```
agents/evaluations/rag/
├── rag_agent_eval.py          # Main evaluation module with RAGAgentEvaluator class
├── run_evaluation.py          # Simple CLI interface to run evaluations
├── README.md                  # This documentation
└── __init__.py                # Package init (optional)
```

```
agents/evaluations/data/
├── single-turn-eval-ds.jsonl              # Evaluation dataset
└── output/
    ├── single-turn-eval-ds-agent-output.jsonl  # Agent responses
    └── evaluation_results.json                 # Evaluation results
```

## 🔧 Configuration

### Model Configuration
```python
model_config = {
    "model": "gpt-4",
    "api_version": "2024-12-01-preview"
}
```

### RAG Agent Configuration
```python
rag_agent = RAGAgent(
    search_service_name="your-search-service",
    search_index_name="your-index",
    azure_openai_endpoint="your-endpoint",
    chat_model="gpt-4",
    top_k_documents=3
)
```

### Evaluation Parameters
- **batch_size**: Number of concurrent evaluations (default: 5)
- **max_retries**: Retry attempts for failed queries (default: 3)
- **top_k_documents**: Number of retrieved documents (default: 3)

## 📈 Results Format

### Evaluation Results JSON Structure
```json
{
  "dataset_size": 10,
  "evaluation_metrics": {
    "groundedness": {"mean": 0.85, "min": 0.6, "max": 1.0, "count": 10},
    "relevance": {"mean": 0.82, "min": 0.7, "max": 0.95, "count": 10},
    "completeness": {"mean": 0.78, "min": 0.5, "max": 0.9, "count": 10},
    "intent_resolution": {"mean": 0.80, "min": 0.6, "max": 0.95, "count": 10}
  },
  "individual_scores": [...],
  "summary_statistics": {...},
  "evaluation_time": 45.2,
  "timestamp": "2024-01-15T10:30:00"
}
```

### Agent Output JSONL Structure
```json
{
  "query": "What is GPT-4?",
  "context": "GPT-4 Technical Report...",
  "ground_truth": "GPT-4 is a large-scale...",
  "agent_response": "GPT-4 is a state-of-the-art...",
  "sources": [{"docid": "doc1", "score": 0.95, "content_preview": "..."}],
  "response_time": 2.3
}
```

## 🔍 Monitoring and Debugging

### Azure Application Insights Integration
The evaluation module automatically integrates with Azure Application Insights for:
- Request tracing and performance monitoring
- Error tracking and diagnostics
- Custom metrics and telemetry

### Logging Levels
```python
import logging
logging.getLogger('rag_agent_eval').setLevel(logging.DEBUG)
```

### Common Issues and Solutions

**Issue**: Authentication errors
```bash
az login  # Ensure Azure CLI authentication
```

**Issue**: Missing dependencies
```bash
uv add azure-ai-evaluation  # Add evaluation SDK
```

**Issue**: API rate limits
- Reduce batch_size parameter
- Add delays between requests
- Use exponential backoff

## 🔄 Workflow

1. **Dataset Loading**: Load evaluation dataset from JSONL file
2. **Agent Execution**: Run RAG agent on all queries with retry logic
3. **Response Collection**: Gather agent responses and metadata
4. **Evaluation**: Apply all metrics to agent responses
5. **Analysis**: Calculate summary statistics and rankings
6. **Reporting**: Generate comprehensive evaluation report

## 📊 Performance Benchmarks

### Expected Performance
- **Processing Speed**: ~2-3 seconds per query (including retrieval + generation)
- **Evaluation Speed**: ~1-2 seconds per metric per query
- **Memory Usage**: ~100-200MB for typical datasets (10-100 queries)

### Optimization Tips
- Use batch processing for large datasets
- Enable concurrent evaluations (batch_size parameter)
- Cache retrieved contexts when possible
- Monitor API quotas and rate limits

## 🛠️ Customization

### Adding Custom Metrics
```python
class CustomEvaluator:
    def __init__(self, project_client, model_config):
        self.project_client = project_client
        self.model_config = model_config
    
    async def __call__(self, query, response, context):
        # Custom evaluation logic
        return CustomScore(score=0.85, reasoning="...")
```

### Custom Dataset Processing
```python
def load_custom_dataset(file_path):
    # Custom dataset loading logic
    return [EvaluationDataPoint(...)]
```

## 🔐 Security and Best Practices

### Authentication
- **Production**: Use Managed Identity
- **Development**: Azure CLI authentication
- **CI/CD**: Service Principal

### Data Protection
- Evaluation data is processed in-memory
- No sensitive data is logged by default
- Results are saved locally unless configured otherwise

### Error Handling
- Comprehensive retry logic for transient failures
- Graceful degradation for individual query failures
- Detailed error logging and reporting

## 📚 References

- [Azure AI Evaluation SDK Documentation](https://learn.microsoft.com/en-us/azure/ai-studio/how-to/develop/evaluate-sdk)
- [Azure AI Foundry Tracing](https://learn.microsoft.com/en-us/azure/ai-studio/how-to/develop/trace-production-sdk)
- [RAG Evaluation Best Practices](https://learn.microsoft.com/en-us/azure/ai-studio/concepts/evaluation-approach-gen-ai)
- [Azure OpenTelemetry Integration](https://learn.microsoft.com/en-us/azure/azure-monitor/app/opentelemetry-enable)

## 🤝 Contributing

When extending the evaluation module:

1. Follow Azure coding best practices
2. Add comprehensive error handling
3. Include proper logging and tracing
4. Write unit tests for new evaluators
5. Update documentation for new features

## 📝 License

This module follows the same license as the parent project.
