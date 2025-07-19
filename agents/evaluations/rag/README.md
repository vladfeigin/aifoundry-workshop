# RAG Agent Evaluation Module

This module provides comprehensive evaluation capabilities for RAG (Retrieval Augmented Generation) agents with support for both local and cloud-based evaluation workflows. It implements multiple evaluation metrics and automated testing workflows to assess RAG agent performance across various dimensions.

## 🚀 Quick Start

### Local Evaluation
1. **Set up environment variables** (see Environment Setup below)
2. **Run local evaluation**: `python -m agents.evaluations.rag.rag_agent_eval`
3. **Check results** in `../data/output/` directory

### Cloud Evaluation
1. **Set up environment variables** (see Environment Setup below)
2. **Run cloud evaluation**: `python -m agents.evaluations.rag.rag_agent_eval_azure`
3. **Monitor results** in Azure AI Foundry portal

## 🏗️ Architecture

The evaluation module consists of two main evaluation workflows:

### Local Evaluation (`rag_agent_eval.py`)
- **RAGAgentEvaluator**: Main evaluation orchestrator running locally
- **Azure AI Evaluation SDK**: Direct local evaluation using Azure AI evaluators
- **Immediate Results**: Get evaluation results immediately in local environment
- **Best for**: Development, debugging, and quick iterations

### Cloud Evaluation (`rag_agent_eval_azure.py`)
- **RAGAgentCloudEvaluator**: Cloud-based evaluation orchestrator
- **Azure AI Foundry**: Scalable cloud evaluation platform
- **Distributed Processing**: Leverage cloud compute for large-scale evaluation
- **Best for**: Production, CI/CD pipelines, and large datasets

## 📊 Evaluation Metrics

Both local and cloud evaluation workflows support the same comprehensive set of metrics:

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

### 3. Response Completeness
**Purpose**: Assesses if the response fully addresses all aspects of the query
- **Scale**: 0.0 - 1.0 (higher is better)
- **Implementation**: Azure AI Evaluation SDK CompletenessEvaluator
- **Focus**: Comprehensiveness and thoroughness

### 4. Intent Resolution
**Purpose**: Determines if the response successfully resolves the user's underlying intent
- **Scale**: 0.0 - 1.0 (higher is better)
- **Implementation**: Azure AI Evaluation SDK IntentResolutionEvaluator
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
export PROJECT_ENDPOINT="your-project-endpoint"
export AZURE_SEARCH_SERVICE_NAME="your-search-service"
export AZURE_SEARCH_INDEX_NAME="your-search-index"
export AZURE_OPENAI_ENDPOINT="your-openai-endpoint"
export AZURE_OPENAI_API_KEY="your-openai-api-key"
export AZURE_EVALUATION_MODEL="gpt-4o"
```

2. **Dependencies**: Ensure all packages are installed
```bash
uv sync  # or pip install -r requirements.txt
```

### Local Evaluation Workflow

**Best for**: Development, debugging, and quick iterations

```bash
# Run local evaluation
cd /path/to/your/project
python -m agents.evaluations.rag.rag_agent_eval

# Results will be saved to:
# - agents/evaluations/data/output/single-turn-eval-ds-agent-output.jsonl
# - agents/evaluations/data/output/evaluation_results.json
```

**Features**:
- ✅ Immediate local results
- ✅ Detailed debugging information
- ✅ Full control over evaluation process
- ✅ Works offline after initial setup

### Cloud Evaluation Workflow

**Best for**: Production, CI/CD pipelines, and large datasets

```bash
# Run cloud evaluation
cd /path/to/your/project
python -m agents.evaluations.rag.rag_agent_eval_azure

# Results will be available in:
# - Azure AI Foundry portal
# - agents/evaluations/data/output/cloud-evaluation-dataset.jsonl
```

**Features**:
- ✅ Scalable cloud compute
- ✅ Integrated with Azure AI Foundry
- ✅ Automatic result logging
- ✅ Perfect for CI/CD pipelines
- ✅ No local compute constraints

### Choosing Between Local and Cloud

| Criteria | Local Evaluation | Cloud Evaluation |
|----------|------------------|------------------|
| **Speed** | Fast for small datasets | Scales with dataset size |
| **Setup** | Simple, minimal dependencies | Requires Azure AI Foundry setup |
| **Cost** | Only API calls | API calls + cloud compute |
| **Debugging** | Full local control | Limited debugging capabilities |
| **CI/CD** | Basic CI/CD support | Optimized for CI/CD pipelines |
| **Scalability** | Limited by local resources | Unlimited cloud scalability |

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

#### Local Evaluation
```python
from agents.evaluations.rag.rag_agent_eval import RAGAgentEvaluator
from agents.rag.rag_agent import RAGAgentService
from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential

# Initialize components
project_client = AIProjectClient(
    endpoint=os.environ["PROJECT_ENDPOINT"],
    credential=DefaultAzureCredential()
)
rag_agent = RAGAgentService()
evaluator = RAGAgentEvaluator(rag_agent, project_client)

# Load and process dataset
dataset = evaluator.load_evaluation_dataset("eval-ds.jsonl")
dataset_with_responses = evaluator.single_turn_agent_run(dataset)

# Run evaluation locally
results = evaluator.evaluate_dataset(dataset_with_responses)
evaluator.print_evaluation_summary(results)
```

#### Cloud Evaluation
```python
from agents.evaluations.rag.rag_agent_eval_azure import RAGAgentCloudEvaluator
from agents.rag.rag_agent import RAGAgentService
from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential

# Initialize components
project_client = AIProjectClient(
    endpoint=os.environ["PROJECT_ENDPOINT"],
    credential=DefaultAzureCredential()
)
rag_agent = RAGAgentService()
evaluator = RAGAgentCloudEvaluator(rag_agent, project_client)

# Load and process dataset
dataset = evaluator.load_evaluation_dataset("eval-ds.jsonl")
dataset_with_responses = evaluator.generate_agent_responses(dataset)

# Prepare and upload dataset for cloud evaluation
prepared_path = evaluator.prepare_evaluation_dataset(dataset_with_responses, "cloud-eval.jsonl")
data_id = evaluator.upload_dataset(prepared_path)

# Configure and run cloud evaluation
evaluators = evaluator.configure_evaluators()
evaluation_name = evaluator.run_cloud_evaluation(data_id, evaluators)

# Monitor and retrieve results
status = evaluator.monitor_evaluation_status(evaluation_name)
if status["status"] == "Completed":
    results = evaluator.get_evaluation_results(evaluation_name)
    evaluator.print_evaluation_summary(results)
```

## 📁 File Structure

```
agents/evaluations/rag/
├── rag_agent_eval.py              # Local evaluation module with RAGAgentEvaluator class
├── rag_agent_eval_azure.py        # Cloud evaluation module with RAGAgentCloudEvaluator class
├── README.md                      # This documentation
└── __init__.py                    # Package init
```

```
agents/evaluations/data/
├── single-turn-eval-ds.jsonl                    # Evaluation dataset
└── output/
    ├── single-turn-eval-ds-agent-output.jsonl   # Agent responses (local evaluation)
    ├── evaluation_results.json                  # Evaluation results (local evaluation)
    └── cloud-evaluation-dataset.jsonl           # Prepared dataset for cloud evaluation
```

## 🔧 Configuration

### Local Evaluation Configuration
```python
# In rag_agent_eval.py
evaluator = RAGAgentEvaluator(
    rag_agent=rag_agent,
    project_client=project_client,
    model_config={"model": "gpt-4o", "api_version": "2024-12-01-preview"}
)
```

### Cloud Evaluation Configuration
```python
# In rag_agent_eval_azure.py
evaluator = RAGAgentCloudEvaluator(
    rag_agent=rag_agent,
    project_client=project_client,
    model_config={"model": "gpt-4o", "api_version": "2024-12-01-preview"}
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

## ☁️ Cloud Evaluation Features

### Azure AI Foundry Integration
- **Automatic Dataset Upload**: Upload evaluation datasets to Azure AI Foundry
- **Centralized Evaluation**: Run evaluations on Azure cloud infrastructure
- **Result Tracking**: Automatic logging and tracking of evaluation results
- **Scalable Compute**: Handle large datasets without local resource constraints

### Monitoring and Status Tracking
```python
# Monitor evaluation progress
status_result = evaluator.monitor_evaluation_status(evaluation_name, timeout=1800)

# Check evaluation status
if status_result["status"] == "Completed":
    results = evaluator.get_evaluation_results(evaluation_name)
    evaluator.print_evaluation_summary(results)
```

### Cloud Evaluation Benefits
- **Scalability**: Handle datasets of any size
- **Consistency**: Reproducible evaluation environment
- **Integration**: Seamless integration with Azure AI Foundry
- **Automation**: Perfect for automated CI/CD pipelines
- **Centralized Results**: All results stored in Azure AI Foundry portal

### When to Use Cloud Evaluation
- ✅ Large datasets (>100 queries)
- ✅ Production evaluation pipelines
- ✅ CI/CD integration requirements
- ✅ Team collaboration needs
- ✅ Centralized result management
- ✅ Compliance and audit requirements

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

## 🔄 Evaluation Workflows

### Local Evaluation Workflow

1. **Dataset Loading**: Load evaluation dataset from JSONL file
2. **Agent Execution**: Run RAG agent on all queries with retry logic
3. **Response Collection**: Gather agent responses and metadata
4. **Local Evaluation**: Apply all metrics locally using Azure AI Evaluation SDK
5. **Analysis**: Calculate summary statistics and rankings
6. **Reporting**: Generate comprehensive evaluation report and save locally

**Key Features**:
- Direct evaluation using Azure AI Evaluation SDK
- Immediate results available locally
- Full debugging capabilities
- Perfect for development and testing

### Cloud Evaluation Workflow

1. **Dataset Loading**: Load evaluation dataset from JSONL file
2. **Agent Execution**: Run RAG agent on all queries with retry logic
3. **Dataset Preparation**: Format dataset for cloud evaluation
4. **Dataset Upload**: Upload prepared dataset to Azure AI Foundry
5. **Evaluator Configuration**: Configure cloud evaluators with proper data mapping
6. **Cloud Evaluation**: Submit evaluation job to Azure AI Foundry
7. **Status Monitoring**: Monitor evaluation progress with polling
8. **Result Retrieval**: Retrieve results from Azure AI Foundry portal

**Key Features**:
- Scalable cloud compute for large datasets
- Integrated with Azure AI Foundry platform
- Automatic result logging and tracking
- Perfect for production and CI/CD pipelines

### Evaluation Metrics Comparison

Both workflows use the same evaluation metrics but with different execution environments:

| Metric | Local Implementation | Cloud Implementation |
|--------|---------------------|---------------------|
| **Groundedness** | `GroundednessEvaluator` (local) | `EvaluatorIds.GROUNDEDNESS` (cloud) |
| **Relevance** | `RelevanceEvaluator` (local) | `EvaluatorIds.RELEVANCE` (cloud) |
| **Completeness** | `ResponseCompletenessEvaluator` (local) | `EvaluatorIds.COMPLETENESS` (cloud) |
| **Intent Resolution** | `IntentResolutionEvaluator` (local) | `EvaluatorIds.INTENT_RESOLUTION` (cloud) |

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
