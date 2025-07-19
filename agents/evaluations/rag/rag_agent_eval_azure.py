"""
RAG Agent Cloud Evaluation Module

This module provides cloud-based evaluation capabilities for the RAG agent using Azure AI Foundry SDK.
It follows the official Azure AI Foundry SDK documentation pattern for running evaluations in the cloud.

Features:
- Cloud-based evaluation using Azure AI Foundry SDK
- Automated dataset upload and management
- Multiple evaluation metrics: Groundedness, Relevance, Completeness, Intent Resolution
- Automatic result logging to Azure AI Foundry project
- Support for large datasets without local compute constraints
- Integration with CI/CD pipelines

Metrics:
- Groundedness: Measures if the response is supported by the retrieved context
- Relevance: Evaluates how relevant the response is to the user query
- Completeness: Assesses if the response fully addresses the query
- Intent Resolution: Determines if the response resolves the user's intent

Usage:
    python -m agents.evaluations.rag.rag_agent_eval_azure from project root directory
"""

import os
import json
import time
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

# Add parent directories to path for importing RAG agent
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

# Azure AI Foundry SDK imports
from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import (
    EvaluatorConfiguration,
    EvaluatorIds,
    Evaluation,
    InputDataset
)
from dotenv import load_dotenv

# Import our RAG agent
from agents.rag.rag_agent import RAGAgentService, RAGResponse

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_agent_responses(dataset_path: str, output_path: str) -> str:
    """
    Generate RAG agent responses for all queries in the dataset.
    
    Args:
        dataset_path: Path to the evaluation dataset
        output_path: Path to save dataset with agent responses
        
    Returns:
        Path to the dataset with agent responses
    """
    logger.info("🤖 Generating RAG agent responses...")
    
    # Initialize RAG Agent
    rag_agent = RAGAgentService()
    
    # Load evaluation dataset
    dataset = []
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                dataset.append(json.loads(line))
    
    logger.info("Loaded %d evaluation queries", len(dataset))
    
    # Generate responses
    results = []
    for i, item in enumerate(dataset, 1):
        logger.info("Processing query %d/%d: %s...", i, len(dataset), item["query"][:50])
        
        try:
            # Get response from RAG agent
            rag_response: RAGResponse = rag_agent.ask(item["query"])
            
            # Create evaluation record
            eval_record = {
                "query": item["query"],
                "response": rag_response.answer,
                "context": item["context"],
                "ground_truth": item["ground_truth"]
            }
            results.append(eval_record)
            
            logger.info("✅ Query %d completed", i)
            
        except Exception as e:
            logger.error("❌ Query %d failed: %s", i, str(e))
            # Add failed record
            eval_record = {
                "query": item["query"],
                "response": f"ERROR: {str(e)}",
                "context": item["context"],
                "ground_truth": item["ground_truth"]
            }
            results.append(eval_record)
    
    # Save results
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for record in results:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    
    logger.info("✅ Saved %d evaluation records to %s", len(results), output_path)
    return output_path


def main():
    """Main cloud evaluation workflow following Azure AI Foundry SDK documentation."""
    try:
        logger.info("🚀 Starting RAG Agent Cloud Evaluation")
        
        # Required environment variables (following the documentation pattern)
        endpoint = os.environ["PROJECT_ENDPOINT"]  # https://<account>.services.ai.azure.com/api/projects/<project>
        model_endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]  # https://<account>.services.ai.azure.com
        model_api_key = os.environ["AZURE_OPENAI_API_KEY"]
        model_deployment_name = os.environ.get("AZURE_EVALUATION_MODEL", "gpt-4o")  # E.g. gpt-4o
        
        # Optional: Reuse an existing dataset
        dataset_name = os.environ.get("DATASET_NAME", f"rag-agent-eval-{datetime.now().strftime('%Y%m%d-%H%M%S')}")
        dataset_version = os.environ.get("DATASET_VERSION", "1.0")
        
        logger.info("Configuration:")
        logger.info("  Project endpoint: %s", endpoint)
        logger.info("  Model endpoint: %s", model_endpoint)
        logger.info("  Model deployment: %s", model_deployment_name)
        logger.info("  Dataset name: %s", dataset_name)
        logger.info("  Dataset version: %s", dataset_version)
        
        # Create the project client (Foundry project and credentials)
        logger.info("🔗 Initializing Azure AI Project Client...")
        project_client = AIProjectClient(
            endpoint=endpoint,
            credential=DefaultAzureCredential(),
        )
        
        # Define paths
        base_path = Path(__file__).parent.parent
        dataset_path = base_path / "data" / "single-turn-eval-ds.jsonl"
        evaluation_dataset_path = base_path / "data" / "output" / "cloud-evaluation-dataset.jsonl"
        
        # Generate agent responses
        logger.info("🤖 Generating agent responses...")
        prepared_dataset_path = generate_agent_responses(str(dataset_path), str(evaluation_dataset_path))
        
        # Upload evaluation data
        logger.info("☁️ Uploading evaluation dataset...")
        data_id = project_client.datasets.upload_file(
            name=dataset_name,
            version=dataset_version,
            file_path=prepared_dataset_path,
        ).id
        
        logger.info("✅ Dataset uploaded with ID: %s", data_id)
        
        # Specify evaluators (following the documentation pattern)
        logger.info("⚙️ Configuring evaluators...")
        evaluators = {
            "groundedness": EvaluatorConfiguration(
                id=EvaluatorIds.GROUNDEDNESS.value,
                init_params={"deployment_name": model_deployment_name},
                data_mapping={
                    "query": "${data.query}",
                    "response": "${data.response}",
                    "context": "${data.context}",
                },
            ),
            "relevance": EvaluatorConfiguration(
                id=EvaluatorIds.RELEVANCE.value,
                init_params={"deployment_name": model_deployment_name},
                data_mapping={
                    "query": "${data.query}",
                    "response": "${data.response}",
                    "context": "${data.context}",
                },
            ),
            "completeness": EvaluatorConfiguration(
                id=EvaluatorIds.COMPLETENESS.value,
                init_params={"deployment_name": model_deployment_name},
                data_mapping={
                    "query": "${data.query}",
                    "response": "${data.response}",
                    "ground_truth": "${data.ground_truth}",
                },
            ),
            "intent_resolution": EvaluatorConfiguration(
                id=EvaluatorIds.INTENT_RESOLUTION.value,
                init_params={"deployment_name": model_deployment_name},
                data_mapping={
                    "query": "${data.query}",
                    "response": "${data.response}",
                    "context": "${data.context}",
                },
            ),
        }
        
        logger.info("✅ Configured %d evaluators", len(evaluators))
        
        # Submit an evaluation in the cloud (following the documentation pattern)
        logger.info("🚀 Submitting cloud evaluation...")
        
        # Create an evaluation with the dataset and evaluators specified
        evaluation = Evaluation(
            display_name=f"RAG Agent Cloud Evaluation - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            description=f"Cloud evaluation of RAG agent performance - {datetime.now().isoformat()}",
            data=InputDataset(id=data_id),
            evaluators=evaluators,
        )
        
        # Run the evaluation
        evaluation_response = project_client.evaluations.create(
            evaluation,
            headers={
                "model-endpoint": model_endpoint,
                "api-key": model_api_key,
            },
        )
        
        logger.info("✅ Cloud evaluation submitted successfully!")
        logger.info("Created evaluation: %s", evaluation_response.name)
        logger.info("Status: %s", evaluation_response.status)
        
        # Print summary
        print("\n" + "="*80)
        print("🏆 RAG AGENT CLOUD EVALUATION SUBMITTED")
        print("="*80)
        print(f"\n📊 Evaluation Information:")
        print(f"   • Evaluation Name: {evaluation_response.name}")
        print(f"   • Status: {evaluation_response.status}")
        print(f"   • Dataset ID: {data_id}")
        print(f"   • Dataset Name: {dataset_name}")
        print(f"   • Model Deployment: {model_deployment_name}")
        print(f"   • Evaluators: {', '.join(evaluators.keys())}")
        print(f"   • Timestamp: {datetime.now().isoformat()}")
        
        print(f"\n📈 Next Steps:")
        print(f"   • Monitor evaluation progress in Azure AI Foundry portal")
        print(f"   • Check evaluation results at: {endpoint.replace('/api/projects/', '/ui/projects/')}")
        print(f"   • Review detailed metrics and insights when completed")
        print(f"   • Use results for model improvement and optimization")
        
        print("\n" + "="*80)
        
        logger.info("🎉 RAG Agent cloud evaluation workflow completed!")
        
    except KeyError as e:
        logger.error("Missing required environment variable: %s", str(e))
        logger.error("Please ensure the following environment variables are set:")
        logger.error("  PROJECT_ENDPOINT=https://<account>.services.ai.azure.com/api/projects/<project>")
        logger.error("  AZURE_OPENAI_ENDPOINT=https://<account>.services.ai.azure.com")
        logger.error("  AZURE_OPENAI_API_KEY=your_api_key")
        logger.error("  AZURE_EVALUATION_MODEL=gpt-4o (optional)")
        raise
    except Exception as e:
        logger.error("Cloud evaluation failed: %s", str(e))
        raise


if __name__ == "__main__":
    main()
