#!/usr/bin/env python3
"""
RAG Agent Evaluation Example

This script demonstrates how to use the RAG Agent Evaluation module
in a step-by-step manner with proper error handling and logging.

Usage:
    python example_evaluation.py

Prerequisites:
    1. Set up environment variables (see .env.template)
    2. Ensure Azure services are properly configured
    3. Have a valid evaluation dataset
"""

import os
import sys
import logging
from pathlib import Path
from typing import Optional

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from dotenv import load_dotenv

# Configure logging for demonstration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_environment() -> bool:
    """Check if all required environment variables are set."""
    required_vars = [
        'AZURE_SUBSCRIPTION_ID',
        'AZURE_RESOURCE_GROUP',
        'AZURE_PROJECT_NAME', 
        'AZURE_PROJECT_ENDPOINT',
        'AZURE_SEARCH_SERVICE_NAME',
        'AZURE_SEARCH_INDEX_NAME',
        'AZURE_OPENAI_ENDPOINT'
    ]
    
    logger.info("Checking environment variables...")
    missing_vars = []
    
    for var in required_vars:
        if not os.environ.get(var):
            missing_vars.append(var)
            logger.error("Missing environment variable: %s", var)
        else:
            logger.info("✅ %s is set", var)
    
    if missing_vars:
        logger.error("Missing %d required environment variables", len(missing_vars))
        logger.error("Please set up your .env file based on .env.template")
        return False
    
    logger.info("✅ All environment variables are set")
    return True


def run_evaluation_example():
    """Run a complete RAG agent evaluation example."""
    try:
        # Load environment variables
        load_dotenv()
        
        # Check environment
        if not check_environment():
            logger.error("Environment check failed. Please configure your environment.")
            return False
        
        # Import Azure and evaluation modules
        logger.info("Importing Azure AI modules...")
        from azure.identity import DefaultAzureCredential
        from azure.ai.projects import AIProjectClient
        from agents.rag.rag_agent import RAGAgent
        from rag_agent_eval import RAGAgentEvaluator
        
        # Initialize Azure AI Project Client
        logger.info("Initializing Azure AI Project Client...")
        project_client = AIProjectClient(
            subscription_id=os.environ["AZURE_SUBSCRIPTION_ID"],
            resource_group_name=os.environ["AZURE_RESOURCE_GROUP"],
            project_name=os.environ["AZURE_PROJECT_NAME"],
            credential=DefaultAzureCredential(),
            endpoint=os.environ["AZURE_PROJECT_ENDPOINT"],
        )
        logger.info("✅ Project client initialized")
        
        # Initialize RAG Agent
        logger.info("Initializing RAG Agent...")
        rag_agent = RAGAgent(
            search_service_name=os.environ["AZURE_SEARCH_SERVICE_NAME"],
            search_index_name=os.environ["AZURE_SEARCH_INDEX_NAME"],
            azure_openai_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            chat_model=os.environ.get("EVALUATION_MODEL", "gpt-4"),
            top_k_documents=3
        )
        logger.info("✅ RAG agent initialized")
        
        # Initialize evaluator
        logger.info("Initializing RAG Agent Evaluator...")
        evaluator = RAGAgentEvaluator(
            rag_agent=rag_agent,
            project_client=project_client,
            model_config={
                "model": os.environ.get("EVALUATION_MODEL", "gpt-4"),
                "api_version": os.environ.get("EVALUATION_API_VERSION", "2024-12-01-preview")
            }
        )
        logger.info("✅ Evaluator initialized")
        
        # Define paths
        base_path = Path(__file__).parent.parent
        dataset_path = base_path / "data" / "single-turn-eval-ds.jsonl"
        output_path = base_path / "data" / "output" / "single-turn-eval-ds-agent-output.jsonl"
        results_path = base_path / "data" / "output" / "evaluation_results.json"
        
        # Verify dataset exists
        if not dataset_path.exists():
            logger.error("Evaluation dataset not found at: %s", dataset_path)
            return False
        
        # Load evaluation dataset
        logger.info("Loading evaluation dataset...")
        dataset = evaluator.load_evaluation_dataset(str(dataset_path))
        logger.info("✅ Loaded %d evaluation data points", len(dataset))
        
        # Test with a smaller subset first (first 3 queries)
        test_dataset = dataset[:3]
        logger.info("Running evaluation on first %d queries for testing...", len(test_dataset))
        
        # Run RAG agent on test queries
        logger.info("🤖 Running RAG agent on test dataset...")
        dataset_with_responses = evaluator.single_turn_agent_run(
            dataset=test_dataset,
            output_path=str(output_path),
            max_retries=int(os.environ.get("EVALUATION_MAX_RETRIES", "3"))
        )
        
        # Check if we got responses
        successful_responses = [dp for dp in dataset_with_responses if dp.agent_response and not dp.agent_response.startswith("ERROR:")]
        logger.info("✅ Got %d successful responses out of %d queries", len(successful_responses), len(test_dataset))
        
        if not successful_responses:
            logger.error("No successful responses to evaluate")
            return False
        
        # Evaluate responses
        logger.info("📊 Evaluating agent responses...")
        evaluation_results = evaluator.evaluate_dataset(dataset_with_responses)
        
        # Save results
        logger.info("💾 Saving evaluation results...")
        evaluator.save_evaluation_results(evaluation_results, str(results_path))
        
        # Print summary
        logger.info("📋 Evaluation Summary:")
        evaluator.print_evaluation_summary(evaluation_results)
        
        logger.info("🎉 Evaluation example completed successfully!")
        logger.info("Results saved to: %s", results_path)
        logger.info("Agent outputs saved to: %s", output_path)
        
        return True
        
    except ImportError as e:
        logger.error("Import error: %s", str(e))
        logger.error("Please ensure all dependencies are installed")
        return False
    except Exception as e:
        logger.error("Evaluation failed: %s", str(e))
        logger.error("Check your Azure configuration and network connectivity")
        return False


def main():
    """Main function for the evaluation example."""
    print("🚀 RAG Agent Evaluation Example")
    print("=" * 50)
    
    success = run_evaluation_example()
    
    if success:
        print("\n✅ Evaluation example completed successfully!")
        print("You can now run the full evaluation using:")
        print("   python run_evaluation.py")
    else:
        print("\n❌ Evaluation example failed!")
        print("Please check the logs above and fix any issues.")
        print("Make sure to:")
        print("1. Set up your .env file based on .env.template")
        print("2. Ensure Azure services are accessible")
        print("3. Check your network connectivity")
        sys.exit(1)


if __name__ == "__main__":
    main()
