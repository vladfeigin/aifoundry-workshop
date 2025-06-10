#!/usr/bin/env python3
"""
Simple CLI script to run RAG agent evaluation.

This script provides a straightforward interface to evaluate the RAG agent
using the evaluation dataset and Azure AI Evaluation SDK.

Usage:
    python run_evaluation.py
"""

import os
import sys
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from agents.evaluations.rag.rag_agent_eval import RAGAgentEvaluator
from agents.rag.rag_agent import RAGAgent
from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main evaluation workflow."""
    logger.info("🚀 Starting RAG Agent Evaluation")
    
    # Load environment variables
    env_path = project_root / '.env'
    if env_path.exists():
        load_dotenv(env_path)
        logger.info("✅ Environment variables loaded")
    else:
        logger.warning("⚠️ No .env file found, using system environment variables")
    
    try:
        # Initialize RAG agent
        logger.info("Initializing RAG agent...")
        rag_agent = RAGAgent()
        
        # Initialize Azure AI project client
        logger.info("Initializing Azure AI project client...")
        
        # Use environment variables for Azure AI project
        project_connection_string = os.environ.get("AZURE_AI_PROJECT_CONNECTION_STRING")
        if not project_connection_string:
            logger.warning("AZURE_AI_PROJECT_CONNECTION_STRING not found, using default configuration")
            project_client = None
        else:
            credential = DefaultAzureCredential()
            project_client = AIProjectClient.from_connection_string(
                conn_str=project_connection_string,
                credential=credential
            )
        
        # Initialize evaluator
        logger.info("Initializing RAG agent evaluator...")
        evaluator = RAGAgentEvaluator(
            rag_agent=rag_agent,
            project_client=project_client
        )
        
        # Define paths
        dataset_path = project_root / "agents" / "evaluations" / "data" / "single-turn-eval-ds.jsonl"
        output_path = project_root / "agents" / "evaluations" / "data" / "output" / "single-turn-eval-ds-agent-output.jsonl"
        
        logger.info(f"📁 Dataset path: {dataset_path}")
        logger.info(f"📁 Output path: {output_path}")
        
        # Load evaluation dataset
        logger.info("Loading evaluation dataset...")
        dataset = evaluator.load_evaluation_dataset(str(dataset_path))
        logger.info(f"✅ Loaded {len(dataset)} evaluation examples")
        
        # Generate agent responses
        logger.info("Generating agent responses...")
        dataset_with_responses = evaluator.generate_agent_responses(
            dataset=dataset,
            output_path=str(output_path)
        )
        logger.info(f"✅ Generated responses for {len(dataset_with_responses)} queries")
        
        # Run evaluations
        logger.info("Running comprehensive evaluation...")
        results = evaluator.evaluate_dataset(dataset_with_responses)
        
        # Display results
        logger.info("🎯 Evaluation Results:")
        logger.info("=" * 50)
        logger.info(f"Dataset Size: {results.dataset_size}")
        logger.info(f"Evaluation Time: {results.evaluation_time:.2f}s")
        logger.info("")
        logger.info("📊 Average Scores:")
        for metric, scores in results.evaluation_metrics.items():
            logger.info(f"  {metric.title()}: {scores['mean']:.3f} (±{scores['std']:.3f})")
        
        logger.info("=" * 50)
        logger.info("🎉 Evaluation completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Evaluation failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
