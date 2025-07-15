"""
RAG Agent Evaluation Module

This module provides comprehensive evaluation capabilities for the RAG agent using Azure AI Evaluation SDK.
It implements multiple evaluation metrics and automated testing workflows.

Features:
- Automated dataset processing and agent response generation
- Multiple evaluation metrics: Groundedness, Relevance, Response Completeness, Intent Resolution
- Batch evaluation with performance monitoring
- Detailed reporting and analytics
- Error handling and retry logic

Metrics:
- Groundedness: Measures if the response is supported by the retrieved context
- Relevance: Evaluates how relevant the response is to the user query
- Response Completeness: Assesses if the response fully addresses the query
- Intent Resolution: Determines if the response resolves the user's intent

Usage:
    #python rag_agent_eval.py --dataset-path ../data/single-turn-eval-ds.jsonl
    python -m agents.evaluations.rag.rag_agent_eval from project root directory
"""

import os
import json
import time
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

# Add parent directories to path for importing RAG agent
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

# Azure AI Evaluation SDK imports
from azure.ai.evaluation import (
    GroundednessEvaluator,
    RelevanceEvaluator,
    ResponseCompletenessEvaluator,
    IntentResolutionEvaluator,
    AzureOpenAIModelConfiguration
)
from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient
from dotenv import load_dotenv

# Import our RAG agent
from agents.rag.rag_agent import RAGAgent, RAGResponse

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class EvaluationDataPoint:
    """Single evaluation data point."""
    query: str
    context: str
    ground_truth: str
    agent_response: Optional[str] = None
    sources: Optional[List[Dict[str, Any]]] = None
    response_time: Optional[float] = None


@dataclass
class EvaluationResults:
    """Complete evaluation results."""
    dataset_size: int
    evaluation_metrics: Dict[str, Any]
    individual_scores: List[Dict[str, Any]]
    summary_statistics: Dict[str, Any]
    evaluation_time: float
    timestamp: str


class RAGAgentEvaluator:
    """
    Comprehensive RAG Agent Evaluator.
    
    This class provides end-to-end evaluation capabilities for RAG agents,
    including dataset processing, response generation, and multi-metric evaluation.
    """
    
    def __init__(
        self,
        rag_agent: RAGAgent,
        project_client: AIProjectClient,
        model_config: Optional[Dict[str, str]] = None
    ):
        """
        Initialize the evaluator.
        
        Args:
            rag_agent: Initialized RAG agent to evaluate
            project_client: Azure AI project client for evaluations
            model_config: Configuration for evaluation models
        """
        self.rag_agent = rag_agent
        self.project_client = project_client
        
        # Default model configuration for evaluations
        self.model_config = AzureOpenAIModelConfiguration(
            azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
            api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
            azure_deployment=model_config.get("model") if model_config else "gpt-4.1",
            api_version=model_config.get("api_version") if model_config else "2024-12-01-preview"
        )
        
        # Initialize evaluators
        self._init_evaluators()
        
    def _init_evaluators(self):
        """Initialize all evaluation metrics."""
        try:
            logger.info("Initializing Azure AI Evaluation metrics...")
            
            # Initialize Azure AI Evaluation SDK evaluators
            self.groundedness_evaluator = GroundednessEvaluator(model_config=self.model_config)
            
            self.relevance_evaluator = RelevanceEvaluator(model_config=self.model_config)
            
            # Initialize built-in evaluators for completeness and intent resolution
            self.completeness_evaluator = ResponseCompletenessEvaluator(model_config=self.model_config)
            
            self.intent_evaluator = IntentResolutionEvaluator(model_config=self.model_config)
            
            logger.info("✅ All evaluators initialized successfully")
            
        except Exception as e:
            logger.error("Failed to initialize evaluators: %s", str(e))
            raise
    
    def load_evaluation_dataset(self, dataset_path: str) -> List[EvaluationDataPoint]:
        """
        Load evaluation dataset from JSONL file.
        
        Args:
            dataset_path: Path to the JSONL evaluation dataset
            
        Returns:
            List of EvaluationDataPoint objects
        """
        logger.info("Loading evaluation dataset from: %s", dataset_path)
        
        dataset = []
        try:
            with open(dataset_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                        
                    try:
                        data = json.loads(line)
                        datapoint = EvaluationDataPoint(
                            query=data["query"],
                            context=data["context"],
                            ground_truth=data["ground_truth"]
                        )
                        dataset.append(datapoint)
                        
                    except (json.JSONDecodeError, KeyError) as e:
                        logger.warning("Skipping invalid line %d: %s", line_num, str(e))
                        continue
                        
            logger.info("✅ Loaded %d evaluation data points", len(dataset))
            return dataset
            
        except FileNotFoundError:
            logger.error("Dataset file not found: %s", dataset_path)
            raise
        except IOError as e:
            logger.error("Error loading dataset: %s", str(e))
            raise
    
    def single_turn_agent_run(
        self, 
        dataset: List[EvaluationDataPoint],
        output_path: Optional[str] = None,
        max_retries: int = 3
    ) -> List[EvaluationDataPoint]:
        """
        Run the RAG agent on all queries in the dataset and collect responses.
        
        Args:
            dataset: List of evaluation data points
            output_path: Optional path to save agent outputs
            max_retries: Maximum number of retries for failed queries
            
        Returns:
            Updated dataset with agent responses
        """
        logger.info("Running RAG agent on %d queries...", len(dataset))
        
        updated_dataset = []
        successful_runs = 0
        failed_runs = 0
        
        for i, datapoint in enumerate(dataset, 1):
            logger.info("Processing query %d/%d: %s...", i, len(dataset), datapoint.query[:50])
            
            # Attempt to get response with retries
            for attempt in range(max_retries):
                try:
                    start_time = time.time()
                    
                    # Get response from RAG agent
                    rag_response: RAGResponse = self.rag_agent.ask(datapoint.query)
                    
                    response_time = time.time() - start_time
                    
                    # Update datapoint with agent response
                    updated_datapoint = EvaluationDataPoint(
                        query=datapoint.query,
                        context=datapoint.context,
                        ground_truth=datapoint.ground_truth,
                        agent_response=rag_response.answer,
                        sources=[{
                            'docid': source['docid'],
                            'score': source['score'],
                            'content_preview': source.get('content_preview', '')
                        } for source in rag_response.sources],
                        response_time=response_time
                    )
                    
                    updated_dataset.append(updated_datapoint)
                    successful_runs += 1
                    
                    logger.info("✅ Query %d completed in %.2fs", i, response_time)
                    break
                    
                except (ConnectionError, TimeoutError, ValueError) as e:
                    logger.warning("Attempt %d failed for query %d: %s", attempt + 1, i, str(e))
                    if attempt == max_retries - 1:
                        # Final attempt failed, add datapoint without response
                        failed_datapoint = EvaluationDataPoint(
                            query=datapoint.query,
                            context=datapoint.context,
                            ground_truth=datapoint.ground_truth,
                            agent_response=f"ERROR: {str(e)}",
                            sources=[],
                            response_time=0.0
                        )
                        updated_dataset.append(failed_datapoint)
                        failed_runs += 1
                        logger.error("❌ Query %d failed after %d attempts", i, max_retries)
        
        logger.info("✅ Agent run completed: %d successful, %d failed", successful_runs, failed_runs)
        
        # Save outputs if path provided
        if output_path:
            self._save_agent_outputs(updated_dataset, output_path)
        
        return updated_dataset
    
    def _save_agent_outputs(self, dataset: List[EvaluationDataPoint], output_path: str):
        """Save dataset with original schema plus agent responses to JSONL file."""
        logger.info("Saving dataset with agent responses to: %s", output_path)
        
        try:
            # Create output directory if it doesn't exist
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                for datapoint in dataset:
                    # Preserve original evaluation dataset schema (query, context, ground_truth) 
                    # and add agent_response column
                    output_data = {
                        "query": datapoint.query,
                        "context": datapoint.context,
                        "ground_truth": datapoint.ground_truth,
                        "agent_response": datapoint.agent_response
                    }
                    f.write(json.dumps(output_data, ensure_ascii=False) + '\n')
                    
            logger.info("Dataset with agent responses saved successfully")
            
        except IOError as e:
            logger.error("Error saving dataset with agent responses: %s", str(e))
            raise
    
    def evaluate_dataset(
        self, 
        dataset: List[EvaluationDataPoint],
        batch_size: int = 5
    ) -> EvaluationResults:
        """
        Evaluate the dataset using all available metrics.
        
        Args:
            dataset: Dataset with agent responses
            batch_size: Number of evaluations to run concurrently
            
        Returns:
            Comprehensive evaluation results
        """
        logger.info("Starting evaluation on %d data points...", len(dataset))
        start_time = time.time()
        
        # Filter out failed responses
        valid_dataset = [dp for dp in dataset if dp.agent_response and not dp.agent_response.startswith("ERROR:")]
        logger.info("Evaluating %d valid responses out of %d total", len(valid_dataset), len(dataset))
        
        all_scores = []
        
        # Process evaluations
        for i, datapoint in enumerate(valid_dataset):
            logger.info("Evaluating datapoint %d/%d", i + 1, len(valid_dataset))
            
            try:
                result = self._evaluate_single_datapoint(datapoint)
                all_scores.append(result)
                
            except (ValueError, KeyError, TypeError) as e:
                logger.error("Evaluation failed for datapoint %d: %s", i, str(e))
                # Add placeholder scores for failed evaluations
                all_scores.append({
                    "query": datapoint.query,
                    "groundedness": 0.0,
                    "relevance": 0.0,
                    "completeness": 0.0,
                    "intent_resolution": 0.0,
                    "error": str(e)
                })
        
        evaluation_time = time.time() - start_time
        
        # Calculate summary statistics
        summary_stats = self._calculate_summary_statistics(all_scores)
        
        results = EvaluationResults(
            dataset_size=len(dataset),
            evaluation_metrics={
                "groundedness": summary_stats["groundedness"],
                "relevance": summary_stats["relevance"],
                "completeness": summary_stats["completeness"],
                "intent_resolution": summary_stats["intent_resolution"]
            },
            individual_scores=all_scores,
            summary_statistics=summary_stats,
            evaluation_time=evaluation_time,
            timestamp=datetime.now().isoformat()
        )
        
        logger.info("✅ Evaluation completed in %.2fs", evaluation_time)
        return results
    
    def _evaluate_single_datapoint(self, datapoint: EvaluationDataPoint) -> Dict[str, Any]:
        """Evaluate a single datapoint with all metrics."""
        try:
            # Run evaluations
            groundedness_result = self.groundedness_evaluator(
                query=datapoint.query,
                response=datapoint.agent_response,
                context=datapoint.context
            )
            
            relevance_result = self.relevance_evaluator(
                query=datapoint.query,
                response=datapoint.agent_response,
                context=datapoint.context
            )
            
            # Log parameters for completeness evaluator
            logger.info("Completeness evaluator parameters - query: %s, response: %s, ground_truth: %s", 
                       datapoint.query[:50], datapoint.agent_response[:100], datapoint.ground_truth[:100])
            
            completeness_result = self.completeness_evaluator(
                query=datapoint.query,
                response=datapoint.agent_response,
                ground_truth=datapoint.ground_truth
            )
            
            logger.info("Completeness evaluator result: %s", completeness_result)
            
            intent_result = self.intent_evaluator(
                query=datapoint.query,
                response=datapoint.agent_response,
                context=datapoint.context
            )
            
            return {
                "query": datapoint.query,
                "groundedness": groundedness_result.get("groundedness", 0.0),
                "relevance": relevance_result.get("relevance", 0.0),
                "completeness": completeness_result.get("completeness", 0.0),
                "intent_resolution": intent_result.get("intent_resolution", 0.0),
                "response_time": datapoint.response_time
            }
            
        except (ValueError, KeyError, TypeError) as e:
            logger.error("Error evaluating datapoint: %s", str(e))
            raise
    
    def _calculate_summary_statistics(self, scores: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate summary statistics for all metrics."""
        metrics = ["groundedness", "relevance", "completeness", "intent_resolution"]
        stats = {}
        
        for metric in metrics:
            valid_scores = [score[metric] for score in scores if isinstance(score.get(metric), (int, float))]
            
            if valid_scores:
                stats[metric] = {
                    "mean": sum(valid_scores) / len(valid_scores),
                    "min": min(valid_scores),
                    "max": max(valid_scores),
                    "count": len(valid_scores)
                }
            else:
                stats[metric] = {
                    "mean": 0.0,
                    "min": 0.0,
                    "max": 0.0,
                    "count": 0
                }
        
        # Calculate overall performance
        all_scores = []
        for score in scores:
            valid_metrics = [score[metric] for metric in metrics if isinstance(score.get(metric), (int, float))]
            if valid_metrics:
                all_scores.append(sum(valid_metrics) / len(valid_metrics))
        
        stats["overall"] = {
            "mean": sum(all_scores) / len(all_scores) if all_scores else 0.0,
            "min": min(all_scores) if all_scores else 0.0,
            "max": max(all_scores) if all_scores else 0.0,
            "count": len(all_scores)
        }
        
        return stats
    
    def save_evaluation_results(self, results: EvaluationResults, output_path: str):
        """Save evaluation results to JSON file."""
        logger.info("Saving evaluation results to: %s", output_path)
        
        try:
            # Create output directory if it doesn't exist
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Convert dataclass to dict
            results_dict = asdict(results)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results_dict, f, indent=2, ensure_ascii=False)
                
            logger.info("Evaluation results saved successfully")
            
        except IOError as e:
            logger.error("Error saving evaluation results: %s", str(e))
            raise
    
    def print_evaluation_summary(self, results: EvaluationResults):
        """Print a formatted evaluation summary."""
        print("\n" + "="*80)
        print("🏆 RAG AGENT EVALUATION RESULTS")
        print("="*80)
        
        print("\n📊 Dataset Information:")
        print(f"   • Total data points: {results.dataset_size}")
        print(f"   • Evaluation time: {results.evaluation_time:.2f}s")
        print(f"   • Timestamp: {results.timestamp}")
        
        print("\n📈 Performance Metrics:")
        for metric, stats in results.evaluation_metrics.items():
            print(f"   • {metric.replace('_', ' ').title()}: {stats['mean']:.3f} "
                  f"(min: {stats['min']:.3f}, max: {stats['max']:.3f}, count: {stats['count']})")
        
        if "overall" in results.summary_statistics:
            overall = results.summary_statistics["overall"]
            print(f"   • Overall Score: {overall['mean']:.3f}")
        
        print("\n🎯 Top Performing Queries:")
        # Sort by overall performance
        sorted_scores = sorted(results.individual_scores, 
                             key=lambda x: sum([x.get(m, 0) for m in ['groundedness', 'relevance', 'completeness', 'intent_resolution']]) / 4,
                             reverse=True)
        
        for i, score in enumerate(sorted_scores[:3], 1):
            overall_score = sum([score.get(m, 0) for m in ['groundedness', 'relevance', 'completeness', 'intent_resolution']]) / 4
            print(f"   {i}. {score['query'][:60]}... (Score: {overall_score:.3f})")
        
        print("\n⚠️  Areas for Improvement:")
        worst_scores = sorted(results.individual_scores, 
                            key=lambda x: sum([x.get(m, 0) for m in ['groundedness', 'relevance', 'completeness', 'intent_resolution']]) / 4)
        
        for i, score in enumerate(worst_scores[:2], 1):
            overall_score = sum([score.get(m, 0) for m in ['groundedness', 'relevance', 'completeness', 'intent_resolution']]) / 4
            print(f"   {i}. {score['query'][:60]}... (Score: {overall_score:.3f})")
        
        print("\n" + "="*80)


def main():
    """Main evaluation workflow."""
    try:
        logger.info("🚀 Starting RAG Agent Evaluation")
        
        # Initialize Azure AI Project Client
        logger.info("Initializing Azure AI Project Client...")
        project_client = AIProjectClient(
            credential=DefaultAzureCredential(),
            endpoint=os.environ["PROJECT_ENDPOINT"],
        )
        
        # Initialize RAG Agent
        logger.info("Initializing RAG Agent...")
        rag_agent = RAGAgent(
            search_service_name=os.environ["AZURE_SEARCH_SERVICE_NAME"],
            search_index_name=os.environ["AZURE_SEARCH_INDEX_NAME"],
            azure_openai_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
            chat_model=os.environ.get("AZURE_OPENAI_CHAT_MODEL", "gpt-4.1"),
            top_k_documents=3
        )
        
        # Initialize evaluator
        evaluator = RAGAgentEvaluator(
            rag_agent=rag_agent,
            project_client=project_client,
            model_config={"model": "gpt-4.1", "api_version": "2024-12-01-preview"}
        )
        
        # Define paths
        base_path = Path(__file__).parent.parent
        dataset_path = base_path / "data" / "single-turn-eval-ds.jsonl"
        output_path = base_path / "data" / "output" / "single-turn-eval-ds-agent-output.jsonl"
        results_path = base_path / "data" / "output" / "evaluation_results.json"
        
        # Load evaluation dataset
        dataset = evaluator.load_evaluation_dataset(str(dataset_path))
        
        # Run RAG agent on all queries and add responses to the evaluation dataset
        logger.info("🤖 Running RAG agent on evaluation dataset...")
        dataset_with_responses = evaluator.single_turn_agent_run(
            dataset=dataset,
            output_path=str(output_path)
        )
        
        # Evaluate all responses
        logger.info("📊 Evaluating agent responses...")
        evaluation_results = evaluator.evaluate_dataset(dataset_with_responses)
        
        # Save results
        evaluator.save_evaluation_results(evaluation_results, str(results_path))
        
        # Print summary
        evaluator.print_evaluation_summary(evaluation_results)
        
        logger.info("🎉 RAG Agent evaluation completed successfully!")
        
    except KeyError as e:
        logger.error("Missing environment variable: %s", str(e))
        raise
    except Exception as e:
        logger.error("Evaluation failed: %s", str(e))
        raise


if __name__ == "__main__":
    main()
