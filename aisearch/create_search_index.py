"""
Azure AI Search Index Creation Script

This script creates an Azure AI Search index with the following fields:
1. docid - Document identifier
2. page - Page content text
3. page_vector - Embedding vector of the page content

Usage:
From the project root directory:
    python ./aisearch/create_search_index.py --search-service <service_name> --index-name <index_name>
"""

import argparse
import os
import logging
from typing import List, Dict, Any
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SearchField,
    SearchFieldDataType,
    SimpleField,
    SearchableField,
    VectorSearch,
    VectorSearchProfile,
    VectorSearchAlgorithmConfiguration,
    HnswAlgorithmConfiguration,
    VectorSearchAlgorithmKind
)
from azure.core.credentials import AzureKeyCredential
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SearchIndexCreator:
    """
    Creates and manages Azure AI Search indexes with vector search capabilities.
    
    This class follows Azure best practices:
    - Uses managed identity for authentication when possible
    - Implements proper error handling and logging
    - Follows secure configuration patterns
    """
    
    def __init__(self, search_service_name: str, use_managed_identity: bool = False):
        """
        Initialize the Search Index Creator.
        
        Args:
            search_service_name: Name of the Azure AI Search service
            use_managed_identity: Whether to use managed identity (recommended for Azure-hosted apps)
        """
        self.search_service_name = search_service_name
        self.search_endpoint = f"https://{search_service_name}.search.windows.net"
        
        # Authentication - prefer managed identity for security
        if use_managed_identity:
            self.credential = DefaultAzureCredential()
            logger.info("Using managed identity for authentication")
        else:
            # Fallback to API key - only for development/testing
            api_key = os.getenv("AZURE_SEARCH_API_KEY")
            if not api_key:
                raise ValueError("AZURE_SEARCH_API_KEY environment variable is required when not using managed identity")
            self.credential = AzureKeyCredential(api_key)
            logger.warning("Using API key authentication - consider using managed identity in production")
        
        # Initialize the search index client
        self.index_client = SearchIndexClient(
            endpoint=self.search_endpoint,
            credential=self.credential
        )
    
    def create_document_index(self, index_name: str, vector_dimensions: int = 1536) -> bool:
        """
        Create an Azure AI Search index optimized for document search with vector embeddings.
        
        Args:
            index_name: Name of the index to create
            vector_dimensions: Dimensions of the embedding vectors (default: 1536 for OpenAI embeddings)
            
        Returns:
            bool: True if index was created successfully, False otherwise
        """
        try:
            logger.info(f"Creating search index: {index_name}")
            
            # Define the index fields
            fields = [
                # Document ID - unique identifier for each document
                SimpleField(
                    name="docid",
                    type=SearchFieldDataType.String,
                    key=True,  # This is the primary key
                    filterable=True,
                    sortable=True
                ),
                
                # Page content - searchable text field
                SearchableField(
                    name="page",
                    type=SearchFieldDataType.String,
                    searchable=True,  # Enable full-text search
                    filterable=False,
                    sortable=False,
                    analyzer_name="standard.lucene"  # Use standard analyzer for text processing
                ),
                
                # Page vector - embedding vector for semantic search
                SearchField(
                    name="page_vector",
                    type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                    searchable=True,
                    vector_search_dimensions=vector_dimensions,
                    vector_search_profile_name="default-vector-profile"
                )
            ]
            
            # Configure vector search with HNSW algorithm for optimal performance
            vector_search = VectorSearch(
                profiles=[
                    VectorSearchProfile(
                        name="default-vector-profile",
                        algorithm_configuration_name="default-hnsw-config"
                    )
                ],
                algorithms=[
                    HnswAlgorithmConfiguration(
                        name="default-hnsw-config",
                        kind=VectorSearchAlgorithmKind.HNSW,
                        parameters={
                            "m": 4,  # Number of bi-directional links for new elements
                            "efConstruction": 400,  # Size of dynamic candidate list
                            "efSearch": 500,  # Size of dynamic candidate list for search
                            "metric": "cosine"  # Distance metric for vector similarity
                        }
                    )
                ]
            )
            
            # Create the search index
            index = SearchIndex(
                name=index_name,
                fields=fields,
                vector_search=vector_search
            )
            
            # Create the index with retry logic
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    result = self.index_client.create_index(index)
                    logger.info(f"Successfully created index: {result.name}")
                    return True
                    
                except Exception as e:
                    if attempt < max_retries - 1:
                        logger.warning(f"Attempt {attempt + 1} failed, retrying: {str(e)}")
                        continue
                    else:
                        raise e
                        
        except Exception as e:
            logger.error(f"Failed to create index {index_name}: {str(e)}")
            return False
    
    def index_exists(self, index_name: str) -> bool:
        """
        Check if an index already exists.
        
        Args:
            index_name: Name of the index to check
            
        Returns:
            bool: True if index exists, False otherwise
        """
        try:
            self.index_client.get_index(index_name)
            return True
        except Exception:
            return False
    
    def delete_index_if_exists(self, index_name: str) -> bool:
        """
        Delete an index if it exists.
        
        Args:
            index_name: Name of the index to delete
            
        Returns:
            bool: True if index was deleted or didn't exist, False on error
        """
        try:
            if self.index_exists(index_name):
                self.index_client.delete_index(index_name)
                logger.info(f"Deleted existing index: {index_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete index {index_name}: {str(e)}")
            return False


def main():
    """
    Main function to handle command line arguments and create the search index.
    """
    parser = argparse.ArgumentParser(description="Create Azure AI Search index for document search")
    parser.add_argument(
        "--search-service",
        required=True,
        help="Name of the Azure AI Search service"
    )
    parser.add_argument(
        "--index-name",
        required=True,
        help="Name of the index to create"
    )
    parser.add_argument(
        "--vector-dimensions",
        type=int,
        default=1536,
        help="Dimensions of the embedding vectors (default: 1536)"
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Delete existing index if it exists before creating new one"
    )
    parser.add_argument(
        "--use-api-key",
        action="store_true",
        help="Use API key authentication instead of managed identity"
    )
    
    args = parser.parse_args()
    
    try:
        # Create the search index creator
        creator = SearchIndexCreator(
            search_service_name=args.search_service,
            use_managed_identity=not args.use_api_key
        )
        
        # Check if index exists and handle recreation
        if creator.index_exists(args.index_name):
            if args.recreate:
                logger.info(f"Index {args.index_name} exists, deleting it...")
                if not creator.delete_index_if_exists(args.index_name):
                    logger.error("Failed to delete existing index")
                    return 1
            else:
                logger.warning(f"Index {args.index_name} already exists. Use --recreate to replace it.")
                return 1
        
        # Create the index
        success = creator.create_document_index(
            index_name=args.index_name,
            vector_dimensions=args.vector_dimensions
        )
        
        if success:
            logger.info("Index creation completed successfully!")
            return 0
        else:
            logger.error("Index creation failed!")
            return 1
            
    except Exception as e:
        logger.error(f"Script execution failed: {str(e)}")
        return 1


if __name__ == "__main__":
    exit(main())
