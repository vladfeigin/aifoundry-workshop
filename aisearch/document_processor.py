"""
Advanced integration example showing how to process documents and populate the search index.

This script demonstrates:
1. Reading documents from the docintel/data directory
2. Creating embeddings using Azure AI Foundry (Azure OpenAI) models
3. Populating the search index with real document content
4. Using API key authentication for Azure AI Foundry
"""

import os
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging
import asyncio
import re
import re
from create_search_index import SearchIndexCreator
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.indexes import SearchIndexClient
from openai import AzureOpenAI
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentProcessor:
    """
    Processes documents and populates the Azure AI Search index using Azure AI Foundry models.
    
    This class demonstrates how to integrate document processing with search indexing
    using Azure OpenAI embeddings for semantic search capabilities.
    """
    
    def __init__(self, search_service_name: str, index_name: str, 
                 azure_openai_endpoint: Optional[str] = None,
                 azure_openai_api_version: str = "2024-12-01-preview",
                 embedding_model: str = "text-embedding-3-small"):
        """
        Initialize the document processor with Azure AI Foundry integration.
        
        Args:
            search_service_name: Name of the Azure AI Search service
            index_name: Name of the search index
            azure_openai_endpoint: Azure OpenAI endpoint URL (optional, can be set via env var)
            azure_openai_api_version: API version for Azure OpenAI
            embedding_model: Name of the embedding model to use
        """
        self.search_service_name = search_service_name
        self.index_name = index_name
        self.embedding_model = embedding_model
        
        api_key = os.getenv("AZURE_SEARCH_API_KEY")
        if not api_key:
            raise ValueError("AZURE_SEARCH_API_KEY environment variable is required when not using managed identity")
        self.credential = AzureKeyCredential(api_key)
        logger.warning("Using API key authentication - consider using managed identity in production")
        
        # Initialize the search index client
        self.search_client = SearchClient(
            endpoint=f"https://{search_service_name}.search.windows.net",
            index_name=self.index_name,
            credential=self.credential
        )

        # Initialize Azure OpenAI client with API key authentication
        self.azure_openai_endpoint = azure_openai_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
        if not self.azure_openai_endpoint:
            raise ValueError("Azure OpenAI endpoint must be provided via parameter or AZURE_OPENAI_ENDPOINT environment variable")
        
        # Get API key from environment variable
        azure_openai_api_key = os.getenv("AZURE_OPENAI_API_KEY")
        if not azure_openai_api_key:
            raise ValueError("AZURE_OPENAI_API_KEY environment variable must be set")
        
        self.openai_client = AzureOpenAI(
            azure_endpoint=self.azure_openai_endpoint,
            api_key=azure_openai_api_key,
            api_version=azure_openai_api_version
        )
        
        logger.info(f"Initialized DocumentProcessor with embedding model: {embedding_model}")
    
    def generate_embedding(self, text: str, max_retries: int = 3) -> List[float]:
        """
        Generate embedding vector using Azure AI Foundry models.
        
        Implements retry logic with exponential backoff for reliability.
        
        Args:
            text: Text to generate embedding for
            max_retries: Maximum number of retry attempts
            
        Returns:
            List of floats representing the embedding vector
            
        Raises:
            Exception: If embedding generation fails after all retries
        """
        if not text.strip():
            logger.warning("Empty text provided for embedding generation")
            return []
        
        # Truncate text if too long (Azure OpenAI has token limits)
        if len(text) > 8000:  # Conservative limit for text-embedding models
            text = text[:8000]
            logger.warning("Text truncated to 8000 characters for embedding generation")
        
        for attempt in range(max_retries):
            try:
                response = self.openai_client.embeddings.create(
                    model=self.embedding_model,
                    input=text
                )
                
                embedding = response.data[0].embedding
                logger.debug(f"Generated embedding with {len(embedding)} dimensions")
                return embedding
                
            except Exception as e:
                wait_time = (2 ** attempt) + 1  # Exponential backoff
                logger.warning(f"Embedding generation attempt {attempt + 1} failed: {str(e)}")
                
                if attempt < max_retries - 1:
                    logger.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Failed to generate embedding after {max_retries} attempts")
                    raise
        
        return []
    
    def read_markdown_files(self, data_dir: str = "./docintel/data") -> List[Dict[str, Any]]:
        """
        Read markdown files from the data directory and prepare them for indexing.
        
        Args:
            data_dir: Path to the directory containing markdown files
            
        Returns:
            List of documents ready for indexing with hash-based document IDs
        """
        documents = []
        data_path = Path(data_dir)
        
        if not data_path.exists():
            logger.warning(f"Data directory {data_dir} does not exist")
            return documents
        
        # Find all markdown files
        md_files = list(data_path.glob("*.md"))
        logger.info(f"Found {len(md_files)} markdown files in {data_dir}")
        
        for md_file in md_files:
            try:
                with open(md_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                logger.info(f"Processing file: {md_file.name} ({len(content)} characters)")
                
                # Split content into pages using smart Markdown splitting
                pages = self._split_into_pages(content, max_length=2000)
                logger.info(f"Split {md_file.name} into {len(pages)} pages")
                
                for page_num, page_content in enumerate(pages, 1):
                    # Generate hash-based document ID from page content
                    page_hash = hashlib.md5(page_content.encode('utf-8')).hexdigest()[:8]
                    # Clean filename to only contain valid characters for Azure Search keys
                    clean_filename = re.sub(r'[^a-zA-Z0-9_\-=]', '_', md_file.stem)
                    doc_id = f"{clean_filename}_page_{page_num}_{page_hash}"
                    
                    # Generate embedding using Azure AI Foundry
                    try:
                        logger.debug(f"Generating embedding for {doc_id}")
                        embedding = self.generate_embedding(page_content)
                        
                        if not embedding:  # Skip if embedding generation failed
                            logger.warning(f"Skipping {doc_id} due to embedding generation failure")
                            continue
                            
                    except Exception as e:
                        logger.error(f"Failed to generate embedding for {doc_id}: {str(e)}")
                        continue  # Skip this document if embedding generation fails
                    
                    # Create document with proper field names matching the index schema
                    document = {
                        "docid": doc_id,
                        "page": page_content.strip(),
                        "page_vector": embedding  # This matches the index field name
                    }
                    
                    documents.append(document)
                    logger.debug(f"Prepared document: {doc_id} ({len(embedding)} dimensions)")
                
            except Exception as e:
                logger.error(f"Error processing file {md_file}: {str(e)}")
                continue
        
        logger.info(f"Successfully prepared {len(documents)} documents for indexing")
        return documents
    
    def _split_into_pages(self, content: str, max_length: int = 2000) -> List[str]:
        """
        Split content into page-sized chunks using the specific page delimiter pattern found in the data files.
        
        The files use this pattern for page delimiters:
        <!---- Page X ---------------------------------------------------------------------------------------------------------------------------------->
        
        Args:
            content: Text content to split
            max_length: Maximum length per page (used for fallback splitting)
            
        Returns:
            List of page contents
        """
        # Look for the specific page delimiter pattern used in the files
        page_delimiter_pattern = r'<!---- Page \d+ ---------------------------------------------------------------------------------------------------------------------------------->.*?\n'
        
        # Split content by page delimiters
        pages = re.split(page_delimiter_pattern, content, flags=re.MULTILINE)
        
        # Filter out empty pages and clean up whitespace
        cleaned_pages = []
        for page in pages:
            page_content = page.strip()
            if page_content:
                cleaned_pages.append(page_content)
        
        # If no page delimiters found, try fallback splitting methods
        if len(cleaned_pages) <= 1 and content.strip():
            logger.info("No specific page delimiters found, using fallback splitting methods")
            
            # Try other common patterns as fallback
            fallback_patterns = [
                r'\n---\n',  # Standard markdown horizontal rule
                r'<!-- pagebreak -->',  # HTML comment page break
                r'\\newpage',  # LaTeX style page break
                r'\n# ',  # Split on main headers (# Title)
            ]
            
            pages = [content]  # Start with the whole content
            
            for pattern in fallback_patterns:
                new_pages = []
                for page in pages:
                    if re.search(pattern, page):
                        # Split on this pattern
                        splits = re.split(pattern, page)
                        # Filter out empty splits and add the delimiter back where appropriate
                        for i, split in enumerate(splits):
                            if split.strip():
                                if pattern == r'\n# ' and i > 0:
                                    split = '# ' + split  # Add the header back
                                new_pages.append(split.strip())
                    else:
                        new_pages.append(page)
                pages = new_pages
                
            cleaned_pages = [page for page in pages if page.strip()]
            
            # If still couldn't split, create artificial pages for large content
            if len(cleaned_pages) == 1 and len(content) > max_length:
                logger.info(f"Creating artificial pages for large content ({len(content)} chars)")
                # Split large content into chunks at paragraph boundaries
                paragraphs = content.split('\n\n')
                cleaned_pages = []
                current_page = ""
                
                for paragraph in paragraphs:
                    paragraph = paragraph.strip()
                    if not paragraph:
                        continue
                        
                    if len(current_page) + len(paragraph) + 2 > max_length and current_page:
                        cleaned_pages.append(current_page)
                        current_page = paragraph
                    else:
                        current_page += '\n\n' + paragraph if current_page else paragraph
                        
                if current_page.strip():
                    cleaned_pages.append(current_page)
        else:
            logger.info(f"Found {len(cleaned_pages)} pages using specific page delimiters")
        
        return cleaned_pages
    
    def populate_index(self, documents: List[Dict[str, Any]], batch_size: int = 100) -> bool:
        """
        Populate the search index with documents.
        
        Args:
            documents: List of documents to upload
            batch_size: Number of documents to upload in each batch
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            logger.info(f"Uploading {len(documents)} documents to index {self.index_name}")
            
            # Upload in batches for better performance
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                result = self.search_client.upload_documents(documents=batch)
                
                # Check for errors
                for doc_result in result:
                    if not doc_result.succeeded:
                        logger.error(f"Failed to upload document {doc_result.key}: {doc_result.error_message}")
                
                logger.info(f"Uploaded batch {i//batch_size + 1}/{(len(documents) + batch_size - 1)//batch_size}")
            
            logger.info("Document upload completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error uploading documents: {str(e)}")
            return False
    
    def search_documents(self, query: str, top: int = 5, use_semantic_search: bool = True) -> List[Dict[str, Any]]:
        """
        Search documents in the index with optional semantic search.
        
        Args:
            query: Search query
            top: Number of results to return
            use_semantic_search: Whether to use vector-based semantic search
            
        Returns:
            List of search results
        """
        try:
            if use_semantic_search:
                # Generate embedding for the query
                query_embedding = self.generate_embedding(query)
                
                # Perform vector search
                results = self.search_client.search(
                    search_text=None,  # Pure vector search
                    vector_queries=[{
                        "kind": "vector",
                        "vector": query_embedding,
                        "k_nearest_neighbors": top,
                        "fields": "page_vector"
                    }],
                    top=top,
                    include_total_count=True
                )
            else:
                # Traditional text search
                results = self.search_client.search(
                    search_text=query,
                    top=top,
                    include_total_count=True
                )
            
            search_results = []
            for result in results:
                search_results.append({
                    "docid": result["docid"],
                    "page": result["page"][:200] + "..." if len(result["page"]) > 200 else result["page"],
                    "score": result.get("@search.score", 0.0)
                })
            
            search_type = "semantic" if use_semantic_search else "text"
            logger.info(f"Found {results.get_count()} total results using {search_type} search, returning top {len(search_results)}")
            return search_results
            
        except Exception as e:
            logger.error(f"Error searching documents: {str(e)}")
            return []
    
    def hybrid_search(self, query: str, top: int = 5, text_weight: float = 0.5) -> List[Dict[str, Any]]:
        """
        Perform hybrid search combining text and semantic search.
        
        Args:
            query: Search query
            top: Number of results to return
            text_weight: Weight for text search (0.0-1.0), semantic gets (1.0-text_weight)
            
        Returns:
            List of search results
        """
        try:
            # Generate embedding for the query
            query_embedding = self.generate_embedding(query)
            
            # Perform hybrid search
            results = self.search_client.search(
                search_text=query,
                vector_queries=[{
                    "kind": "vector",
                    "vector": query_embedding,
                    "k_nearest_neighbors": top * 2,  # Get more candidates for better ranking
                    "fields": "page_vector"
                }],
                top=top,
                include_total_count=True
            )
            
            search_results = []
            for result in results:
                search_results.append({
                    "docid": result["docid"],
                    "page": result["page"][:200] + "..." if len(result["page"]) > 200 else result["page"],
                    "score": result.get("@search.score", 0.0)
                })
            
            logger.info(f"Hybrid search found {results.get_count()} total results, returning top {len(search_results)}")
            return search_results
            
        except Exception as e:
            logger.error(f"Error in hybrid search: {str(e)}")
            return []


def main():
    """
    Main function to demonstrate document processing and search with Azure AI Foundry.
    """
    # Configuration - can be set via environment variables
    search_service_name = os.getenv("AZURE_SEARCH_SERVICE_NAME", "your-search-service")
    azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    index_name = "ai-foundry-workshop-index-v1"
    data_directory = "./docintel/data"  # Path to markdown files
    embedding_model = os.getenv("AZURE_OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    
    print(f"embedding model = {embedding_model}.")
    print(f"azure_openai_endpoint = {azure_openai_endpoint}.")

    # Validate required configuration
    if search_service_name == "your-search-service":
        print("❌ Error: Please set AZURE_SEARCH_SERVICE_NAME environment variable")
        print("   Example: export AZURE_SEARCH_SERVICE_NAME=your-actual-service-name")
        return
    
    if not azure_openai_endpoint:
        print("❌ Error: Please set AZURE_OPENAI_ENDPOINT environment variable")
        print("   Example: export AZURE_OPENAI_ENDPOINT=https://your-openai-service.openai.azure.com/")
        return
    
    # Check for Azure OpenAI API key
    azure_openai_api_key = os.getenv("AZURE_OPENAI_API_KEY")
    if not azure_openai_api_key:
        print("❌ Error: Please set AZURE_OPENAI_API_KEY environment variable")
        print("   Example: export AZURE_OPENAI_API_KEY=your-api-key")
        return
    
    try:
        # Step 1: Create the search index
        print("🔧 Step 1: Creating search index...")
        creator = SearchIndexCreator(search_service_name)
        
        if not creator.index_exists(index_name):
            success = creator.create_document_index(index_name)
            if not success:
                print("❌ Failed to create index")
                return
            print(f"✅ Index '{index_name}' created successfully!")
        else:
            print(f"ℹ️  Index '{index_name}' already exists")
        
        # Step 2: Initialize document processor with Azure AI Foundry
        print("\n🤖 Step 2: Initializing Azure AI Foundry integration...")
        processor = DocumentProcessor(
            search_service_name=search_service_name,
            index_name=index_name,
            azure_openai_endpoint=azure_openai_endpoint,
            embedding_model=embedding_model
        )
        print(f"✅ Connected to Azure OpenAI service with model: {embedding_model}")
        
        # Step 3: Process documents
        print(f"\n📄 Step 3: Processing documents from {data_directory}...")
        documents = processor.read_markdown_files(data_directory)
        
        if not documents:
            print("⚠️  No documents found to process")
            print(f"   Please ensure markdown files exist in: {data_directory}")
            return
        
        print(f"📊 Found {len(documents)} document pages to process")
        
        # Step 4: Populate the index
        print(f"\n⬆️  Step 4: Uploading documents with embeddings to search index...")
        success = processor.populate_index(documents)
        
        if not success:
            print("❌ Failed to populate index")
            return
        
        print("✅ Index populated successfully!")
        
        # Step 5: Test different search methods
        print("\n🔍 Step 5: Testing search functionality...")
        test_queries = [
            "Azure AI capabilities",
            "GPT-4 performance evaluation",
            "document intelligence features",
            "machine learning models"
        ]
        
        for query in test_queries:
            print(f"\n🔍 Testing query: '{query}'")
            
            # Test semantic search
            print("   📡 Semantic search results:")
            semantic_results = processor.search_documents(query, top=3, use_semantic_search=True)
            if semantic_results:
                for i, result in enumerate(semantic_results, 1):
                    print(f"      {i}. {result['docid']} (score: {result['score']:.3f})")
                    print(f"         {result['page'][:100]}...")
            else:
                print("      No results found")
            
            # Test hybrid search
            print("   🔀 Hybrid search results:")
            hybrid_results = processor.hybrid_search(query, top=3)
            if hybrid_results:
                for i, result in enumerate(hybrid_results, 1):
                    print(f"      {i}. {result['docid']} (score: {result['score']:.3f})")
                    print(f"         {result['page'][:100]}...")
            else:
                print("      No results found")
        
        print("\n🎉 Azure AI Foundry document processing and search demo completed!")
        print("\n💡 Next steps:")
        print("   • Experiment with different embedding models")
        print("   • Try semantic ranking for better results")
        print("   • Implement faceted search for categorization")
        print("   • Add metadata fields for better filtering")
        
    except Exception as e:
        logger.error(f"Demo failed: {str(e)}")
        print(f"\n❌ Error: {str(e)}")
        print("\n🔧 Setup checklist:")
        print("   1. Set AZURE_SEARCH_SERVICE_NAME environment variable")
        print("   2. Set AZURE_OPENAI_ENDPOINT environment variable")
        print("   3. Set AZURE_OPENAI_API_KEY environment variable")
        print("   4. Ensure managed identity has access to Azure AI Search service")
        print("   5. Verify markdown files exist in docintel/data directory")
        print("   6. Check that embedding model is deployed in Azure OpenAI")


if __name__ == "__main__":
    main()
