#!/usr/bin/env python3
"""
Basic RAG Query Example

This example demonstrates how to perform a simple RAG query using the Azure AI Foundry Workshop.

Prerequisites:
- Azure AI services configured
- Search index created and populated
- Environment variables set
"""

import os
from dotenv import load_dotenv
from agents.rag.rag_agent import RAGAgentService

def main():
    """Run a basic RAG query example"""
    
    # Load environment variables
    load_dotenv()
    
    # Initialize RAG agent
    print("🤖 Initializing RAG Agent...")
    agent = RAGAgentService(
        project_endpoint=os.getenv("PROJECT_ENDPOINT"),
        search_index_name=os.getenv("AZURE_SEARCH_INDEX_NAME", "ai-foundry-workshop-index-v1"),
        chat_model=os.getenv("AZURE_OPENAI_CHAT_MODEL", "gpt-4o")
    )
    
    # Example queries
    queries = [
        "What is GPT-4?",
        "How does document intelligence work?", 
        "What are the key features of Azure AI services?"
    ]
    
    print("🔍 Running example queries...\n")
    
    for i, query in enumerate(queries, 1):
        print(f"Query {i}: {query}")
        print("-" * 50)
        
        try:
            # Get response from RAG agent
            response = agent.ask(query)
            
            # Display results
            print(f"Answer: {response.answer}")
            print(f"Sources: {len(response.sources)} documents retrieved")
            print(f"Response time: {response.total_time:.2f} seconds")
            print(f"Thread ID: {response.thread_id}")
            
            # Show source details
            if response.sources:
                print("\nSource documents:")
                for j, source in enumerate(response.sources[:2], 1):  # Show first 2 sources
                    print(f"  {j}. {source}")
            
        except Exception as e:
            print(f"❌ Error processing query: {e}")
        
        print("\n" + "="*60 + "\n")
    
    print("✅ Example completed!")

if __name__ == "__main__":
    main()