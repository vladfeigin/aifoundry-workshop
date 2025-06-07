#!/bin/bash

# Install dependencies for Azure AI Search functionality

echo "Installing Azure AI Search dependencies..."

# Install the package in development mode
pip install -e .

echo "Dependencies installed successfully!"
echo ""
echo "Next steps:"
echo "1. Configure your Azure AI Search service name"
echo "2. Set up authentication (managed identity recommended)"
echo "3. Run: python aisearch/create_search_index.py --search-service YOUR_SERVICE --index-name documents-index"
echo ""
echo "For more information, see aisearch/README.md"
