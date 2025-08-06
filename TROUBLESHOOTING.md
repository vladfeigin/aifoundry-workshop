# Troubleshooting Guide

This guide provides solutions to common issues you might encounter while working with the Azure AI Foundry Workshop.

## Table of Contents

- [Authentication Issues](#authentication-issues)
- [Azure Service Connection Problems](#azure-service-connection-problems)
- [Environment Configuration Issues](#environment-configuration-issues)
- [Python Dependencies and Installation](#python-dependencies-and-installation)
- [Document Processing Issues](#document-processing-issues)
- [Search Index Problems](#search-index-problems)
- [RAG Agent Issues](#rag-agent-issues)
- [Evaluation Problems](#evaluation-problems)
- [Performance Issues](#performance-issues)
- [Development Environment Issues](#development-environment-issues)

## Authentication Issues

### Problem: "DefaultAzureCredential failed to retrieve a token"

**Symptoms:**
```
azure.core.exceptions.ClientAuthenticationError: DefaultAzureCredential failed to retrieve a token from the included credentials.
```

**Solutions:**

1. **Azure CLI Authentication** (Recommended for development):
   ```bash
   az login
   az account show  # Verify you're logged in
   az account set --subscription "your-subscription-id"
   ```

2. **Environment Variables** (Alternative):
   ```bash
   export AZURE_CLIENT_ID="your-client-id"
   export AZURE_CLIENT_SECRET="your-client-secret"
   export AZURE_TENANT_ID="your-tenant-id"
   ```

3. **Managed Identity** (For Azure-hosted environments):
   - Ensure your Azure resource has a managed identity assigned
   - Grant appropriate permissions to the managed identity

### Problem: "Insufficient permissions" or "Access denied"

**Symptoms:**
```
azure.core.exceptions.HttpResponseError: (Forbidden) Access denied
```

**Solutions:**

1. **Check Role Assignments**:
   - Ensure you have "Azure AI User" role or equivalent
   - For specific services, check:
     - **AI Search**: "Search Service Contributor" or "Search Index Data Contributor"
     - **Document Intelligence**: "Cognitive Services User"
     - **OpenAI**: "Cognitive Services OpenAI User"

2. **Grant Permissions via Azure Portal**:
   ```bash
   # Using Azure CLI
   az role assignment create \
     --assignee "your-user-email@domain.com" \
     --role "Azure AI User" \
     --scope "/subscriptions/your-subscription-id"
   ```

3. **Service Principal Permissions**:
   ```bash
   # Create service principal with required permissions
   az ad sp create-for-rbac \
     --name "aifoundry-workshop-sp" \
     --role "Azure AI User" \
     --scopes "/subscriptions/your-subscription-id"
   ```

## Azure Service Connection Problems

### Problem: Azure AI Search connection failures

**Symptoms:**
```
requests.exceptions.ConnectionError: HTTPSConnectionPool(host='your-search-service.search.windows.net', port=443)
```

**Solutions:**

1. **Verify Service Endpoint**:
   ```bash
   # Test connectivity
   curl -I "https://your-search-service.search.windows.net"
   ```

2. **Check Environment Variables**:
   ```bash
   echo $AZURE_SEARCH_SERVICE_NAME
   echo $AZURE_SEARCH_INDEX_NAME
   # Ensure they match your Azure portal configuration
   ```

3. **Network Connectivity**:
   - Check if your network allows outbound HTTPS (port 443)
   - Verify firewall settings
   - For corporate networks, check proxy configurations

4. **Service Availability**:
   - Check Azure Service Health in the portal
   - Verify the service is running and not under maintenance

### Problem: Azure OpenAI service unavailable

**Symptoms:**
```
openai.BadRequestError: Error code: 404 - {'error': {'code': 'DeploymentNotFound'}}
```

**Solutions:**

1. **Verify Model Deployment**:
   - Check Azure OpenAI Studio that your models are deployed
   - Ensure model names match your configuration
   - Verify deployment status is "Succeeded"

2. **Check Model Names**:
   ```bash
   # In .env file, verify model names match deployments
   AZURE_OPENAI_CHAT_MODEL="gpt-4o"  # Must match deployment name
   AZURE_OPENAI_EMBEDDING_MODEL="text-embedding-3-small"
   ```

3. **Regional Availability**:
   - Ensure the model is available in your Azure region
   - Some models have limited regional availability

### Problem: Document Intelligence service errors

**Symptoms:**
```
azure.ai.documentintelligence.DocumentIntelligenceServiceError: Operation returned an invalid status 'Failed'
```

**Solutions:**

1. **Check Document Format**:
   - Ensure PDF is not corrupted
   - Verify file size is within limits (50MB for synchronous, 500MB for asynchronous)
   - Check if PDF is password-protected or encrypted

2. **Service Limits**:
   - Verify you haven't exceeded rate limits
   - Check your service tier quotas

3. **Document Content**:
   - Ensure document contains extractable text (not just images)
   - For image-heavy documents, ensure images are clear and readable

## Environment Configuration Issues

### Problem: Environment variables not loading

**Symptoms:**
```python
KeyError: 'AZURE_SEARCH_SERVICE_NAME'
```

**Solutions:**

1. **Check .env File Location**:
   ```bash
   # .env file should be in project root
   ls -la .env
   pwd  # Should be in aifoundry-workshop directory
   ```

2. **Verify .env File Content**:
   ```bash
   cat .env | grep -v "^#" | grep -v "^$"  # Show non-comment lines
   ```

3. **Load Environment Variables Manually**:
   ```python
   from dotenv import load_dotenv
   import os
   
   load_dotenv()  # Load from .env file
   print(os.environ.get('AZURE_SEARCH_SERVICE_NAME'))  # Should not be None
   ```

4. **Environment Variable Syntax**:
   ```bash
   # Correct format (no spaces around =)
   AZURE_SEARCH_SERVICE_NAME=your-service-name
   
   # Incorrect format
   AZURE_SEARCH_SERVICE_NAME = your-service-name  # ❌ Spaces around =
   ```

### Problem: Invalid environment variable values

**Symptoms:**
- Services return 401/403 errors
- Connection failures with correct credentials

**Solutions:**

1. **Remove Quotes from Values**:
   ```bash
   # Correct
   AZURE_SEARCH_API_KEY=your-actual-key-value
   
   # Incorrect
   AZURE_SEARCH_API_KEY="your-actual-key-value"  # ❌ Quotes included
   ```

2. **Check for Trailing Spaces**:
   ```bash
   # Remove trailing whitespace
   sed -i 's/[[:space:]]*$//' .env
   ```

3. **Validate Endpoints**:
   ```bash
   # Azure endpoints should be full URLs
   AZURE_OPENAI_ENDPOINT=https://your-service.openai.azure.com/  # ✅
   PROJECT_ENDPOINT=https://your-project.services.ai.azure.com/api/projects/your-project  # ✅
   ```

## Python Dependencies and Installation

### Problem: Package installation failures

**Symptoms:**
```
ERROR: Could not find a version that satisfies the requirement azure-ai-projects
```

**Solutions:**

1. **Update pip and setuptools**:
   ```bash
   pip install --upgrade pip setuptools wheel
   ```

2. **Install Dependencies Individually**:
   ```bash
   pip install azure-ai-documentintelligence
   pip install azure-search-documents
   pip install azure-identity
   pip install python-dotenv
   pip install rich
   pip install openai
   pip install azure-ai-projects
   pip install azure-monitor-opentelemetry
   pip install azure-ai-evaluation
   ```

3. **Python Version Compatibility**:
   ```bash
   python --version  # Should be 3.12 or higher
   ```

4. **Use Virtual Environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

### Problem: Import errors after installation

**Symptoms:**
```python
ModuleNotFoundError: No module named 'azure.ai.projects'
```

**Solutions:**

1. **Verify Installation**:
   ```bash
   pip list | grep azure-ai-projects
   ```

2. **Check Python Path**:
   ```python
   import sys
   print(sys.path)
   ```

3. **Reinstall Package**:
   ```bash
   pip uninstall azure-ai-projects
   pip install azure-ai-projects
   ```

4. **Check Virtual Environment**:
   ```bash
   which python  # Should point to your virtual environment
   ```

### Problem: "Multiple top-level packages discovered" error

**Symptoms:**
```
error: Multiple top-level packages discovered in a flat-layout: ['agents', 'aisearch', 'docintel'].
```

**Solutions:**

1. **Install Dependencies Only** (Don't install the project as editable):
   ```bash
   # Instead of: pip install -e .
   # Install dependencies individually as shown above
   ```

2. **Run Modules Directly**:
   ```bash
   # Instead of importing, run as modules
   python -m agents.rag.rag_agent
   python -m aisearch.create_search_index
   ```

## Document Processing Issues

### Problem: PDF conversion fails

**Symptoms:**
```
azure.ai.documentintelligence.DocumentIntelligenceServiceError: (InvalidContent) The input content is not valid.
```

**Solutions:**

1. **Check PDF File**:
   ```bash
   # Verify file exists and is readable
   ls -la your-document.pdf
   file your-document.pdf  # Should show "PDF document"
   ```

2. **File Size Limits**:
   - Maximum 50MB for synchronous operations
   - Maximum 500MB for asynchronous operations
   - Use smaller files or split large documents

3. **PDF Content Issues**:
   - Ensure PDF contains extractable text
   - Avoid password-protected or encrypted PDFs
   - Test with a simple, known-good PDF first

4. **URL Issues** (when using HTTP URLs):
   ```bash
   # Test URL accessibility
   curl -I "https://example.com/document.pdf"
   
   # Should return HTTP 200 and Content-Type: application/pdf
   ```

### Problem: Empty or incomplete Markdown output

**Symptoms:**
- PDF processes successfully but produces minimal or no content
- Missing tables or formatting

**Solutions:**

1. **Check PDF Content**:
   - Verify the PDF contains text (not just images)
   - Test with a different PDF file

2. **Increase Processing Time**:
   - Large documents may need more processing time
   - Check if the operation completed successfully

3. **Review Conversion Settings**:
   ```python
   # In pdf-2-md.py, check model configuration
   # Ensure using 'prebuilt-layout' model
   ```

## Search Index Problems

### Problem: Index creation fails

**Symptoms:**
```
azure.search.documents.RequestEntityTooLargeError: Operation returned an invalid status 'Request Entity Too Large'
```

**Solutions:**

1. **Check Index Schema Size**:
   - Reduce vector dimensions if possible
   - Simplify field definitions

2. **Service Tier Limitations**:
   - Verify your search service tier supports the index size
   - Consider upgrading service tier if needed

3. **Retry with Smaller Batches**:
   ```python
   # In create_search_index.py, reduce batch size
   batch_size = 100  # Reduce from default
   ```

### Problem: Document ingestion fails

**Symptoms:**
```
azure.search.documents.RequestEntityTooLargeError: The request payload is too large.
```

**Solutions:**

1. **Reduce Document Size**:
   ```python
   # In ingest_documents.py, increase chunking
   MAX_CONTENT_LENGTH = 8000  # Reduce from default
   ```

2. **Batch Size Optimization**:
   ```python
   # Process smaller batches
   batch_size = 10  # Reduce batch size
   ```

3. **Content Truncation**:
   ```python
   # Truncate long content before indexing
   content = content[:8000] if len(content) > 8000 else content
   ```

### Problem: Search returns no results

**Symptoms:**
- Search queries return empty results
- Index appears to be populated

**Solutions:**

1. **Verify Index Content**:
   ```bash
   # Check document count in Azure portal
   # AI Search > Indexes > your-index > Documents tab
   ```

2. **Test Simple Queries**:
   ```python
   # Try basic keyword search first
   results = search_client.search(search_text="test")
   ```

3. **Check Field Configuration**:
   - Ensure fields are marked as "searchable"
   - Verify analyzer settings

4. **Vector Search Issues**:
   - Ensure embedding model matches the one used for indexing
   - Verify vector dimensions match index schema

## RAG Agent Issues

### Problem: RAG agent returns generic responses

**Symptoms:**
- Responses don't reference retrieved documents
- Answers seem disconnected from search results

**Solutions:**

1. **Check Search Results**:
   ```python
   # Add debug logging to see retrieved documents
   logger.info(f"Retrieved {len(search_results)} documents")
   for doc in search_results:
       logger.info(f"Doc: {doc['docid']}, Score: {doc['@search.score']}")
   ```

2. **Verify Context Formation**:
   ```python
   # Check if context is properly formatted
   logger.info(f"Context length: {len(context)}")
   logger.info(f"Context preview: {context[:500]}...")
   ```

3. **Prompt Engineering**:
   - Review system prompts in the RAG agent
   - Ensure prompts instruct the model to use provided context
   - Add explicit instructions to cite sources

### Problem: Tracing not working

**Symptoms:**
- No traces appearing in Azure AI Foundry portal
- Missing observability data

**Solutions:**

1. **Check Tracing Configuration**:
   ```bash
   # Verify environment variables
   echo $PROJECT_ENDPOINT
   echo $AZURE_TRACING_GEN_AI_CONTENT_RECORDING_ENABLED
   ```

2. **Azure Monitor Setup**:
   ```python
   # Ensure tracing is properly initialized
   from azure.monitor.opentelemetry import configure_azure_monitor
   configure_azure_monitor(connection_string=connection_string)
   ```

3. **Check Portal Access**:
   - Navigate to Azure AI Foundry > Your Project > Tracing
   - Verify you have appropriate permissions to view traces

## Evaluation Problems

### Problem: Evaluation metrics return low scores

**Symptoms:**
- All evaluation scores are below 0.5
- Groundedness scores consistently low

**Solutions:**

1. **Check Evaluation Data Quality**:
   ```python
   # Verify evaluation dataset format
   with open('eval-dataset.jsonl', 'r') as f:
       for line in f:
           data = json.loads(line)
           print(f"Query: {data['query'][:100]}...")
           print(f"Ground truth: {data['ground_truth'][:100]}...")
   ```

2. **Review Ground Truth Quality**:
   - Ensure ground truth answers are comprehensive
   - Verify they match the expected response style

3. **Check Search Quality**:
   - Test if search returns relevant documents
   - Verify retrieved context contains information needed to answer

4. **Model Configuration**:
   ```python
   # Check evaluation model settings
   model_config = {
       "model": "gpt-4o",  # Use a capable model for evaluation
       "api_version": "2024-12-01-preview"
   }
   ```

### Problem: Cloud evaluation fails

**Symptoms:**
```
azure.ai.evaluation.exceptions.EvaluationException: Evaluation job failed
```

**Solutions:**

1. **Check Dataset Format**:
   - Ensure JSONL format is correct
   - Verify required fields are present

2. **Data Upload Issues**:
   ```python
   # Check if dataset uploaded successfully
   print(f"Dataset ID: {data_id}")
   # Verify dataset appears in Azure AI Foundry portal
   ```

3. **Evaluation Configuration**:
   ```python
   # Verify evaluator configuration
   evaluators = {
       "groundedness": EvaluatorIds.GROUNDEDNESS,
       "relevance": EvaluatorIds.RELEVANCE
   }
   ```

## Performance Issues

### Problem: Slow response times

**Symptoms:**
- RAG queries take more than 10 seconds
- Timeout errors during evaluation

**Solutions:**

1. **Optimize Search Parameters**:
   ```python
   # Reduce number of retrieved documents
   top_k_documents = 3  # Instead of 5 or more
   ```

2. **Parallel Processing**:
   ```python
   # Use concurrent execution for evaluation
   import asyncio
   async def evaluate_batch(queries):
       tasks = [evaluate_single(q) for q in queries]
       return await asyncio.gather(*tasks)
   ```

3. **Content Optimization**:
   ```python
   # Truncate context if too long
   MAX_CONTEXT_LENGTH = 8000
   context = context[:MAX_CONTEXT_LENGTH] if len(context) > MAX_CONTEXT_LENGTH else context
   ```

4. **Model Selection**:
   - Use faster models for development (e.g., gpt-3.5-turbo)
   - Use gpt-4o only when needed for quality

### Problem: Rate limiting errors

**Symptoms:**
```
openai.RateLimitError: Error code: 429
```

**Solutions:**

1. **Implement Exponential Backoff**:
   ```python
   import time
   import random
   
   def retry_with_backoff(func, max_retries=3):
       for attempt in range(max_retries):
           try:
               return func()
           except RateLimitError:
               if attempt == max_retries - 1:
                   raise
               wait_time = (2 ** attempt) + random.uniform(0, 1)
               time.sleep(wait_time)
   ```

2. **Reduce Request Frequency**:
   ```python
   # Add delays between requests
   time.sleep(0.1)  # 100ms delay
   ```

3. **Check Service Quotas**:
   - Review your Azure OpenAI quota limits
   - Consider requesting quota increases if needed

## Development Environment Issues

### Problem: VS Code debugger not working

**Symptoms:**
- Breakpoints not hitting
- Debugger fails to start

**Solutions:**

1. **Check Python Interpreter**:
   ```bash
   # Ensure VS Code is using the correct Python interpreter
   # Command Palette (Ctrl+Shift+P) > "Python: Select Interpreter"
   ```

2. **Verify Launch Configuration**:
   ```json
   // In .vscode/launch.json
   {
       "python": "${workspaceFolder}/.venv/bin/python",  // Check path
       "envFile": "${workspaceFolder}/.env"  // Ensure .env exists
   }
   ```

3. **Environment Variables**:
   ```bash
   # Ensure .env file is properly formatted
   # Check that environment variables load correctly
   ```

### Problem: Development container issues

**Symptoms:**
- Container fails to build
- Extensions not working in container

**Solutions:**

1. **Rebuild Container**:
   ```bash
   # Command Palette > "Dev Containers: Rebuild Container"
   ```

2. **Check Docker Resources**:
   - Ensure Docker has sufficient memory allocated (at least 4GB)
   - Verify disk space is available

3. **Extension Issues**:
   ```json
   // In .devcontainer/devcontainer.json
   "extensions": [
       "ms-python.python",
       "ms-python.debugpy"
   ]
   ```

## Getting Additional Help

### Debugging Steps

1. **Enable Debug Logging**:
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```

2. **Check Service Status**:
   - Azure Service Health dashboard
   - Azure status page: https://status.azure.com/

3. **Test Minimal Examples**:
   - Start with simple, working examples
   - Gradually add complexity

### Support Resources

- **GitHub Issues**: Report bugs and request features
- **Azure Documentation**: Official Azure service documentation
- **Azure Support**: For service-specific issues
- **Community Forums**: Stack Overflow, Azure Community

### Common Diagnostic Commands

```bash
# Check Azure CLI status
az account show

# Test network connectivity
curl -I https://api.openai.com/v1/models

# Verify Python environment
python -c "import azure.search.documents; print('Azure Search OK')"
python -c "import openai; print('OpenAI OK')"

# Check environment variables
printenv | grep AZURE_

# Test authentication
az ad signed-in-user show
```

Remember: Most issues are related to authentication, network connectivity, or configuration. Start with these areas when troubleshooting.