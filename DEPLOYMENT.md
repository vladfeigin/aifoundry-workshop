# Deployment Guide

This guide provides comprehensive instructions for deploying the Azure AI Foundry Workshop to various environments, from development to production.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Infrastructure Setup](#infrastructure-setup)
- [Development Deployment](#development-deployment)
- [Staging Deployment](#staging-deployment)
- [Production Deployment](#production-deployment)
- [Container Deployment](#container-deployment)
- [CI/CD Pipeline](#cicd-pipeline)
- [Monitoring and Observability](#monitoring-and-observability)
- [Security Considerations](#security-considerations)
- [Scaling and Performance](#scaling-and-performance)
- [Backup and Disaster Recovery](#backup-and-disaster-recovery)

## Prerequisites

### Required Azure Services

Before deploying, ensure you have the following Azure services provisioned:

1. **Azure AI Foundry Hub and Project**
2. **Azure AI Search Service** (Standard tier or higher for production)
3. **Azure OpenAI Service** with model deployments:
   - GPT-4o (or gpt-4) for chat completions
   - text-embedding-3-small for embeddings
4. **Azure Document Intelligence Service**
5. **Azure Application Insights** (for monitoring)
6. **Azure Key Vault** (for secrets management)
7. **Azure Storage Account** (for file storage)

### Development Tools

- Azure CLI 2.50.0 or later
- Docker (for containerized deployments)
- Python 3.12 or later
- Git
- Terraform or Azure ARM templates (for infrastructure as code)

### Permissions Required

Ensure your deployment account has:
- **Contributor** role on the resource group
- **Azure AI User** role for AI services
- **Search Service Contributor** for AI Search
- **Cognitive Services OpenAI User** for Azure OpenAI

## Infrastructure Setup

### Option 1: Azure Portal Setup (Quick Start)

1. **Create Resource Group**:
   ```bash
   az group create --name rg-aifoundry-workshop --location eastus2
   ```

2. **Create Azure AI Foundry Hub**:
   - Navigate to Azure Portal > Create a Resource > Azure AI Foundry
   - Choose your subscription and resource group
   - Select a region that supports all required services
   - Configure networking (public or private endpoints)

3. **Create Azure AI Search**:
   ```bash
   az search service create \
     --name search-aifoundry-workshop \
     --resource-group rg-aifoundry-workshop \
     --sku Standard \
     --location eastus2
   ```

4. **Create Azure OpenAI Service**:
   ```bash
   az cognitiveservices account create \
     --name openai-aifoundry-workshop \
     --resource-group rg-aifoundry-workshop \
     --kind OpenAI \
     --sku S0 \
     --location eastus2
   ```

5. **Deploy Models**:
   ```bash
   # Deploy GPT-4o model
   az cognitiveservices account deployment create \
     --name openai-aifoundry-workshop \
     --resource-group rg-aifoundry-workshop \
     --deployment-name gpt-4o \
     --model-name gpt-4o \
     --model-version "2024-08-06" \
     --model-format OpenAI \
     --sku-capacity 10

   # Deploy embedding model
   az cognitiveservices account deployment create \
     --name openai-aifoundry-workshop \
     --resource-group rg-aifoundry-workshop \
     --deployment-name text-embedding-3-small \
     --model-name text-embedding-3-small \
     --model-version "1" \
     --model-format OpenAI \
     --sku-capacity 10
   ```

### Option 2: Infrastructure as Code (Terraform)

Create `main.tf` file:

```hcl
terraform {
  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.0"
    }
  }
}

provider "azurerm" {
  features {}
}

# Resource Group
resource "azurerm_resource_group" "main" {
  name     = "rg-aifoundry-workshop-${var.environment}"
  location = var.location
}

# Azure AI Search
resource "azurerm_search_service" "main" {
  name                = "search-aifoundry-${var.environment}"
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location
  sku                 = var.search_sku
  
  tags = var.tags
}

# Azure OpenAI Service
resource "azurerm_cognitive_account" "openai" {
  name                = "openai-aifoundry-${var.environment}"
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location
  kind                = "OpenAI"
  sku_name            = "S0"
  
  tags = var.tags
}

# Document Intelligence
resource "azurerm_cognitive_account" "doc_intel" {
  name                = "docintel-aifoundry-${var.environment}"
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location
  kind                = "FormRecognizer"
  sku_name            = "S0"
  
  tags = var.tags
}

# Application Insights
resource "azurerm_application_insights" "main" {
  name                = "appi-aifoundry-${var.environment}"
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location
  application_type    = "web"
  
  tags = var.tags
}

# Key Vault
resource "azurerm_key_vault" "main" {
  name                = "kv-aifoundry-${var.environment}"
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location
  tenant_id           = data.azurerm_client_config.current.tenant_id
  sku_name            = "standard"
  
  tags = var.tags
}

# Variables
variable "environment" {
  description = "Environment name (dev, staging, prod)"
  type        = string
}

variable "location" {
  description = "Azure region"
  type        = string
  default     = "East US 2"
}

variable "search_sku" {
  description = "Azure Search SKU"
  type        = string
  default     = "standard"
}

variable "tags" {
  description = "Resource tags"
  type        = map(string)
  default = {
    Project     = "AIFoundryWorkshop"
    Environment = "dev"
  }
}

# Outputs
output "search_service_name" {
  value = azurerm_search_service.main.name
}

output "openai_endpoint" {
  value = azurerm_cognitive_account.openai.endpoint
}

output "doc_intel_endpoint" {
  value = azurerm_cognitive_account.doc_intel.endpoint
}
```

Deploy with Terraform:
```bash
terraform init
terraform plan -var="environment=dev"
terraform apply -var="environment=dev"
```

## Development Deployment

### Local Development Setup

1. **Clone Repository**:
   ```bash
   git clone https://github.com/vladfeigin/aifoundry-workshop.git
   cd aifoundry-workshop
   ```

2. **Environment Configuration**:
   ```bash
   cp .env.template .env
   # Edit .env with your Azure service endpoints and keys
   ```

3. **Install Dependencies**:
   ```bash
   # Using pip
   pip install azure-ai-documentintelligence azure-search-documents azure-identity python-dotenv rich openai azure-ai-projects azure-monitor-opentelemetry opentelemetry-sdk azure-ai-evaluation

   # Or using UV (if available)
   uv sync
   ```

4. **Initialize Search Index**:
   ```bash
   python -m aisearch.create_search_index \
     --search-service your-search-service-name \
     --index-name ai-foundry-workshop-index \
     --use-api-key
   ```

5. **Process Sample Documents**:
   ```bash
   # Convert PDFs to Markdown
   python -m docintel.pdf-2-md ./docintel/data/GPT-4-Technical-Report.pdf ./docintel/data/GPT-4-Technical-Report.md

   # Ingest documents into search index
   python -m aisearch.ingest_documents \
     --search-service your-search-service-name \
     --index-name ai-foundry-workshop-index
   ```

6. **Test RAG Agent**:
   ```bash
   python -m agents.rag.rag_agent
   ```

### Development Container Setup

1. **Prerequisites**:
   - Docker Desktop
   - VS Code with Dev Containers extension

2. **Open in Container**:
   ```bash
   code .
   # VS Code will prompt to "Reopen in Container"
   ```

3. **Container Configuration** (`.devcontainer/devcontainer.json`):
   ```json
   {
     "name": "Azure AI Foundry Workshop",
     "image": "mcr.microsoft.com/devcontainers/python:3.12",
     "features": {
       "ghcr.io/devcontainers/features/azure-cli:1": {}
     },
     "postCreateCommand": "pip install -r requirements.txt",
     "customizations": {
       "vscode": {
         "extensions": [
           "ms-python.python",
           "ms-python.debugpy",
           "ms-azuretools.vscode-azureresourcegroups"
         ]
       }
     },
     "forwardPorts": [8000],
     "remoteEnv": {
       "PYTHONPATH": "${containerWorkspaceFolder}"
     }
   }
   ```

## Staging Deployment

### Azure Container Instances

1. **Create Container Registry**:
   ```bash
   az acr create \
     --name acraifoundryworkshop \
     --resource-group rg-aifoundry-workshop \
     --sku Standard \
     --admin-enabled true
   ```

2. **Build and Push Image**:
   ```bash
   # Create Dockerfile
   cat > Dockerfile << 'EOF'
   FROM python:3.12-slim

   WORKDIR /app
   COPY . /app

   RUN pip install azure-ai-documentintelligence azure-search-documents azure-identity python-dotenv rich openai azure-ai-projects azure-monitor-opentelemetry opentelemetry-sdk azure-ai-evaluation

   EXPOSE 8000
   CMD ["python", "-m", "agents.rag.rag_agent"]
   EOF

   # Build and push
   az acr build --registry acraifoundryworkshop --image aifoundry-workshop:latest .
   ```

3. **Deploy to Container Instances**:
   ```bash
   az container create \
     --resource-group rg-aifoundry-workshop \
     --name aci-aifoundry-workshop \
     --image acraifoundryworkshop.azurecr.io/aifoundry-workshop:latest \
     --cpu 2 \
     --memory 4 \
     --registry-login-server acraifoundryworkshop.azurecr.io \
     --registry-username $(az acr credential show --name acraifoundryworkshop --query username -o tsv) \
     --registry-password $(az acr credential show --name acraifoundryworkshop --query passwords[0].value -o tsv) \
     --environment-variables \
       AZURE_SEARCH_SERVICE_NAME=your-search-service \
       AZURE_SEARCH_INDEX_NAME=your-index-name \
       PROJECT_ENDPOINT=your-project-endpoint
   ```

### Azure Container Apps

1. **Create Container App Environment**:
   ```bash
   az containerapp env create \
     --name cae-aifoundry-workshop \
     --resource-group rg-aifoundry-workshop \
     --location eastus2
   ```

2. **Deploy Container App**:
   ```bash
   az containerapp create \
     --name ca-aifoundry-workshop \
     --resource-group rg-aifoundry-workshop \
     --environment cae-aifoundry-workshop \
     --image acraifoundryworkshop.azurecr.io/aifoundry-workshop:latest \
     --target-port 8000 \
     --ingress external \
     --registry-server acraifoundryworkshop.azurecr.io \
     --registry-username $(az acr credential show --name acraifoundryworkshop --query username -o tsv) \
     --registry-password $(az acr credential show --name acraifoundryworkshop --query passwords[0].value -o tsv) \
     --env-vars \
       AZURE_SEARCH_SERVICE_NAME=your-search-service \
       AZURE_SEARCH_INDEX_NAME=your-index-name \
       PROJECT_ENDPOINT=your-project-endpoint
   ```

## Production Deployment

### Azure Kubernetes Service (AKS)

1. **Create AKS Cluster**:
   ```bash
   az aks create \
     --resource-group rg-aifoundry-workshop \
     --name aks-aifoundry-workshop \
     --node-count 3 \
     --node-vm-size Standard_D2s_v3 \
     --enable-addons monitoring \
     --enable-managed-identity \
     --attach-acr acraifoundryworkshop
   ```

2. **Get Credentials**:
   ```bash
   az aks get-credentials \
     --resource-group rg-aifoundry-workshop \
     --name aks-aifoundry-workshop
   ```

3. **Create Kubernetes Manifests**:

   **Namespace** (`k8s/namespace.yaml`):
   ```yaml
   apiVersion: v1
   kind: Namespace
   metadata:
     name: aifoundry-workshop
   ```

   **ConfigMap** (`k8s/configmap.yaml`):
   ```yaml
   apiVersion: v1
   kind: ConfigMap
   metadata:
     name: aifoundry-config
     namespace: aifoundry-workshop
   data:
     AZURE_SEARCH_SERVICE_NAME: "your-search-service"
     AZURE_SEARCH_INDEX_NAME: "your-index-name"
     PROJECT_ENDPOINT: "your-project-endpoint"
   ```

   **Secret** (`k8s/secret.yaml`):
   ```yaml
   apiVersion: v1
   kind: Secret
   metadata:
     name: aifoundry-secrets
     namespace: aifoundry-workshop
   type: Opaque
   data:
     AZURE_SEARCH_API_KEY: <base64-encoded-key>
     AZURE_OPENAI_API_KEY: <base64-encoded-key>
   ```

   **Deployment** (`k8s/deployment.yaml`):
   ```yaml
   apiVersion: apps/v1
   kind: Deployment
   metadata:
     name: aifoundry-workshop
     namespace: aifoundry-workshop
   spec:
     replicas: 3
     selector:
       matchLabels:
         app: aifoundry-workshop
     template:
       metadata:
         labels:
           app: aifoundry-workshop
       spec:
         containers:
         - name: aifoundry-workshop
           image: acraifoundryworkshop.azurecr.io/aifoundry-workshop:latest
           ports:
           - containerPort: 8000
           envFrom:
           - configMapRef:
               name: aifoundry-config
           - secretRef:
               name: aifoundry-secrets
           resources:
             requests:
               memory: "2Gi"
               cpu: "1000m"
             limits:
               memory: "4Gi"
               cpu: "2000m"
           livenessProbe:
             httpGet:
               path: /health
               port: 8000
             initialDelaySeconds: 30
             periodSeconds: 10
           readinessProbe:
             httpGet:
               path: /ready
               port: 8000
             initialDelaySeconds: 5
             periodSeconds: 5
   ```

   **Service** (`k8s/service.yaml`):
   ```yaml
   apiVersion: v1
   kind: Service
   metadata:
     name: aifoundry-workshop-service
     namespace: aifoundry-workshop
   spec:
     selector:
       app: aifoundry-workshop
     ports:
     - protocol: TCP
       port: 80
       targetPort: 8000
     type: LoadBalancer
   ```

4. **Deploy to AKS**:
   ```bash
   kubectl apply -f k8s/
   ```

### Azure App Service

1. **Create App Service Plan**:
   ```bash
   az appservice plan create \
     --name asp-aifoundry-workshop \
     --resource-group rg-aifoundry-workshop \
     --sku P1V2 \
     --is-linux
   ```

2. **Create Web App**:
   ```bash
   az webapp create \
     --resource-group rg-aifoundry-workshop \
     --plan asp-aifoundry-workshop \
     --name wa-aifoundry-workshop \
     --deployment-container-image-name acraifoundryworkshop.azurecr.io/aifoundry-workshop:latest \
     --docker-registry-server-url https://acraifoundryworkshop.azurecr.io \
     --docker-registry-server-user $(az acr credential show --name acraifoundryworkshop --query username -o tsv) \
     --docker-registry-server-password $(az acr credential show --name acraifoundryworkshop --query passwords[0].value -o tsv)
   ```

3. **Configure App Settings**:
   ```bash
   az webapp config appsettings set \
     --resource-group rg-aifoundry-workshop \
     --name wa-aifoundry-workshop \
     --settings \
       AZURE_SEARCH_SERVICE_NAME=your-search-service \
       AZURE_SEARCH_INDEX_NAME=your-index-name \
       PROJECT_ENDPOINT=your-project-endpoint
   ```

## Container Deployment

### Multi-stage Dockerfile for Production

```dockerfile
# Build stage
FROM python:3.12-slim as builder

WORKDIR /app
COPY pyproject.toml ./
RUN pip install --no-cache-dir build && \
    python -m build

# Production stage
FROM python:3.12-slim

# Create non-root user
RUN useradd --create-home --shell /bin/bash appuser

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY --from=builder /app/dist/*.whl ./
RUN pip install --no-cache-dir *.whl && \
    rm *.whl

# Copy application code
COPY . .

# Change ownership to appuser
RUN chown -R appuser:appuser /app
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

# Use exec form for proper signal handling
CMD ["python", "-m", "agents.rag.rag_agent"]
```

### Docker Compose for Development

```yaml
version: '3.8'

services:
  aifoundry-workshop:
    build: .
    ports:
      - "8000:8000"
    environment:
      - AZURE_SEARCH_SERVICE_NAME=${AZURE_SEARCH_SERVICE_NAME}
      - AZURE_SEARCH_INDEX_NAME=${AZURE_SEARCH_INDEX_NAME}
      - PROJECT_ENDPOINT=${PROJECT_ENDPOINT}
    env_file:
      - .env
    volumes:
      - ./logs:/app/logs
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped

volumes:
  redis_data:
```

## CI/CD Pipeline

### GitHub Actions Workflow

```yaml
name: Deploy Azure AI Foundry Workshop

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

env:
  REGISTRY: acraifoundryworkshop.azurecr.io
  IMAGE_NAME: aifoundry-workshop

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.12'
    
    - name: Install dependencies
      run: |
        pip install azure-ai-documentintelligence azure-search-documents azure-identity python-dotenv rich openai azure-ai-projects azure-monitor-opentelemetry opentelemetry-sdk azure-ai-evaluation
        pip install pytest flake8 black mypy
    
    - name: Lint with flake8
      run: flake8 .
    
    - name: Format with black
      run: black --check .
    
    - name: Type check with mypy
      run: mypy .
    
    - name: Test with pytest
      run: pytest

  build-and-deploy:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Azure Login
      uses: azure/login@v1
      with:
        creds: ${{ secrets.AZURE_CREDENTIALS }}
    
    - name: Build and push Docker image
      run: |
        az acr build --registry ${{ env.REGISTRY }} --image ${{ env.IMAGE_NAME }}:${{ github.sha }} .
        az acr build --registry ${{ env.REGISTRY }} --image ${{ env.IMAGE_NAME }}:latest .
    
    - name: Deploy to AKS
      run: |
        az aks get-credentials --resource-group rg-aifoundry-workshop --name aks-aifoundry-workshop
        kubectl set image deployment/aifoundry-workshop aifoundry-workshop=${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:${{ github.sha }} -n aifoundry-workshop
        kubectl rollout status deployment/aifoundry-workshop -n aifoundry-workshop
```

### Azure DevOps Pipeline

```yaml
trigger:
- main

variables:
  imageRepository: 'aifoundry-workshop'
  containerRegistry: 'acraifoundryworkshop.azurecr.io'
  dockerfilePath: '$(Build.SourcesDirectory)/Dockerfile'
  tag: '$(Build.BuildId)'

stages:
- stage: Build
  displayName: Build and push stage
  jobs:
  - job: Build
    displayName: Build
    pool:
      vmImage: ubuntu-latest
    steps:
    - task: Docker@2
      displayName: Build and push an image to container registry
      inputs:
        command: buildAndPush
        repository: $(imageRepository)
        dockerfile: $(dockerfilePath)
        containerRegistry: $(dockerRegistryServiceConnection)
        tags: |
          $(tag)
          latest

- stage: Deploy
  displayName: Deploy stage
  dependsOn: Build
  jobs:
  - deployment: Deploy
    displayName: Deploy
    pool:
      vmImage: ubuntu-latest
    environment: 'production'
    strategy:
      runOnce:
        deploy:
          steps:
          - task: KubernetesManifest@0
            displayName: Deploy to Kubernetes cluster
            inputs:
              action: deploy
              manifests: |
                $(Pipeline.Workspace)/manifests/deployment.yml
                $(Pipeline.Workspace)/manifests/service.yml
              containers: |
                $(containerRegistry)/$(imageRepository):$(tag)
```

## Monitoring and Observability

### Application Insights Configuration

```python
# monitoring.py
from azure.monitor.opentelemetry import configure_azure_monitor
from opentelemetry import trace
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.instrumentation.openai import OpenAIInstrumentor

def configure_monitoring(connection_string: str):
    # Configure Azure Monitor
    configure_azure_monitor(connection_string=connection_string)
    
    # Enable auto-instrumentation
    RequestsInstrumentor().instrument()
    OpenAIInstrumentor().instrument()
    
    return trace.get_tracer(__name__)
```

### Custom Metrics

```python
# metrics.py
from azure.monitor.opentelemetry.exporter import AzureMonitorMetricExporter
from opentelemetry import metrics
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader

def setup_custom_metrics():
    metric_reader = PeriodicExportingMetricReader(
        AzureMonitorMetricExporter(connection_string=connection_string),
        export_interval_millis=5000,
    )
    
    metrics.set_meter_provider(MeterProvider(metric_readers=[metric_reader]))
    meter = metrics.get_meter(__name__)
    
    # Custom metrics
    response_time_histogram = meter.create_histogram(
        name="rag_response_time",
        description="RAG agent response time",
        unit="ms"
    )
    
    accuracy_gauge = meter.create_up_down_counter(
        name="rag_accuracy_score",
        description="RAG agent accuracy score"
    )
    
    return response_time_histogram, accuracy_gauge
```

### Alerting Rules

```json
{
  "name": "RAG Agent High Error Rate",
  "description": "Alert when error rate exceeds 5%",
  "enabled": true,
  "condition": {
    "allOf": [
      {
        "metricName": "exceptions/count",
        "operator": "GreaterThan",
        "threshold": 5,
        "timeAggregation": "Average",
        "windowSize": "PT5M"
      }
    ]
  },
  "actions": [
    {
      "actionGroupId": "/subscriptions/{subscription-id}/resourceGroups/{resource-group}/providers/Microsoft.Insights/actionGroups/{action-group-name}"
    }
  ]
}
```

## Security Considerations

### Network Security

1. **Virtual Network Integration**:
   ```bash
   # Create VNet
   az network vnet create \
     --name vnet-aifoundry-workshop \
     --resource-group rg-aifoundry-workshop \
     --address-prefix 10.0.0.0/16 \
     --subnet-name subnet-apps \
     --subnet-prefix 10.0.1.0/24
   ```

2. **Private Endpoints**:
   ```bash
   # Create private endpoint for AI Search
   az network private-endpoint create \
     --name pe-search-aifoundry \
     --resource-group rg-aifoundry-workshop \
     --vnet-name vnet-aifoundry-workshop \
     --subnet subnet-apps \
     --private-connection-resource-id /subscriptions/{subscription}/resourceGroups/rg-aifoundry-workshop/providers/Microsoft.Search/searchServices/search-aifoundry-workshop \
     --connection-name search-connection \
     --group-id searchService
   ```

### Identity and Access Management

1. **Managed Identity**:
   ```bash
   # Create user-assigned managed identity
   az identity create \
     --resource-group rg-aifoundry-workshop \
     --name id-aifoundry-workshop

   # Grant permissions
   az role assignment create \
     --assignee $(az identity show --resource-group rg-aifoundry-workshop --name id-aifoundry-workshop --query principalId -o tsv) \
     --role "Azure AI User" \
     --scope /subscriptions/{subscription}/resourceGroups/rg-aifoundry-workshop
   ```

2. **Key Vault Integration**:
   ```python
   from azure.keyvault.secrets import SecretClient
   from azure.identity import DefaultAzureCredential

   def get_secret_from_keyvault(vault_url: str, secret_name: str) -> str:
       credential = DefaultAzureCredential()
       client = SecretClient(vault_url=vault_url, credential=credential)
       secret = client.get_secret(secret_name)
       return secret.value
   ```

### Content Security

```python
# content_security.py
from azure.ai.contentsafety import ContentSafetyClient
from azure.core.credentials import AzureKeyCredential

class ContentSecurityFilter:
    def __init__(self, endpoint: str, key: str):
        self.client = ContentSafetyClient(endpoint, AzureKeyCredential(key))
    
    def analyze_text(self, text: str) -> bool:
        request = AnalyzeTextOptions(text=text)
        response = self.client.analyze_text(request)
        
        # Check if content is safe
        for result in response.categoriesAnalysis:
            if result.severity > 2:  # High severity threshold
                return False
        return True
```

## Scaling and Performance

### Horizontal Pod Autoscaler (HPA)

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: aifoundry-workshop-hpa
  namespace: aifoundry-workshop
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: aifoundry-workshop
  minReplicas: 3
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

### Connection Pooling

```python
# connection_pool.py
import asyncio
from typing import Dict, Any
from azure.search.documents.aio import SearchClient
from openai import AsyncAzureOpenAI

class ConnectionPool:
    def __init__(self):
        self._search_clients: Dict[str, SearchClient] = {}
        self._openai_clients: Dict[str, AsyncAzureOpenAI] = {}
        self._semaphore = asyncio.Semaphore(50)  # Limit concurrent connections
    
    async def get_search_client(self, service_name: str) -> SearchClient:
        if service_name not in self._search_clients:
            self._search_clients[service_name] = SearchClient(
                endpoint=f"https://{service_name}.search.windows.net",
                index_name="default",
                credential=DefaultAzureCredential()
            )
        return self._search_clients[service_name]
    
    async def execute_with_limit(self, coro):
        async with self._semaphore:
            return await coro
```

### Caching Strategy

```python
# caching.py
import redis
import json
import hashlib
from typing import Optional, Any

class ResponseCache:
    def __init__(self, redis_url: str, ttl: int = 3600):
        self.redis_client = redis.from_url(redis_url)
        self.ttl = ttl
    
    def _generate_key(self, query: str, context_hash: str) -> str:
        combined = f"{query}:{context_hash}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    def get(self, query: str, context_hash: str) -> Optional[dict]:
        key = self._generate_key(query, context_hash)
        cached = self.redis_client.get(key)
        return json.loads(cached) if cached else None
    
    def set(self, query: str, context_hash: str, response: dict) -> None:
        key = self._generate_key(query, context_hash)
        self.redis_client.setex(key, self.ttl, json.dumps(response))
```

## Backup and Disaster Recovery

### Data Backup Strategy

1. **Search Index Backup**:
   ```python
   # backup_search_index.py
   from azure.search.documents import SearchClient
   from azure.storage.blob import BlobServiceClient
   import json

   def backup_search_index(search_client: SearchClient, storage_client: BlobServiceClient):
       # Export all documents
       results = search_client.search("*", include_total_count=True)
       documents = [doc for doc in results]
       
       # Upload to blob storage
       backup_data = json.dumps(documents, default=str)
       blob_client = storage_client.get_blob_client(
           container="backups",
           blob=f"search-index-{datetime.now().isoformat()}.json"
       )
       blob_client.upload_blob(backup_data, overwrite=True)
   ```

2. **Configuration Backup**:
   ```bash
   # Export ARM template
   az group export \
     --resource-group rg-aifoundry-workshop \
     --include-parameter-default-value \
     > backup/infrastructure-template.json
   ```

### Disaster Recovery Plan

1. **Multi-Region Deployment**:
   ```yaml
   # terraform/disaster-recovery.tf
   resource "azurerm_traffic_manager_profile" "main" {
     name                = "tm-aifoundry-workshop"
     resource_group_name = azurerm_resource_group.main.name
     
     traffic_routing_method = "Priority"
     
     dns_config {
       relative_name = "aifoundry-workshop"
       ttl           = 30
     }
     
     monitor_config {
       protocol = "HTTPS"
       port     = 443
       path     = "/health"
     }
   }
   ```

2. **Recovery Procedures**:
   ```bash
   #!/bin/bash
   # disaster-recovery.sh
   
   # 1. Deploy infrastructure to secondary region
   terraform apply -var="location=westus2" -var="environment=dr"
   
   # 2. Restore search index
   python scripts/restore_search_index.py --source-backup latest
   
   # 3. Update DNS to point to secondary region
   az network traffic-manager endpoint update \
     --resource-group rg-aifoundry-workshop \
     --profile-name tm-aifoundry-workshop \
     --name primary \
     --type azureEndpoints \
     --endpoint-status Disabled
   
   az network traffic-manager endpoint update \
     --resource-group rg-aifoundry-workshop \
     --profile-name tm-aifoundry-workshop \
     --name secondary \
     --type azureEndpoints \
     --endpoint-status Enabled
   ```

### Health Checks

```python
# health_checks.py
from fastapi import FastAPI, HTTPException
import asyncio

app = FastAPI()

@app.get("/health")
async def health_check():
    checks = {
        "search_service": await check_search_service(),
        "openai_service": await check_openai_service(),
        "doc_intel_service": await check_doc_intel_service()
    }
    
    if all(checks.values()):
        return {"status": "healthy", "checks": checks}
    else:
        raise HTTPException(status_code=503, detail={"status": "unhealthy", "checks": checks})

@app.get("/ready")
async def readiness_check():
    # Quick check for readiness
    return {"status": "ready"}

async def check_search_service() -> bool:
    try:
        # Test search service connectivity
        return True
    except Exception:
        return False
```

This deployment guide provides comprehensive instructions for deploying the Azure AI Foundry Workshop across different environments and scenarios. Choose the deployment strategy that best fits your requirements and environment constraints.