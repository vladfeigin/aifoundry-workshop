


import os
import os
from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient

#load environment variables
from dotenv import load_dotenv
load_dotenv()

# Required environment variables

model_endpoint = os.environ["AZURE_OPENAI_ENDPOINT"] # https://<account>.services.ai.azure.com
model_api_key = os.environ["AZURE_OPENAI_API_KEY"]
model_deployment_name = os.environ["AZURE_EVALUATION_MODEL"] # e.g. gpt-4o-mini

# Optional – reuse an existing dataset
dataset_name    = os.environ.get("DATASET_NAME",    "single-turn-eval-ds-agent-output")
dataset_version = os.environ.get("DATASET_VERSION", "1.0")

# Create the project client (Foundry project and credentials)
project_client = AIProjectClient(
            subscription_id=os.environ["AZURE_SUBSCRIPTION_ID"],
            resource_group_name=os.environ["AZURE_RESOURCE_GROUP"],
            project_name=os.environ["AZURE_PROJECT_NAME"],
            credential=DefaultAzureCredential(),
            endpoint=os.environ["AZURE_PROJECT_ENDPOINT"],
        )

def upload_evaluation_dataset():
    

    # Upload a local jsonl file (skip if you already have a Dataset registered)
    data_id = project_client.datasets.upload_file(
        name=dataset_name,
        version=dataset_version,
        file_path="../data/output/single-turn-eval-ds.jsonl",
    ).id


if __name__ == "__main__":
    upload_evaluation_dataset()
