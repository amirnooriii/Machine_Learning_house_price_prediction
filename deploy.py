from google.cloud import aiplatform
import os

# Load environment variables
PROJECT_ID = os.environ['PROJECT_ID']
REGION = os.environ.get('REGION', 'us-central1')
BUCKET_NAME = os.environ['MODEL_BUCKET']
MODEL_DISPLAY_NAME = os.environ.get('MODEL_DISPLAY_NAME', 'csv-model')
ENDPOINT_DISPLAY_NAME = os.environ.get('ENDPOINT_DISPLAY_NAME', 'csv-endpoint')

# GCS path where model artifacts are stored
artifact_uri = f"gs://{BUCKET_NAME}/model/"

# Initialize Vertex AI
aiplatform.init(
    project=PROJECT_ID,
    location=REGION,
    staging_bucket=BUCKET_NAME
)

# Upload the model to Vertex AI
model = aiplatform.Model.upload(
    display_name=MODEL_DISPLAY_NAME,
    artifact_uri=artifact_uri,
    serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.0-24:latest"
)

# Try to find existing endpoint, otherwise create a new one
existing_endpoints = aiplatform.Endpoint.list(
    filter=f'display_name="{ENDPOINT_DISPLAY_NAME}"',
    order_by="create_time desc"
)

if existing_endpoints:
    endpoint = existing_endpoints[0]
    print(f"Using existing endpoint: {endpoint.display_name}")
else:
    endpoint = aiplatform.Endpoint.create(display_name=ENDPOINT_DISPLAY_NAME)
    print(f"Created new endpoint: {endpoint.display_name}")

# Deploy model to the endpoint
model.deploy(
    endpoint=endpoint,
    machine_type="n1-standard-2",
    traffic_split={"0": 100}
)

print("✅ Model deployed successfully.")
