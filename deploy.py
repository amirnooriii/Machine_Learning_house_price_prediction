from google.cloud import aiplatform
import os

PROJECT_ID = "studious-pulsar-468715-d6"
REGION = "us-central1"
BUCKET_NAME = "data-analytics-prod-2847-kw-region"

aiplatform.init(project=PROJECT_ID, location=REGION, staging_bucket=BUCKET_NAME)

model = aiplatform.Model.upload(
    display_name="csv-model",
    artifact_uri=f"gs://{BUCKET_NAME}/model/",
    serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.0-24:latest"
)

endpoint = aiplatform.Endpoint.create(display_name="csv-endpoint")
model.deploy(endpoint=endpoint, machine_type="n1-standard-2")
