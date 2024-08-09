#!/bin/bash

# Load environment variables from /src/train_model/.env file
if [ -f ../src/train_model/.env ]; then
  export $(cat ../src/train_model/.env | xargs)
else
  echo ".env file not found in /src/train_model. Make sure it exists."
  exit 1
fi

# 1. Define the environment variables
echo "STORAGE_BUCKET: $STORAGE_BUCKET"
echo "PROJECT_ID: $GCP_PROJECT_ID"
echo "ACCOUNT: $GCP_ACCOUNT"


# Step 2: Define repository name and create the repository in Artifact Registry
REPO_NAME='birds-app'
echo "Creating repository '$REPO_NAME' in Artifact Registry..."
gcloud artifacts repositories create $REPO_NAME --repository-format=docker \
--location=europe-west1 --description="Docker repository"

# Step 3: Define the image URI
IMAGE_URI="europe-west1-docker.pkg.dev/$PROJECT_ID/$REPO_NAME/birds_image:latest"

# Step 4: Configure Docker to use gcloud as a credential helper
echo "Configuring Docker..."
gcloud auth configure-docker europe-west1-docker.pkg.dev

# Step 5: Build the Docker image, specifying the correct directory for the Dockerfile
echo "Building Docker image..."
docker build ../src/train_model -t $IMAGE_URI

# Step 6: Push the Docker image to Artifact Registry
echo "Pushing Docker image to Artifact Registry..."
docker push $IMAGE_URI

# Step 7: Create the Vertex AI Custom Training Job using gcloud
echo "Creating Vertex AI Custom Training Job..."
JOB_NAME="birds-sdk-job"

gcloud ai custom-jobs create \
  --region=europe-west1 \
  --display-name=$JOB_NAME \
  --container-image-uri=$IMAGE_URI \
  --job-dir=gs://$STORAGE_BUCKET \
  --worker-pool-spec=machine-type=n1-standard-8,accelerator-type=NVIDIA_TESLA_V100,accelerator-count=1,replica-count=1

echo "Training job created successfully. Check the Vertex AI console for job details."

# Step 8: Use Vertex AI to make predictions after the model has been trained
echo "Creating Vertex AI Endpoint for predictions..."

ENDPOINT_NAME="birds-predict-endpoint"

gcloud ai endpoints create \
  --region=europe-west1 \
  --display-name=$ENDPOINT_NAME

ENDPOINT_ID=$(gcloud ai endpoints list --region=europe-west1 --filter="displayName:$ENDPOINT_NAME" --format="value(name)")

if [ -z "$ENDPOINT_ID" ]; then
  echo "Error: Endpoint creation failed."
  exit 1
fi

echo "Deploying model to the endpoint..."

MODEL_NAME="birds-model"
MODEL_URI="gs://$STORAGE_BUCKET/model"
DEPLOYED_MODEL_NAME="birds-deployed-model"

gcloud ai models upload \
  --region=europe-west1 \
  --display-name=$MODEL_NAME \
  --artifact-uri=$MODEL_URI \
  --container-image-uri=$IMAGE_URI

MODEL_ID=$(gcloud ai models list --region=europe-west1 --filter="displayName:$MODEL_NAME" --format="value(name)")

if [ -z "$MODEL_ID" ]; then
  echo "Error: Model upload failed."
  exit 1
fi

gcloud ai endpoints deploy-model $ENDPOINT_ID \
  --region=europe-west1 \
  --model=$MODEL_ID \
  --display-name=$DEPLOYED_MODEL_NAME \
  --traffic-split=0=100

echo "Model deployed successfully to endpoint $ENDPOINT_NAME with ID $ENDPOINT_ID."

echo "You can now make predictions using the deployed model. Example command:"
echo "gcloud ai endpoints predict $ENDPOINT_ID --region=europe-west1 --json-request=INPUT_JSON"
