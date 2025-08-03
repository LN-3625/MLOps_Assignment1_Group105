#!/bin/bash

# Stop and remove existing container if it exists
docker rm -f housing-api-container 2>/dev/null

# Remove old image if it exists
docker rmi housing-api 2>/dev/null

# Build new Docker image
docker build -t housing-api .

# Run the container with volume mount for mlruns
docker run -d --name housing-api-container -p 8000:8000 \
  -v "$(pwd)/app/mlruns:/app/mlruns" \
  -e MLFLOW_TRACKING_URI="file:/app/mlruns" \
  housing-api