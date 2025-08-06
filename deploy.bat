@echo off
SET IMAGE_NAME=narayana3625/housing-api:latest

echo Building Docker image...
docker build -t %IMAGE_NAME% .

echo Pushing to Docker Hub...
docker push %IMAGE_NAME%

echo Starting services with Docker Compose...
docker-compose up --build