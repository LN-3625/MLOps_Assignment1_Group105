@echo off
SET IMAGE_NAME=narayana3625/housing-api:latest

echo Building Docker image...
docker build -t %IMAGE_NAME% .

echo Pushing to Docker Hub...
docker push %IMAGE_NAME%

echo Starting services with Docker Compose...
docker-compose up --build

@REM @echo off
@REM docker rm -f housing-api-container
@REM docker rmi housing-api
@REM docker build -t housing-api .
@REM docker run -d --name housing-api-container -p 8000:8000 ^
@REM   -v "%cd%/app/mlruns:/app/mlruns" ^
@REM   -e MLFLOW_TRACKING_URI="file:app/mlruns" ^
@REM   housing-api
