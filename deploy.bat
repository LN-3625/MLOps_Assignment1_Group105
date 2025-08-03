@echo off
docker rm -f housing-api-container
docker rmi housing-api
docker build -t housing-api .
docker run -d --name housing-api-container -p 8000:8000 ^
  -v "%cd%/app/mlruns:/app/mlruns" ^
  -e MLFLOW_TRACKING_URI="file:app/mlruns" ^
  housing-api