@echo off
echo Pulling latest image from Docker Hub...
docker pull narayana3625/housing-api:latest

echo Stopping and removing old container (if exists)...
docker rm -f housing-api-container >nul 2>&1

docker run -p 8000:8000 narayana3625/housing-api