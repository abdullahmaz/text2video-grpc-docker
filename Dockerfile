# Use the official Python image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Create app directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    && apt-get clean

# Create necessary directories with correct permissions
RUN mkdir -p /home/user/hf_cache && mkdir -p /app/videos

# Copy dependency specification
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy app files
COPY . .

# Change ownership of the directories to the current user
RUN chown -R $(id -u):$(id -g) /home/user/hf_cache /app/videos

# Expose gRPC port
EXPOSE 50051

# Run the app
CMD ["python", "app/nlp.py"]