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

# Copy dependency specification
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download the model before copying app code
# COPY scripts/download_model.py ./scripts/download_model.py
# RUN python3 ./scripts/download_model.py

# Copy app files
COPY . .

# Create directory for videos
RUN mkdir -p videos

# Expose gRPC port
EXPOSE 50051

# Run the app
CMD ["python", "app/nlp.py"]