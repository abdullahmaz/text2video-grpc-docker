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

# Create a non-root user and switch to it
RUN useradd -m -u 1000 user
USER user

# Set environment variables for the user
ENV HOME=/home/user
ENV HF_HOME=$HOME/hf_cache

# Create necessary directories with correct permissions
RUN mkdir -p $HF_HOME && mkdir -p /app/videos && chown -R user:user $HF_HOME /app/videos

# Copy dependency specification
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy app files
COPY --chown=user:user . .

# Expose gRPC port
EXPOSE 50051

# Run the app
CMD ["python", "app/nlp.py"]