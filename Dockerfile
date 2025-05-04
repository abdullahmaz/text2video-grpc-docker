# Use the official Python image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Create a non-root user with user ID 1000
RUN useradd -m -u 1000 user

# Set environment variables for the new user
ENV HOME=/home/user
ENV PATH="$HOME/.local/bin:$PATH"

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Create necessary directories with correct permissions
RUN mkdir -p /app/hf_cache /app/videos && \
    chown -R user:user /app

# Copy dependency specification
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy app files with appropriate ownership
COPY --chown=user . .

# Switch to the non-root user
USER user

# Set environment variable for Hugging Face cache
ENV HF_HOME=/app/hf_cache

# Expose gRPC port
EXPOSE 50051

# Run the app
CMD ["python", "-m", "app.nlp"]
