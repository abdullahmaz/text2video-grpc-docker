FROM python:3.10-slim

WORKDIR /app

# Install dependencies
RUN apt-get update && apt-get install -y ffmpeg && apt-get clean

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download the model before copying app code
COPY scripts/download_model.py ./scripts/download_model.py
RUN python3 ./scripts/download_model.py

# Copy source files
COPY app/ .

# Expose frontend port
EXPOSE 7860

# Run server in background and frontend as main app
CMD python3 frontend.py & python3 server.py 