---
title: Text-to-Video Generator
emoji: 🎥
colorFrom: pink
colorTo: blue
sdk: gradio
sdk_version: 5.29.0
app_file: app/nlp.py
pinned: false
hardware: gpu
---


# 🎥 Text-to-Video gRPC Microservice

This project implements a text-to-video generation microservice using a gRPC backend, powered by the `zeroscope_v2_576w` diffusion model. It features a containerized API, concurrent request support, and a minimal Gradio frontend for user interaction. It is designed for reproducibility, ease of testing, and deployment.

---

## 🚀 Features

- Generate videos from text prompts using Hugging Face's Diffusers
- gRPC API with structured response (status code, message, video path)
- Minimal Gradio frontend for user testing
- Video filtering options (None, Grayscale, Sepia) for stylized output
- Audio transcription support using Whisper model
- Full Docker containerization
- Concurrent request support via multithreading
- Postman-compatible testable gRPC API
- Unit + Load testing support
- GitHub Actions CI for build and test

---

## 📦 Setup

### 1. Clone the Repository

```bash
git clone https://github.com/abdullahmaz/text2video-grpc-docker.git
cd text2video-grpc-docker
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run Locally

```bash
python -m app.nlp
```

- gRPC server starts at `127.0.0.1:50051`
- Gradio UI launches at `http://127.0.0.1:7860`

---

## 🐋 Docker Usage

### Build the Image

```bash
docker build -t text2video-service .
```

### Run the Container

```bash
docker run -p 50051:50051 -p 7860:7860 text2video-service
```

---

## 🧪 Testing

### Unit Tests

```bash
python -m unittest discover tests
```

### Load Testing

```bash
python tests/load_test.py
```

### Postman

- Import `text2video.proto` into Postman
- Use gRPC tab, method: `VideoGenerator.Generate`
- Message input:

```json
{
  "prompt": "A robot teaching in a floating classroom",
  "audio_path": "",
  "filter_option": "Sepia"
}
```

- Test scripts use `pm.response.messages` for validation.

---

## 📤 API Specification

### gRPC Service

**Service:** `VideoGenerator`  
**Method:** `Generate`

#### Request

```protobuf
message VideoResponse {
  string video_path = 1;   // Path to the generated video file
  string message = 2;      // Human-readable status message
  int32 status_code = 3;   // 200 for success, 400/500 for errors
}
```

#### Response

```protobuf
message VideoResponse {
  string video_path = 1;
  string message = 2;
  int32 status_code = 3;
}
```

---

## 📊 Performance Graph

Below is the performance graph showing the average response time against the number of concurrent users:

![performance_graph](https://github.com/user-attachments/assets/ca668311-90a0-4e4e-85c7-a5b9ab0c5505)

---

## 🎥 Sample Video

Here is a sample video generated by the service:

https://github.com/user-attachments/assets/cf5ca219-552a-4573-8d3c-677a09a68c14

---

## 🧱 Architecture Overview

```txt
[ Gradio UI ]         [ Postman ]
       │                   │
       ▼                   ▼
   ┌──────────── gRPC Interface ─────────────┐
   │                                         │
   │       VideoGeneratorServicer            │
   │       ┌──────────────────────────┐      │
   │       │  DiffusionPipeline       │      │
   │       │  (zeroscope_v2_576w)     │      │
   │       └──────────────────────────┘      │
   └─────────────────────────────────────────┘
                     │
                     ▼
                MP4 video file
```

---

## 🧠 Model Source

- **Model**: [`cerspense/zeroscope_v2_576w`](https://huggingface.co/cerspense/zeroscope_v2_576w)
- **Scheduler**: `DPMSolverMultistepScheduler`
- **Framework**: Hugging Face `diffusers`, `torch`, `gradio`

---

## ⚠️ Limitations

- May be slow to start due to model size and video rendering
- GPU recommended for practical response time
- Text prompts may not always generate contextually accurate results
- No prompt history or user management

---

## 👤 Authors

**Abdullah Mazhar**
**Katrina Bodani**
**Haider Niaz**   
[Hugging Face Space](https://huggingface.co/spaces/abdullahmazhar3/text2video-grpc-docker)

---
