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

# Text-to-Video Generator 🎥

This project implements a Text-to-Video generation microservice using a gRPC backend, powered by the `zeroscope_v2_576w` diffusion model.

---

## 🚀 Features

- Generate videos from text prompts using Diffusers
- gRPC API with status-aware responses
- Fully containerized (Docker)
- Gradio-based minimal frontend
- Concurrent request handling
- Postman-compatible API testing

---

## 📦 Setup

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/text2video-grpc-docker.git
cd text2video-grpc-docker