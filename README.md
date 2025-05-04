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

# Text2Video gRPC Docker Space

This Hugging Face Space uses a gRPC backend to generate videos from text prompts using the `zeroscope_v2_576w` diffusion model.

- Built with Python, Gradio, Diffusers, and gRPC
- Fully Dockerized deployment
- Server and frontend are launched from one container