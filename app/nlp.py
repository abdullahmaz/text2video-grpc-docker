import os
import gradio as gr
import grpc
import text2video_pb2
import text2video_pb2_grpc
import torch
from concurrent import futures
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.utils import export_to_video
import uuid
import threading

def generate_video(prompt):
    try:
        # Connect to gRPC server
        channel = grpc.insecure_channel('localhost:50051')
        stub = text2video_pb2_grpc.VideoGeneratorStub(channel)

        # Send request
        request = text2video_pb2.TextPrompt(prompt=prompt)
        response = stub.Generate(request)

        # Check status
        if response.status_code == 200 and os.path.exists(response.video_path):
            return response.video_path
        else:
            return f"Error {response.status_code}: {response.message}"
    except Exception as e:
        return f"Exception: {str(e)}"

# Gradio interface
iface = gr.Interface(
    fn=generate_video,
    inputs=gr.Textbox(label="Enter your prompt", placeholder="e.g., A knight fighting a dragon in the clouds"),
    outputs=gr.Video(label="Generated Video"),
    title="Text-to-Video Generator",
    description="Enter a text prompt and generate a video using a gRPC-powered diffusion model.",
)

class VideoGeneratorServicer(text2video_pb2_grpc.VideoGeneratorServicer):
    def __init__(self):
        print("Initializing pipeline...")
        self.pipe = DiffusionPipeline.from_pretrained(
            "cerspense/zeroscope_v2_576w",
            torch_dtype=torch.float16
        )
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.enable_model_cpu_offload()
        self.pipe.enable_vae_slicing()
        print("Pipeline ready.")

    def Generate(self, request, context):
        prompt = request.prompt.strip()
        if not prompt:
            return text2video_pb2.VideoResponse(
                video_path="",
                message="Prompt cannot be empty.",
                status_code=400
            )

        try:
            output = self.pipe(prompt, num_inference_steps=40, height=320, width=576, num_frames=35, output_type="pil")
            video_frames = output.frames[0]

            video_filename = f"{uuid.uuid4()}.mp4"
            video_path = os.path.join("videos", video_filename)
            os.makedirs("videos", exist_ok=True)
            export_to_video(video_frames, output_video_path=video_path)

            return text2video_pb2.VideoResponse(
                video_path=video_path,
                message="Success",
                status_code=200
            )
        except Exception as e:
            return text2video_pb2.VideoResponse(
                video_path="",
                message=f"Internal error: {str(e)}",
                status_code=500
            )

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    text2video_pb2_grpc.add_VideoGeneratorServicer_to_server(VideoGeneratorServicer(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    print("Server running on port 50051...")
    server.wait_for_termination()

print("Starting Gradio interface...")
iface.queue()  # Enable queuing for better handling of multiple requests

# Start the server in a separate thread
print("Starting server and loading model (this may take a while)...")
server_thread = threading.Thread(target=serve)
server_thread.daemon = True
server_thread.start()

# Launch the Gradio interface in the main thread
print("Launching Gradio interface without sharing...")
iface.launch(share=False, server_name="0.0.0.0", server_port=7860)