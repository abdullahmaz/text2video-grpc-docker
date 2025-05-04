import spaces
import os
import gradio as gr
import grpc
import sys
sys.path.append('/app')
import app.text2video_pb2 as text2video_pb2
import app.text2video_pb2_grpc as text2video_pb2_grpc
import torch
from concurrent import futures
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.utils import export_to_video
import uuid
import threading

# Set Hugging Face cache directory
os.environ['HF_HOME'] = os.path.join(os.getcwd(), 'hf_cache')
os.makedirs(os.environ['HF_HOME'], exist_ok=True)
os.environ['GRPC_VERBOSITY'] = 'ERROR' 

pipe = None

@spaces.GPU(duration=120)  # Allocate GPU for up to 120 seconds
def initialize_pipeline():
    global pipe
    print("Initializing pipeline...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    pipe = DiffusionPipeline.from_pretrained(
        "cerspense/zeroscope_v2_576w",
        torch_dtype=dtype,
        cache_dir=os.environ['HF_HOME'],
        use_safetensors=False
    ).to(device)

    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    if device == "cuda":
        pipe.enable_model_cpu_offload()
        pipe.enable_vae_slicing()

    print("Pipeline ready.")

@spaces.GPU(duration=120)  # Allocate GPU for up to 120 seconds
def generate_video(prompt):
    global pipe
    
    if pipe is None:
        initialize_pipeline()

    prompt = prompt.strip()
    if not prompt:
        # Return a tuple with error flag and message for Gradio to display
        return None, gr.Warning("Prompt cannot be empty. Please enter a description for your video.")

    try:
        output = pipe(prompt, num_inference_steps=40, height=320, width=576, num_frames=40, output_type="pil")
        video_frames = output.frames[0]

        video_filename = f"{uuid.uuid4()}.mp4"
        video_path = os.path.join("videos", video_filename)
        os.makedirs("videos", exist_ok=True)
        export_to_video(video_frames, output_video_path=video_path)

        # Return the video path and no warning
        return video_path, None
    except Exception as e:
        error_message = f"Internal error: {str(e)}"
        # Return None for video and error message
        return None, gr.Error(error_message)

iface = gr.Interface(
    fn=generate_video,
    inputs=gr.Textbox(label="Enter your prompt", placeholder="e.g., A knight fighting a dragon in the clouds"),
    outputs=[
        gr.Video(label="Generated Video"),
        "html"
    ],
    title="Text-to-Video Generator",
    description="Enter a text prompt and generate a video using a gRPC-powered diffusion model.",
)

def serve():
    class VideoGeneratorServicer(text2video_pb2_grpc.VideoGeneratorServicer):
        def Generate(self, request, context):
            prompt = request.prompt.strip()
            if not prompt:
                return text2video_pb2.VideoResponse(
                    video_path="",
                    message="Prompt cannot be empty.",
                    status_code=400
                )

            try:
                video_path, _ = generate_video(prompt)
                if video_path and os.path.exists(video_path):
                    return text2video_pb2.VideoResponse(
                        video_path=video_path,
                        message="Success",
                        status_code=200
                    )
                else:
                    return text2video_pb2.VideoResponse(
                        video_path="",
                        message="Video generation failed.",
                        status_code=500
                    )
            except Exception as e:
                return text2video_pb2.VideoResponse(
                    video_path="",
                    message=f"Internal error: {str(e)}",
                    status_code=500
                )

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    text2video_pb2_grpc.add_VideoGeneratorServicer_to_server(VideoGeneratorServicer(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    print("Server running on port 50051...")
    server.wait_for_termination()

print("Starting Gradio interface...")
iface.queue()

# Start the server in a separate thread
print("Starting server...")
server_thread = threading.Thread(target=serve)
server_thread.daemon = True
server_thread.start()

# Launch the Gradio interface in the main thread
print("Launching Gradio interface without sharing...")
iface.launch(share=False, server_name="0.0.0.0", server_port=7860)
