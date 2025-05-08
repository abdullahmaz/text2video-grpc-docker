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
import imageio
import imageio.plugins.ffmpeg
import cv2
import whisper
import numpy as np

# Set Hugging Face cache directory
os.environ['HF_HOME'] = os.path.join(os.getcwd(), 'hf_cache')
os.makedirs(os.environ['HF_HOME'], exist_ok=True)
os.environ['GRPC_VERBOSITY'] = 'ERROR' 

# Initialize global variables for models
# These will be loaded on-demand to save resources
pipe = None  # Text-to-video diffusion pipeline
whisper_model = None  # Speech-to-text model

@spaces.GPU(duration=120)  # Allocate GPU for up to 120 seconds
def initialize_pipeline():
    """Initialize the text-to-video and speech-to-text models.
    This function loads both the Whisper speech-to-text model and the
    ZeroScope text-to-video diffusion pipeline.
    """
    global pipe, whisper_model
    print("Initializing pipeline...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    whisper_model = whisper.load_model("base")

    # Load the text-to-video diffusion pipeline
    pipe = DiffusionPipeline.from_pretrained(
        "cerspense/zeroscope_v2_576w",
        torch_dtype=dtype,
        cache_dir=os.environ['HF_HOME'],
        use_safetensors=False
    ).to(device)

    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    # Apply optimizations for GPU usage
    if device == "cuda":
        pipe.enable_model_cpu_offload()
        pipe.enable_vae_slicing()

    print("Pipeline ready.")

# Function to apply filter
def apply_filter(video_path, filter_type):
    """Apply visual filter to a video file.
    
    Args:
        video_path: Path to the input video file
        filter_type: Type of filter to apply ("Grayscale" or "Sepia")
        
    Returns:
        Path to the filtered video file
    """
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_path = os.path.join("videos", f"filtered_{uuid.uuid4()}.mp4")
    fps = cap.get(cv2.CAP_PROP_FPS)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(out_path, fourcc, fps, (W, H))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if filter_type == "Grayscale":
            # Convert to grayscale and back to BGR (required for video writing)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif filter_type == "Sepia":
            # Apply sepia tone matrix transformation
            sepia_filter = np.array([[0.272, 0.534, 0.131],
                                     [0.349, 0.686, 0.168],
                                     [0.393, 0.769, 0.189]])
            frame = cv2.transform(frame, sepia_filter)
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        out.write(frame)

    cap.release()
    out.release()
    return out_path

# Add a semaphore to limit concurrent model access
# This prevents multiple requests from using the GPU simultaneously
model_semaphore = threading.Semaphore(1)  # Only one request can access the model at a time

@spaces.GPU(duration=120)  # Allocate GPU for up to 120 seconds
def generate_video(audio, text, filter_option):
    """Generate video from text prompt or audio transcription.
    
    Args:
        audio: Path to audio file for transcription (optional)
        text: Text prompt for video generation (optional)
        filter_option: Visual filter to apply to the video
        
    Returns:
        Tuple of (video_path, message)
    """
    global pipe, whisper_model
    
    # Use a semaphore to ensure only one request uses the model at a time
    with model_semaphore:
        # Initialize models if not already loaded
        if pipe is None:
            initialize_pipeline()

        # Determine the prompt source (audio transcription or direct text)
        if audio:
            result = whisper_model.transcribe(audio)
            prompt = result["text"]
        elif text:
            prompt = text
        else:
            return None, gr.Warning("Prompt cannot be empty. Please enter a description for your video.")

        prompt = prompt.strip()
        os.makedirs("videos", exist_ok=True)
        output_path = os.path.join("videos", f"{uuid.uuid4()}.mp4")

        try:
            output = pipe(prompt, num_inference_steps=40, height=320, width=576, num_frames=40, output_type="pil")
            frames = output.frames if hasattr(output, "frames") else output.images
            export_to_video(frames[0], output_video_path=output_path)

            # Apply selected filter if requested
            if filter_option != "None":
                output_path = apply_filter(output_path, filter_option)

            return output_path, f"Prompt used: {prompt}"
        except Exception as e:
            return None, f"Error: {str(e)}"

# Interaction with Server
def call_grpc_server(audio, text, filter_option):
    """Call the gRPC server to generate a video.
    This is the function called by the Gradio interface.
    
    Args:
        audio: Path to audio file for transcription
        text: Text prompt for video generation
        filter_option: Visual filter to apply
        
    Returns:
        Tuple of (video_path, status_message)
    """
    # Create an insecure gRPC channel to the server
    with grpc.insecure_channel('localhost:50051') as channel:
        stub = text2video_pb2_grpc.VideoGeneratorStub(channel)

        request = text2video_pb2.VideoRequest(
            prompt=text.strip() if text else "",
            audio_path=audio if audio else "",
            filter_option=filter_option
        )

        response = stub.Generate(request)
        if response.status_code == 200:
            return response.video_path, f"<div>{response.message}</div>"
        else:
            return None, f"<div>{response.message}</div>"

# Server
def serve():
    """Start the gRPC server to handle video generation requests."""
    # Define the service implementation
    class VideoGeneratorServicer(text2video_pb2_grpc.VideoGeneratorServicer):
        def Generate(self, request, context):
            """Handle video generation requests from clients.
            
            Args:
                request: The VideoRequest message
                context: The gRPC context
                
            Returns:
                VideoResponse message with generation results
            """
            # Extract and sanitize fields from the request
            prompt = request.prompt.strip()
            audio_path = request.audio_path.strip()
            filter_option = request.filter_option.strip()
            
            # Log incoming request details
            print('\n')
            print("---- Incoming Request ----")
            print("Prompt:", prompt)
            print("Audio Path:", audio_path)
            print("Filter Option:", filter_option)
            print('\n')

            # Validate the filter option
            valid_filters = {"None", "Grayscale", "Sepia"}
            if filter_option not in valid_filters:
                # Return error response for invalid filter
                return text2video_pb2.VideoResponse(
                    video_path="",
                    message=f"Invalid filter_option '{filter_option}'. Must be one of: {', '.join(valid_filters)}.",
                    status_code=400  # Bad request
                )

            # Ensure at least one form of input is provided
            if not prompt and not audio_path:
                return text2video_pb2.VideoResponse(
                    video_path="",
                    message="Either prompt or audio path must be provided.",
                    status_code=400  # Bad request
                )

            try:
                # Generate the video using the core function
                video_path, _ = generate_video(audio_path, prompt, filter_option)
                # Check if generation was successful
                if video_path and os.path.exists(video_path):
                    # Return success response with video path
                    return text2video_pb2.VideoResponse(
                        video_path=video_path,
                        message="Success",
                        status_code=200  # OK
                    )
                else:
                    # Return error response for failed generation
                    return text2video_pb2.VideoResponse(
                        video_path="",
                        message="Video generation failed.",
                        status_code=500  # Internal server error
                    )
            except Exception as e:
                # Return error response for any exceptions
                return text2video_pb2.VideoResponse(
                    video_path="",
                    message=f"Internal error: {str(e)}",
                    status_code=500  # Internal server error
                )

    # Create a gRPC server with a thread pool
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=8))
    # Register the service implementation
    text2video_pb2_grpc.add_VideoGeneratorServicer_to_server(VideoGeneratorServicer(), server)
    # Listen on port 50051
    server.add_insecure_port('[::]:50051')
    # Start the server
    server.start()
    print("Server running on port 50051...")
    # Keep the server running until terminated
    server.wait_for_termination()

# Gradio web interface configuration
iface = gr.Interface(
    fn=call_grpc_server,
    inputs=[
        gr.Audio(type="filepath", label="🎙 Upload or Record Audio (Optional)"),
        gr.Textbox(label="📝 Enter Text Prompt (Required if no audio)", placeholder="e.g., A knight fighting a dragon in the clouds"),
        gr.Dropdown(["None", "Grayscale", "Sepia"], label="🎨 Choose Video Filter", value="None")
    ],
    outputs=[
        gr.Video(label="Generated Video"),
        gr.HTML(label="Status/Error")
    ],
    title="Audio/Text-to-Video Generator",
    description="Upload audio or enter a text prompt to generate a video. Optionally apply a filter.",
    flagging_mode="never"  # Disable flagging feature
)

# Start the gRPC server in a separate thread
print("Starting server...")
server_thread = threading.Thread(target=serve)
server_thread.daemon = True  # Allow the thread to exit when the main program exits
server_thread.start()

# Launch Gradio Interface
print("Launching Gradio interface without sharing...")
iface.launch(share=False, server_name="0.0.0.0", server_port=7860)
