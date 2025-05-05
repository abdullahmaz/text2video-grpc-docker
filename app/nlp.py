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

# Set Hugging Face cache directory
os.environ['HF_HOME'] = os.path.join(os.getcwd(), 'hf_cache')
os.makedirs(os.environ['HF_HOME'], exist_ok=True)
os.environ['GRPC_VERBOSITY'] = 'ERROR' 

pipe = None
whisper_model = None

@spaces.GPU(duration=120)  # Allocate GPU for up to 120 seconds
def initialize_pipeline():
    global pipe, whisper_model
    print("Initializing pipeline...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    whisper_model = whisper.load_model("base")

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

# Function to apply filter
def apply_filter(video_path, filter_type):
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
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif filter_type == "Sepia":
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
model_semaphore = threading.Semaphore(1)

@spaces.GPU(duration=120)  # Allocate GPU for up to 120 seconds
def generate_video(audio, text, filter_option):
    global pipe, whisper_model
    
    # Use a semaphore to ensure only one request uses the model at a time
    with model_semaphore:
        if pipe is None:
            initialize_pipeline()

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

            # Apply selected filter
            if filter_option != "None":
                output_path = apply_filter(output_path, filter_option)

            return output_path, f"Prompt used: {prompt}"
        except Exception as e:
            return None, f"Error: {str(e)}"

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

iface = gr.Interface(
    fn=generate_video,
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
    flagging_mode="never"
)

# Start the server in a separate thread
print("Starting server...")
server_thread = threading.Thread(target=serve)
server_thread.daemon = True
server_thread.start()

# Launch Gradio Interface
print("Launching Gradio interface without sharing...")
iface.launch(share=False, server_name="0.0.0.0", server_port=7860)
