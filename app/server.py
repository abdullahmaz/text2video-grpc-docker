# server.py

import grpc
import text2video_pb2
import text2video_pb2_grpc
import os
import torch
import uuid
from concurrent import futures
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.utils import export_to_video

class VideoGeneratorServicer(text2video_pb2_grpc.VideoGeneratorServicer):
    def __init__(self):
        print("Initializing pipeline...")
        self.pipe = DiffusionPipeline.from_pretrained(
            "cerspense/zeroscope_v2_576w",
            torch_dtype=torch.float16
            cache_dir="/app/cache"
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

            os.makedirs("videos", exist_ok=True)
            video_filename = f"{uuid.uuid4()}.mp4"
            video_path = os.path.join("videos", video_filename)
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
    print("gRPC Server running on port 50051...")
    server.wait_for_termination()

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    serve()