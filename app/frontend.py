# frontend.py

import gradio as gr
import grpc
import text2video_pb2
import text2video_pb2_grpc
import os

def generate_video(prompt):
    try:
        channel = grpc.insecure_channel('localhost:50051')
        stub = text2video_pb2_grpc.VideoGeneratorStub(channel)

        request = text2video_pb2.TextPrompt(prompt=prompt)
        response = stub.Generate(request)

        if response.status_code == 200 and os.path.exists(response.video_path):
            return response.video_path
        else:
            print(f"Error {response.status_code}: {response.message}")
            return None
    except Exception as e:
        print(f"Exception: {str(e)}")
        return None

def gradio_interface(prompt):
    path = generate_video(prompt)
    if path:
        return path, ""
    else:
        return None, "Failed to generate video. Please check your input or try again later."

iface = gr.Interface(
    fn=gradio_interface,
    inputs=gr.Textbox(label="Enter your prompt"),
    outputs=[
        gr.Video(label="Generated Video"),
        gr.Textbox(label="Status", interactive=False)
    ],
    title="Text-to-Video Generator",
    description="Generate a video using a gRPC-powered diffusion model."
)

if __name__ == "__main__":
    iface.launch(share=True)
