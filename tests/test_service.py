import grpc
import unittest
import app.text2video_pb2 as pb2
import app.text2video_pb2_grpc as pb2_grpc

class TestText2VideoService(unittest.TestCase):
    def setUp(self):
        self.channel = grpc.insecure_channel('localhost:50051')
        self.stub = pb2_grpc.VideoGeneratorStub(self.channel)

    def test_generate_success(self):
        prompt = pb2.TextPrompt(prompt="A futuristic city under the sea")
        response = self.stub.Generate(prompt)
        self.assertEqual(response.status_code, 200)
        self.assertIn(".mp4", response.video_path)

    def test_generate_empty_prompt(self):
        prompt = pb2.TextPrompt(prompt="")
        response = self.stub.Generate(prompt)
        self.assertEqual(response.status_code, 400)

if __name__ == "__main__":
    unittest.main()