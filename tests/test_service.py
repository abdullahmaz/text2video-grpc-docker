import grpc
import unittest
import app.text2video_pb2 as pb2
import app.text2video_pb2_grpc as pb2_grpc

class TestText2VideoService(unittest.TestCase):
    def setUp(self):
        self.channel = grpc.insecure_channel('localhost:50051')
        self.stub = pb2_grpc.VideoGeneratorStub(self.channel)

    def test_generate_success(self):
        request = pb2.VideoRequest(prompt="A futuristic city under the sea", audio_path="", filter_option="None")
        response = self.stub.Generate(request)
        self.assertEqual(response.status_code, 200)
        self.assertIn(".mp4", response.video_path)

    def test_generate_empty_prompt(self):
        request = pb2.VideoRequest(prompt="", audio_path="", filter_option="None")
        response = self.stub.Generate(request)
        self.assertEqual(response.status_code, 400)
        
    def test_generate_with_filter(self):
        request = pb2.VideoRequest(prompt="A forest in autumn", audio_path="", filter_option="Sepia")
        response = self.stub.Generate(request)
        self.assertEqual(response.status_code, 200)
        self.assertIn(".mp4", response.video_path)
        
    def test_generate_invalid_filter(self):
        request = pb2.VideoRequest(prompt="A beach at sunset", audio_path="", filter_option="InvalidFilter")
        response = self.stub.Generate(request)
        self.assertEqual(response.status_code, 400)
        
    def test_generate_very_long_prompt(self):
        long_prompt = "A " + "very " * 100 + "long prompt"
        request = pb2.VideoRequest(prompt=long_prompt, audio_path="", filter_option="None")
        response = self.stub.Generate(request)
        # Depending on your implementation, this might succeed or return an error
        # Adjust the assertion based on your service's expected behavior
        self.assertIn(response.status_code, [200, 400])
        
    def test_generate_special_characters(self):
        request = pb2.VideoRequest(prompt="Special characters: !@#$%^&*()", audio_path="", filter_option="None")
        response = self.stub.Generate(request)
        self.assertEqual(response.status_code, 200)
        self.assertIn(".mp4", response.video_path)
        
    def test_timeout_handling(self):
        # Test with a timeout to ensure the service handles timeouts gracefully
        try:
            channel = grpc.insecure_channel('localhost:50051')
            stub = pb2_grpc.VideoGeneratorStub(channel)
            request = pb2.VideoRequest(prompt="A complex scene that takes time", audio_path="", filter_option="None")
            # Set a very short timeout to force a timeout error
            response = stub.Generate(request, timeout=0.001)
            self.fail("Expected timeout exception was not raised")
        except grpc.RpcError as e:
            # Just verify that we can catch the timeout exception
            pass

if __name__ == "__main__":
    unittest.main()