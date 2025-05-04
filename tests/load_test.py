import grpc
import time
import threading
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import app.text2video_pb2 as pb2
import app.text2video_pb2_grpc as pb2_grpc

def grpc_request(prompt, results, index):
    with grpc.insecure_channel('localhost:50051') as channel:
        stub = pb2_grpc.VideoGeneratorStub(channel)
        start = time.time()
        response = stub.Generate(pb2.TextPrompt(prompt=prompt))
        end = time.time()
        results[index] = {
            'status_code': response.status_code,
            'response_time': end - start
        }

def simulate_load(concurrent_requests=5, prompt="A beach in the metaverse"):
    results = [{} for _ in range(concurrent_requests)]
    threads = []
    for i in range(concurrent_requests):
        t = threading.Thread(target=grpc_request, args=(prompt, results, i))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()
    for i, r in enumerate(results):
        print(f"Request {i+1}: Status={r['status_code']}, Time={r['response_time']:.2f}s")

if __name__ == "__main__":
    for c in [1, 2, 4, 8, 16]:
        print(f"\n--- {c} Concurrent Users ---")
        simulate_load(c)
