import grpc
import time
import threading
import sys
import os
import matplotlib.pyplot as plt
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import app.text2video_pb2 as pb2
import app.text2video_pb2_grpc as pb2_grpc

def grpc_request(prompt, results, index):
    with grpc.insecure_channel('localhost:50051') as channel:
        stub = pb2_grpc.VideoGeneratorStub(channel)
        start = time.time()
        response = stub.Generate(pb2.VideoRequest(prompt=prompt, audio_path="", filter_option="None"))
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
    return results

def plot_performance(concurrent_users, response_times):
    plt.figure(figsize=(10, 5))
    plt.plot(concurrent_users, response_times, marker='o')
    plt.title('Performance Graph')
    plt.xlabel('Number of Concurrent Users')
    plt.ylabel('Average Response Time (s)')
    plt.grid(True)
    plt.savefig('performance_graph.png')

if __name__ == "__main__":
    concurrent_users = [1, 2, 4, 8]
    average_response_times = []

    for c in concurrent_users:
        print(f"\n--- {c} Concurrent Users ---")
        results = simulate_load(c)
        avg_time = sum(r['response_time'] for r in results) / len(results)
        average_response_times.append(avg_time)
        print(f"Average Response Time: {avg_time:.2f}s")

    plot_performance(concurrent_users, average_response_times)