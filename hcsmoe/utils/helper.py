import json
import torch.cuda as cuda


def print_gpu_memory():
    for i in range(cuda.device_count()):
        print(f"Device {i}: {cuda.memory_allocated(i) / 1024**2:.2f} MB")

def save_json(data, filename):
    with open(filename, 'w') as fp:
        json.dump(data, fp, sort_keys=True, indent=4)
