import torch

def check_pytorch_cuda():
    print("Checking CUDA devices with PyTorch...")
    if torch.cuda.is_available():
        num_devices = torch.cuda.device_count()
        print(f"Number of available CUDA devices: {num_devices}")
        for i in range(num_devices):
            print(f"Device {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("No CUDA devices available with PyTorch.")


if __name__ == "__main__":
    check_pytorch_cuda()