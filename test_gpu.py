import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    major, minor = torch.cuda.get_device_capability(i)
    print(f"  GPU {i}: {torch.cuda.get_device_name(i)}  (compute capability {major}.{minor})")
    if major < 8:
        print(f"    -> No bf16 hardware support (needs >=8.0). model.py will use float16 "
              f"for the 4-bit compute dtype automatically — this is expected on this GPU.")