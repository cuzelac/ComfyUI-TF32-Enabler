#!/usr/bin/env python3
"""
Test script to verify torch.compile works with CUDA allocator fix
"""

import os
import torch

# Apply the fix
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = "expandable_segments:True"

# Enable TF32
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

print("=" * 60)
print("Testing torch.compile with CUDA allocator fix")
print("=" * 60)

if not torch.cuda.is_available():
    print("❌ CUDA not available")
    exit(1)

print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
print(f"✅ CUDA Allocator: {os.environ.get('PYTORCH_CUDA_ALLOC_CONF')}")
print(f"✅ TF32 Enabled: {torch.backends.cuda.matmul.allow_tf32}")

# Create a simple test model
class SimpleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(128, 128)
    
    def forward(self, x):
        return self.linear(x)

print("\n" + "=" * 60)
print("Compiling test model...")
print("=" * 60)

try:
    model = SimpleModel().cuda()
    model = torch.compile(model, mode="max-autotune")
    
    # Test forward pass
    x = torch.randn(1, 128, device='cuda')
    output = model(x)
    
    print("✅ torch.compile works!")
    print("✅ No cudaMallocAsync error!")
    print("=" * 60)
    
except RuntimeError as e:
    if "cudaMallocAsync" in str(e):
        print(f"❌ Still getting error: {e}")
        print("\nTry setting environment variable before starting Python:")
        print("export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True")
    else:
        print(f"❌ Different error: {e}")
    print("=" * 60)
