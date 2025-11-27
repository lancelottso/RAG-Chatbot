"""Enhanced debug runner for Individual_Project.py

- Detects whether `torch` is installed and prints an actionable message if not.
- Detects the best available device (MPS, CUDA, CPU) and uses it to run a simple tensor op.
- Prints diagnostic details including torch version and device info.
"""

import sys

try:
    import torch
except Exception as e:
    print("PyTorch import failed:", e)
    print("Please activate your virtual environment or install torch. Example:")
    print("  python3 -m pip install --upgrade pip")
    print("  python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu")
    sys.exit(1)

print(f"PyTorch version: {torch.__version__}")

# Detect the best device
device = torch.device('cpu')
if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = torch.device('mps')
elif torch.cuda.is_available():
    device = torch.device('cuda')

print(f"Using device: {device}")

# Create a tensor on the selected device and perform a basic operation
x = torch.rand(2, 3, device=device)
print("Tensor:", x)
print("Tensor device:", x.device)

# Do a trivial operation to ensure the backend runs operations
y = x * 2.0
print("Tensor * 2.0:", y)

# If device isn't CPU, also show where operations are executed
if device.type != 'cpu':
    print(f"Device details: {device}, is_mps:{getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available()}, cuda:{torch.cuda.is_available()}")

print("All OK: brief tutorial & device diagnostics complete.")
