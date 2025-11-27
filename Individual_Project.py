import torch

x = torch.rand(2, 3)
print("Tensor:", x)

if torch.backends.mps.is_available():
    print("MPS is available!")