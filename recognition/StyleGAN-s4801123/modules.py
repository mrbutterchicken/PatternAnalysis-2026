"""
Containing the source code of the components of your model. 
Each component must be implemented as a class or a function in PyTorch or TensorFlow/Keras. 
Your implementation of the model should not depend on NumPy (or other Python libraries not part of PyTorch / TensorFlow) in any way unless otherwise approved by the teaching staff.
"""

import torch

if torch.cuda.is_available():
    device = torch.device("cuda")       # NVIDIA GPU
elif torch.backends.mps.is_available():
    device = torch.device("mps")        # Apple Silicon (M1/M2/M3/M4)
else:
    device = torch.device("cpu")        # Fallback Default

print(f"Using device: {device}")

import dataset