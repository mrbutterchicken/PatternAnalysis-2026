"""
Contains the source code for training, validating, testing, and saving your model.
The model should be imported from modules.py and the data loader should be 
imported from dataset.py. 
Make sure to plot the losses and metrics during training.
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
import modules