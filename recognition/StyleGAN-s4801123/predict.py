"""
Showing example usage of your **trained** model. It should load your saved model
weights, run inference on test cases, print out any results, and provide 
visualisations where applicable (e.g., prediction overlays, generated samples, 
heatmaps). NumPy is allowed in this script for visualization or loading data.
"""

import torch

if torch.cuda.is_available():
    device = torch.device("cuda")       # NVIDIA GPU
elif torch.backends.mps.is_available():
    device = torch.device("mps")        # Apple Silicon (M1/M2/M3/M4)
else:
    device = torch.device("cpu")        # Fallback Default

print(f"Using device: {device}")

import numpy as np
from matplotlib import pyplot as plt

import dataset
import modules
import train