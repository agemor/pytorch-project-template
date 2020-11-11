import os
import random
import pathlib

import torch
import numpy as np

# Get project ./data folder path
def get_data_path(*paths):
    base_path = os.path.join(os.path.dirname(__file__), 'data')
    path = os.path.join(base_path, *paths)

    # Ensure the target folder exists
    pathlib.Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)

    return path

# Make PyTorch deterministic for better reproducibility
def set_deterministic(seed=0):
    torch.set_deterministic(True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
