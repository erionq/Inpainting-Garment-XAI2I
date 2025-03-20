"""Helpers for the Continuity and Contrastivity metrics.
"""
import os
import copy
import random
from os.path import join as opj

import cv2
import numpy as np
import torch

# =========================================================================== #
# Helpers for Continuity / Consistency / Contrastivity (need extra runs)
# =========================================================================== #

def add_gaussian_noise_to_cloth(batch, sigma=0.05):
    batch = copy.deepcopy(batch)
    noise = torch.randn_like(batch["cloth"]) * sigma
    batch["cloth"] = torch.clamp(batch["cloth"] + noise, -1.0, 1.0)
    return batch


def load_random_mismatched_cloth(data_root_dir, data_type, current_cloth_fn, img_H, img_W):
    """Loads a different cloth image from the same folder, for the
    Contrastivity metric. Self-contained (doesn't require a second Dataset
    instance) -- normalizes identically to dataset.py's imread()."""
    cloth_dir = opj(data_root_dir, data_type, "cloth")
    candidates = [f for f in os.listdir(cloth_dir) if f != current_cloth_fn]
    chosen = random.choice(candidates)
    img = cv2.imread(opj(cloth_dir, chosen))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_W, img_H))
    img = (img.astype(np.float32) / 127.5) - 1.0
    return torch.from_numpy(img).unsqueeze(0)
