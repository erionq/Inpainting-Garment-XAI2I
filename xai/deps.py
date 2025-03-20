"""Shared imports and optional-dependency flags.

"""
import os
import csv
import copy
import random
import argparse
from os.path import join as opj
from importlib import import_module

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from skimage.segmentation import slic

from cldm.model import create_model
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.modules.attention import SpatialTransformer
from cldm.warping_cldm_network import CustomSpatialTransformer
from hook import CrossAttentionHook
from utils import tensor2img

try:
    from sklearn.metrics import average_precision_score
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False
    average_precision_score = None  # noqa: F811 (name must exist for `from .deps import`)

try:
    import shap as shap_lib
    _HAS_SHAP = True
except ImportError:
    _HAS_SHAP = False
    shap_lib = None  # noqa: F811 (name must exist for `from .deps import`)

try:
    import open_clip
    _HAS_OPEN_CLIP = True
except ImportError:
    _HAS_OPEN_CLIP = False
    open_clip = None  # noqa: F811 (name must exist for `from .deps import`)
