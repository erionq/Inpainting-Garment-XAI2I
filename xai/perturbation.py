"""Superpixel segmentation, perturbation, and mask helpers.
"""
import cv2
import numpy as np
import torch
from skimage.segmentation import slic

# =========================================================================== #
# Superpixel perturbation utilities (shared by LIME / SHAP / RISE / faithfulness)
# =========================================================================== #

def get_superpixels(img_uint8_rgb, n_segments=20, compactness=10, region_mask=None):
    """
    region_mask: optional [H,W] array in {0,1}. When given, superpixels are
    generated ONLY inside it and background pixels get label -1, so they are
    never perturbed.
    """
    if region_mask is not None:
        seg = slic(img_uint8_rgb, n_segments=n_segments, compactness=compactness,
                   mask=region_mask.astype(bool), start_label=1)
        return seg.astype(np.int32) - 1  # background 0 -> -1; regions 1..n -> 0..n-1
    return slic(img_uint8_rgb, n_segments=n_segments, compactness=compactness, start_label=0)


def perturb_image_segments(img_uint8_rgb, segments, mask_vector, baseline="mean"):
    out = img_uint8_rgb.copy()
    if baseline == "mean":
        fill = img_uint8_rgb.reshape(-1, 3).mean(axis=0).astype(np.uint8)
    elif baseline == "black":
        fill = np.zeros(3, dtype=np.uint8)
    else:
        raise ValueError(f"unknown baseline {baseline}")
    for seg_id, keep in enumerate(mask_vector):
        if not keep:
            out[segments == seg_id] = fill
    return out


def get_garment_mask(batch):
    """Cloth silhouette from batch['cloth_mask'] ([1,H,W,1], 1 = garment).
    Used to keep superpixels inside the garment."""
    m = batch["cloth_mask"]
    m = m.detach().cpu().numpy() if torch.is_tensor(m) else np.asarray(m)
    return (m.squeeze() > 0.5).astype(np.uint8)


def get_inpaint_region_mask(batch):
    m = batch["agn_mask"]
    m = m.detach().cpu().numpy() if torch.is_tensor(m) else np.asarray(m)
    m = m.squeeze()
    return (1.0 - m > 0.5).astype(np.uint8)


def map_to_cloth_resolution(map_arr, target_hw):
    arr = map_arr.astype(np.float32)
    return cv2.resize(arr, (target_hw[1], target_hw[0]), interpolation=cv2.INTER_CUBIC)
