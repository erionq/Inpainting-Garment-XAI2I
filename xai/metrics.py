"""Quantus-style and Co-12 evaluation metrics."""
import copy

import cv2
import numpy as np
import torch

from .deps import _HAS_SKLEARN, average_precision_score
from .perturbation import get_superpixels, perturb_image_segments, map_to_cloth_resolution
from .forward import run_inference_on_batch

# =========================================================================== #
# Metrics
# =========================================================================== #

def _normalize01(arr):
    arr = arr.astype(np.float64)
    return (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)


def metric_localization(map_arr, gt_mask_tensor, threshold=0.5):
    """Quantus-style Localization: IoU + Average Precision."""
    arr_norm = _normalize01(map_arr)
    gt = gt_mask_tensor.detach().cpu().numpy() if torch.is_tensor(gt_mask_tensor) else np.asarray(gt_mask_tensor)
    gt = gt.squeeze()
    pred = map_to_cloth_resolution(arr_norm, gt.shape)
    pred_binary = pred >= threshold
    gt_binary = gt > 0.5

    intersection = np.logical_and(pred_binary, gt_binary).sum()
    union = np.logical_or(pred_binary, gt_binary).sum()
    iou = float(intersection / (union + 1e-8))
    ap = float(average_precision_score(gt_binary.flatten().astype(int), pred.flatten())) if _HAS_SKLEARN else None
    return {"localization_iou": iou, "localization_ap": ap}


def metric_completeness(map_arr, gt_mask_tensor, threshold=0.5):
    """Co-12 Completeness: recall of the ground-truth region"""
    arr_norm = _normalize01(map_arr)
    gt = gt_mask_tensor.detach().cpu().numpy() if torch.is_tensor(gt_mask_tensor) else np.asarray(gt_mask_tensor)
    gt = gt.squeeze()
    pred = map_to_cloth_resolution(arr_norm, gt.shape)
    pred_binary = pred >= threshold
    gt_binary = gt > 0.5

    true_positive = np.logical_and(pred_binary, gt_binary).sum()
    recall = float(true_positive / (gt_binary.sum() + 1e-8))
    return {"completeness_recall": recall}


def metric_complexity(map_arr, sparsity_threshold=0.1):
    """Quantus-style Complexity: sparsity + normalized entropy."""
    arr_norm = _normalize01(map_arr)
    sparsity = float(np.mean(arr_norm > sparsity_threshold))

    p = np.clip(arr_norm.flatten(), 1e-8, None)
    p = p / p.sum()
    H = -np.sum(p * np.log(p))
    H_norm = float(H / np.log(len(p)))
    return {"complexity_sparsity": sparsity, "complexity_normalized_entropy": H_norm}


def metric_compactness(map_arr, segments, mass_fraction=0.9):
    """Co-12 Compactness: fraction of superpixels needed to reach
    `mass_fraction` of the total attribution mass."""
    map_resized = map_to_cloth_resolution(map_arr, segments.shape)
    map_resized = np.clip(map_resized, 0, None)
    n_segs = int(segments.max()) + 1
    seg_mass = np.array([map_resized[segments == s].sum() for s in range(n_segs)])
    order = np.argsort(seg_mass)[::-1]
    cum = np.cumsum(seg_mass[order])
    total = cum[-1] if cum[-1] > 0 else 1e-8
    n_needed = int(np.searchsorted(cum, mass_fraction * total) + 1)
    return {"compactness_fraction_segments": float(n_needed / n_segs)}


def metric_continuity(map1, map2):
    """Co-12 Continuity (same operationalization as objective_metrics.py's
    'Robustness'): similarity of explanation under a small input perturbation."""
    a = map1.astype(np.float64).flatten()
    b = map2.astype(np.float64) if map2.shape == map1.shape else map_to_cloth_resolution(map2, map1.shape)
    b = b.astype(np.float64).flatten()
    cosine = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))
    pearson = float(np.corrcoef(a, b)[0, 1])
    return {"continuity_cosine": cosine, "continuity_pearson": pearson}


def metric_consistency(map1, map2):
    """Co-12 Consistency: same input, same explanation, run twice.
    Meaningful mainly for stochastic explainers (LIME/SHAP/RISE);
    deterministic explainers (I2AM/GradCAM/IG)"""
    a = map1.astype(np.float64).flatten()
    b = map2.astype(np.float64) if map2.shape == map1.shape else map_to_cloth_resolution(map2, map1.shape)
    b = b.astype(np.float64).flatten()
    cosine = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))
    return {"consistency_cosine": cosine}


def metric_contrastivity(map_paired, map_mismatched):
    """Co-12 Contrastivity: dissimilarity between the explanation for the
    correct (paired) reference vs. a randomly swapped (unpaired) reference."""
    a = map_paired.astype(np.float64).flatten()
    b = map_mismatched.astype(np.float64) if map_mismatched.shape == map_paired.shape else \
        map_to_cloth_resolution(map_mismatched, map_paired.shape)
    b = b.astype(np.float64).flatten()
    cosine = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))
    return {"contrastivity_dissimilarity": 1.0 - cosine}


def metric_faithfulness(model, sampler, batch, map_arr, scorer, args):
    """Quantus-style Faithfulness: positive/negative perturbation AUC,
    ranking superpixels by the explanation's own importance values."""
    cloth_tensor = batch["cloth"][0].cpu().numpy()
    cloth_img = np.uint8(((cloth_tensor + 1.0) * 127.5))
    segments = get_superpixels(cloth_img, n_segments=args.n_segments, region_mask=get_garment_mask(batch) if args.mask_segments else None)
    n_segs = int(segments.max()) + 1

    map_resized = map_to_cloth_resolution(map_arr, cloth_img.shape[:2])
    seg_importance = np.array([map_resized[segments == s].mean() for s in range(n_segs)])
    order_most_first = np.argsort(seg_importance)[::-1]
    order_least_first = order_most_first[::-1]

    original_generated = run_inference_on_batch(model, sampler, copy.deepcopy(batch), args.denoise_steps, args.img_H, args.img_W, fixed_noise=getattr(args, 'fixed_noise', None))

    def run_curve(order):
        keep_mask = np.ones(n_segs, dtype=np.int32)
        curve = [scorer.score(original_generated, original_generated)]
        for seg_id in order:
            keep_mask[seg_id] = 0
            perturbed = perturb_image_segments(cloth_img, segments, keep_mask, baseline=args.baseline)
            perturbed_norm = (perturbed.astype(np.float32) / 127.5) - 1.0
            batch_i = copy.deepcopy(batch)
            batch_i["cloth"] = torch.from_numpy(perturbed_norm).unsqueeze(0)
            gen_img = run_inference_on_batch(model, sampler, batch_i, args.denoise_steps, args.img_H, args.img_W, fixed_noise=getattr(args, 'fixed_noise', None))
            curve.append(scorer.score(gen_img, original_generated, region_mask=getattr(args, 'score_region_mask', None)))
        return curve

    pos_curve = run_curve(order_most_first)
    neg_curve = run_curve(order_least_first)
    pos_auc = float(np.trapz(pos_curve) / (len(pos_curve) - 1))
    neg_auc = float(np.trapz(neg_curve) / (len(neg_curve) - 1))
    return {"faithfulness_positive_auc": pos_auc, "faithfulness_negative_auc": neg_auc}
