"""
Usage:

    python objective_metrics.py \
        --config_path ./configs/VITONHD.yaml \
        --model_load_path ./ckpts/VITONHD.ckpt \
        --data_root_dir ./zalando-hd-resized \
        --save_dir ./results_objective_metrics \
        --only_one_data True \
        --certain_data_idx 00273_00.jpg \
        --generated_image True \
        --controlnet False \
        --n_segments 30 \
        --scorer pixel_l2
"""
import os
import csv
import copy
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
from ldm.modules.attention import CrossAttention, SpatialTransformer, BasicTransformerBlock, MemoryEfficientCrossAttention
from cldm.warping_cldm_network import CustomBasicTransformerBlock, CustomSpatialTransformer
from hook import CrossAttentionHook
from utils import tensor2img

try:
    from sklearn.metrics import average_precision_score
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False

try:
    import open_clip
    _HAS_OPEN_CLIP = True
except ImportError:
    _HAS_OPEN_CLIP = False


# --------------------------------------------------------------------------- #
# Scoring functions for the faithfulness perturbation curves (kept consistent
# with lime_i2i.py / shap_i2i.py so results are comparable across methods)
# --------------------------------------------------------------------------- #

class PixelSimilarityScorer:
    name = "pixel_l2"

    def score(self, generated_img, original_generated_img):
        a = generated_img.astype(np.float32) / 255.0
        b = original_generated_img.astype(np.float32) / 255.0
        dist = float(np.mean((a - b) ** 2))
        return float(np.exp(-dist * 10.0))


class CLIPSimilarityScorer:
    name = "clip"

    def __init__(self, model_name="ViT-B-32", pretrained="openai", device="cuda"):
        if not _HAS_OPEN_CLIP:
            raise ImportError("open_clip_torch not installed")
        self.device = device
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        self.model = self.model.to(device).eval()

    @torch.no_grad()
    def score(self, generated_img, original_generated_img):
        from PIL import Image
        a_t = self.preprocess(Image.fromarray(generated_img)).unsqueeze(0).to(self.device)
        b_t = self.preprocess(Image.fromarray(original_generated_img)).unsqueeze(0).to(self.device)
        a_f = self.model.encode_image(a_t)
        b_f = self.model.encode_image(b_t)
        a_f = a_f / a_f.norm(dim=-1, keepdim=True)
        b_f = b_f / b_f.norm(dim=-1, keepdim=True)
        return float((a_f @ b_f.T).item())


SCORER_REGISTRY = {
    "pixel_l2": lambda: PixelSimilarityScorer(),
    "clip": lambda: CLIPSimilarityScorer(),
}


# --------------------------------------------------------------------------- #
# Superpixel perturbation (identical convention to lime_i2i.py / shap_i2i.py)
# --------------------------------------------------------------------------- #

def get_superpixels(img_uint8_rgb, n_segments=30, compactness=10):
    return slic(img_uint8_rgb, n_segments=n_segments, compactness=compactness, start_label=0)


def perturb_image(img_uint8_rgb, segments, mask_vector, baseline="mean"):
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


# --------------------------------------------------------------------------- #
# Model forward pass (mirrors inference.py's main loop)
# --------------------------------------------------------------------------- #

@torch.no_grad()
def run_inference_on_batch(model, sampler, batch, denoise_steps, img_H, img_W, eta=0.0):
    batch = {k: (v.cuda() if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}

    z, c = model.get_input(batch, "image")
    bs = z.shape[0]
    c_crossattn = c["c_crossattn"][0][:bs]
    if c_crossattn.ndim == 4:
        c_crossattn = model.get_learned_conditioning(c_crossattn)
        c["c_crossattn"] = [c_crossattn]
    uc_cross = model.get_unconditional_conditioning(bs)
    uc_full = {"c_concat": c["c_concat"], "c_crossattn": [uc_cross]}
    uc_full["first_stage_cond"] = c["first_stage_cond"]

    sampler.model.batch = batch

    ts = torch.full((1,), 999, device=z.device, dtype=torch.long)
    start_code = model.q_sample(z, ts)

    shape = (4, img_H // 8, img_W // 8)
    samples, _, _ = sampler.sample(
        denoise_steps, bs, shape, c,
        x_T=start_code, verbose=False, eta=eta,
        unconditional_conditioning=uc_full,
    )
    x_samples = model.decode_first_stage(samples)
    return tensor2img(x_samples[0])


@torch.no_grad()
def run_inference_and_get_attention(model, sampler, hook, batch, denoise_steps, img_H, img_W, eta=0.0):
    """Same as run_inference_on_batch, but also returns the I2AM attention
    map produced during this forward pass (hook must already be attached
    via hook.take_module(...) before calling)."""
    gen_img = run_inference_on_batch(model, sampler, batch, denoise_steps, img_H, img_W, eta)
    agn_mask = batch["agn_mask"].cuda() if isinstance(batch["agn_mask"], torch.Tensor) else batch["agn_mask"]
    attention_map = hook.make_attention_maps(agn_mask)  # also clears hook state internally
    return gen_img, attention_map


# --------------------------------------------------------------------------- #
# Complexity (Eq. 26 + normalized entropy) -- free, no extra forward passes
# --------------------------------------------------------------------------- #

def complexity_metrics(attention_map, sparsity_threshold=0.1):
    """attention_map: 2D tensor/array of raw (non-thresholded) attention values."""
    arr = attention_map.detach().cpu().numpy() if torch.is_tensor(attention_map) else np.asarray(attention_map)
    arr = arr.astype(np.float64)
    arr_norm = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)

    # sparsity: fraction of elements above threshold -- proxy for the paper's
    # L0-norm-based complexity metric (Eq. 26), adapted for continuous maps
    # where an exact zero count is meaningless.
    sparsity = float(np.mean(arr_norm > sparsity_threshold))

    # normalized Shannon entropy: treat the map as a probability distribution
    # over spatial positions. Lower entropy = more concentrated/simple
    # explanation; higher entropy = more diffuse/complex explanation.
    p = arr_norm.flatten()
    p = np.clip(p, 1e-8, None)
    p = p / p.sum()
    H = -np.sum(p * np.log(p))
    H_norm = float(H / np.log(len(p)))

    return {"sparsity": sparsity, "normalized_entropy": H_norm}


# --------------------------------------------------------------------------- #
# Localization (Eq. 24) -- free, reuses the attention map + your existing masks
# --------------------------------------------------------------------------- #

def localization_metrics(attention_map, gt_mask_tensor, threshold=0.5):
    """
    attention_map: 2D tensor/array, raw I2AM attention values (small resolution).
    gt_mask_tensor: ground-truth mask tensor as used elsewhere in this repo
        (e.g. batch['agn_mask'] or batch['cloth_mask']), shape [1,H,W,1],
        values in {0,1} (mind hook.py's mask-inversion conventions -- pass
        whichever mask/orientation you use as ground truth for your target).
    """
    arr = attention_map.detach().cpu().numpy() if torch.is_tensor(attention_map) else np.asarray(attention_map)
    arr = arr.astype(np.float64)
    arr_norm = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)

    gt = gt_mask_tensor.detach().cpu().numpy() if torch.is_tensor(gt_mask_tensor) else np.asarray(gt_mask_tensor)
    gt = gt.squeeze()  # -> [H, W]

    pred_resized = cv2.resize(arr_norm.astype(np.float32), (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_CUBIC)
    pred_binary = pred_resized >= threshold
    gt_binary = gt > 0.5

    intersection = np.logical_and(pred_binary, gt_binary).sum()
    union = np.logical_or(pred_binary, gt_binary).sum()
    iou = float(intersection / (union + 1e-8))

    ap = None
    if _HAS_SKLEARN:
        ap = float(average_precision_score(gt_binary.flatten().astype(int), pred_resized.flatten()))

    return {"iou": iou, "average_precision": ap}


# --------------------------------------------------------------------------- #
# Robustness (Eq. 22) -- 1 extra forward pass per sample
# --------------------------------------------------------------------------- #

def robustness_metrics(attention_map_original, attention_map_perturbed):
    a = attention_map_original.detach().cpu().numpy() if torch.is_tensor(attention_map_original) else np.asarray(attention_map_original)
    b = attention_map_perturbed.detach().cpu().numpy() if torch.is_tensor(attention_map_perturbed) else np.asarray(attention_map_perturbed)

    if a.shape != b.shape:
        b = cv2.resize(b.astype(np.float32), (a.shape[1], a.shape[0]), interpolation=cv2.INTER_CUBIC)

    a_flat, b_flat = a.flatten(), b.flatten()
    cosine = float(np.dot(a_flat, b_flat) / (np.linalg.norm(a_flat) * np.linalg.norm(b_flat) + 1e-8))
    pearson = float(np.corrcoef(a_flat, b_flat)[0, 1])
    return {"cosine_similarity": cosine, "pearson_correlation": pearson}


def add_gaussian_noise_to_cloth(batch, sigma=0.05):
    """Perturbs batch['cloth'] (range [-1,1]) with small Gaussian noise,
    used as the small-perturbation delta (delta) for the robustness metric."""
    batch = copy.deepcopy(batch)
    cloth = batch["cloth"]
    noise = torch.randn_like(cloth) * sigma
    batch["cloth"] = torch.clamp(cloth + noise, -1.0, 1.0)
    return batch


# --------------------------------------------------------------------------- #
# Faithfulness (Eq. 17) -- 2 * n_segments extra forward passes per sample
# --------------------------------------------------------------------------- #

def faithfulness_metrics(model, sampler, hook, base_batch, attention_map,
                          scorer, denoise_steps, img_H, img_W,
                          n_segments=30, baseline="mean"):
    """
    Positive/negative perturbation AUC, following the protocol used by
    Chefer et al. (2021) and Huang et al. (2023) as described in the source
    paper: rank input superpixels by attention importance, progressively
    mask them out (most-important-first for positive, least-important-first
    for negative), and track how a scoring function degrades.

    A well-behaved (faithful) explanation should show a SHARP DROP under
    positive perturbation and STABLE scores under negative perturbation.
    """
    cloth_tensor = base_batch["cloth"][0].cpu().numpy()
    cloth_img = np.uint8(((cloth_tensor + 1.0) * 127.5))

    segments = get_superpixels(cloth_img, n_segments=n_segments)
    n_segs = int(segments.max()) + 1

    attn_arr = attention_map.detach().cpu().numpy() if torch.is_tensor(attention_map) else np.asarray(attention_map)
    attn_resized = cv2.resize(attn_arr.astype(np.float32), (cloth_img.shape[1], cloth_img.shape[0]),
                               interpolation=cv2.INTER_CUBIC)

    seg_importance = np.array([attn_resized[segments == seg_id].mean() for seg_id in range(n_segs)])
    order_most_first = np.argsort(seg_importance)[::-1]
    order_least_first = order_most_first[::-1]

    original_generated = run_inference_on_batch(model, sampler, copy.deepcopy(base_batch), denoise_steps, img_H, img_W)

    def run_curve(order, curve_name):
        keep_mask = np.ones(n_segs, dtype=np.int32)
        curve = [scorer.score(original_generated, original_generated)]  # step 0: nothing removed
        for step, seg_id in enumerate(order, 1):
            keep_mask[seg_id] = 0
            perturbed = perturb_image(cloth_img, segments, keep_mask, baseline=baseline)
            perturbed_norm = (perturbed.astype(np.float32) / 127.5) - 1.0

            batch_i = copy.deepcopy(base_batch)
            batch_i["cloth"] = torch.from_numpy(perturbed_norm).unsqueeze(0)

            gen_img = run_inference_on_batch(model, sampler, batch_i, denoise_steps, img_H, img_W)
            s = scorer.score(gen_img, original_generated)
            curve.append(s)
            print(f"  [faithfulness:{curve_name}] step {step}/{n_segs} -> score {s:.4f}")
        return curve

    pos_curve = run_curve(order_most_first, "positive")
    neg_curve = run_curve(order_least_first, "negative")

    pos_auc = float(np.trapz(pos_curve) / (len(pos_curve) - 1))
    neg_auc = float(np.trapz(neg_curve) / (len(neg_curve) - 1))

    return {
        "positive_perturbation_auc": pos_auc,
        "negative_perturbation_auc": neg_auc,
        "positive_curve": pos_curve,
        "negative_curve": neg_curve,
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def build_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config_path", type=str, required=True)
    p.add_argument("--model_load_path", type=str, required=True)
    p.add_argument("--data_root_dir", type=str, required=True)
    p.add_argument("--save_dir", type=str, default="./results_objective_metrics")
    p.add_argument("--img_H", type=int, default=512)
    p.add_argument("--img_W", type=int, default=384)
    p.add_argument("--denoise_steps", type=int, default=50)
    p.add_argument("--eta", type=float, default=0.0)
    p.add_argument("--only_one_data", type=str2bool, default=True)
    p.add_argument("--certain_data_idx", type=str, default="00273_00.jpg")
    p.add_argument("--unpair", action="store_true")

    # I2AM hook configuration -- match the settings you're evaluating
    p.add_argument("--generated_image", type=str2bool, default=True)
    p.add_argument("--controlnet", type=str2bool, default=False)

    # metric-specific settings
    p.add_argument("--n_segments", type=int, default=30, help="for faithfulness")
    p.add_argument("--baseline", type=str, default="mean", choices=["mean", "black"])
    p.add_argument("--scorer", type=str, default="pixel_l2", choices=list(SCORER_REGISTRY))
    p.add_argument("--localization_threshold", type=float, default=0.5)
    p.add_argument("--complexity_threshold", type=float, default=0.1)
    p.add_argument("--robustness_sigma", type=float, default=0.05)
    p.add_argument("--skip_faithfulness", action="store_true", help="skip the expensive 2*n_segments-pass metric")
    p.add_argument("--skip_robustness", action="store_true", help="skip the +1-pass metric")
    return p.parse_args()


def main():
    args = build_args()

    config = OmegaConf.load(args.config_path)
    config.model.params.img_H = args.img_H
    config.model.params.img_W = args.img_W

    model = create_model(config_path=None, config=config)
    load_cp = torch.load(args.model_load_path, map_location="cpu")
    load_cp = load_cp["state_dict"] if "state_dict" in load_cp.keys() else load_cp
    model.load_state_dict(load_cp)
    model = model.cuda().eval()

    sampler = DDIMSampler(model)

    hook = CrossAttentionHook(
        generated_image=args.generated_image, per_time_step=False,
        per_attention_head=False, controlnet=args.controlnet, new_attention_map=False,
    )
    hook.take_module(sampler.model.model.diffusion_model)

    scorer = SCORER_REGISTRY[args.scorer]()

    dataset = getattr(import_module("dataset"), config.dataset_name)(
        data_root_dir=args.data_root_dir,
        img_H=args.img_H, img_W=args.img_W,
        is_paired=not args.unpair, is_test=True, is_sorted=True,
        only_one_data=args.only_one_data, certain_data_idx=args.certain_data_idx,
    )
    dataloader = DataLoader(dataset, num_workers=0, shuffle=False, batch_size=1, pin_memory=True)

    os.makedirs(args.save_dir, exist_ok=True)
    csv_path = opj(args.save_dir, "objective_metrics.csv")
    fieldnames = [
        "img_fn", "cloth_fn",
        "localization_iou", "localization_ap",
        "complexity_sparsity", "complexity_normalized_entropy",
        "robustness_cosine", "robustness_pearson",
        "faithfulness_positive_auc", "faithfulness_negative_auc",
    ]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for batch_idx, batch in enumerate(dataloader):
            print(f"--- sample {batch_idx} ({batch['img_fn'][0]}) ---")
            row = {"img_fn": batch["img_fn"][0], "cloth_fn": batch["cloth_fn"][0]}

            gen_img, attention_map = run_inference_and_get_attention(
                model, sampler, hook, copy.deepcopy(batch), args.denoise_steps, args.img_H, args.img_W, args.eta
            )

            # --- Localization ---
            # ground-truth choice mirrors hook.py's own cal_IMACS convention:
            # controlnet mode explains cloth->generated, so cloth_mask is the
            # relevant ground truth; otherwise agn_mask (the inpainting region).
            gt_mask = batch["cloth_mask"] if args.controlnet else batch["agn_mask"]
            loc = localization_metrics(attention_map, gt_mask, threshold=args.localization_threshold)
            row["localization_iou"] = loc["iou"]
            row["localization_ap"] = loc["average_precision"]

            # --- Complexity ---
            comp = complexity_metrics(attention_map, sparsity_threshold=args.complexity_threshold)
            row["complexity_sparsity"] = comp["sparsity"]
            row["complexity_normalized_entropy"] = comp["normalized_entropy"]

            # --- Robustness ---
            if not args.skip_robustness:
                perturbed_batch = add_gaussian_noise_to_cloth(batch, sigma=args.robustness_sigma)
                _, attention_map_perturbed = run_inference_and_get_attention(
                    model, sampler, hook, perturbed_batch, args.denoise_steps, args.img_H, args.img_W, args.eta
                )
                rob = robustness_metrics(attention_map, attention_map_perturbed)
                row["robustness_cosine"] = rob["cosine_similarity"]
                row["robustness_pearson"] = rob["pearson_correlation"]

            # --- Faithfulness ---
            if not args.skip_faithfulness:
                faith = faithfulness_metrics(
                    model, sampler, hook, batch, attention_map, scorer,
                    args.denoise_steps, args.img_H, args.img_W,
                    n_segments=args.n_segments, baseline=args.baseline,
                )
                row["faithfulness_positive_auc"] = faith["positive_perturbation_auc"]
                row["faithfulness_negative_auc"] = faith["negative_perturbation_auc"]

            writer.writerow(row)
            f.flush()
            print(f"  {row}")

    print(f"\nSaved: {csv_path}")


if __name__ == "__main__":
    main()
