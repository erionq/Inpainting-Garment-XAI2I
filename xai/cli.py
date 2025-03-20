"""Argument parsing and the evaluation driver.

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
from torch.utils.data import DataLoader
from omegaconf import OmegaConf

from cldm.model import create_model
from ldm.models.diffusion.ddim import DDIMSampler
from hook import CrossAttentionHook

from .scorers import SCORER_REGISTRY
from .explainers import EXPLAINER_REGISTRY
from .perturbation import get_superpixels, get_garment_mask, get_inpaint_region_mask
from .forward import make_fixed_noise
from .metrics import (
    _normalize01,
    metric_localization, metric_completeness, metric_complexity,
    metric_compactness, metric_continuity, metric_consistency,
    metric_contrastivity, metric_faithfulness,
)
from .sampling_helpers import add_gaussian_noise_to_cloth, load_random_mismatched_cloth

# =========================================================================== #
# CLI
# =========================================================================== #

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
    p.add_argument("--save_dir", type=str, default="./results_xai_eval")
    p.add_argument("--img_H", type=int, default=512)
    p.add_argument("--img_W", type=int, default=384)
    p.add_argument("--denoise_steps", type=int, default=50)
    p.add_argument("--eta", type=float, default=0.0)
    p.add_argument("--only_one_data", type=str2bool, default=True)
    p.add_argument("--certain_data_idx", type=str, default="00273_00.jpg")
    p.add_argument("--unpair", action="store_true")

    p.add_argument("--generated_image", type=str2bool, default=True)
    p.add_argument("--controlnet", type=str2bool, default=False)

    p.add_argument("--methods", type=str, default="i2am,lime,shap,rise,gradcam,ig",
                    help="comma-separated subset of: " + ",".join(EXPLAINER_REGISTRY))
    p.add_argument("--scorer", type=str, default="pixel_l2", choices=list(SCORER_REGISTRY))
    p.add_argument("--baseline", type=str, default="mean", choices=["mean", "black"])

    p.add_argument("--n_segments", type=int, default=20, help="for LIME/SHAP/faithfulness/compactness")
    p.add_argument("--n_perturb_samples", type=int, default=60, help="for LIME/SHAP/RISE")
    p.add_argument("--rise_grid_size", type=int, default=8)
    p.add_argument("--rise_keep_prob", type=float, default=0.5)
    p.add_argument("--ig_steps", type=int, default=20)

    p.add_argument("--localization_threshold", type=float, default=0.5)
    p.add_argument("--complexity_threshold", type=float, default=0.1)
    p.add_argument("--continuity_sigma", type=float, default=0.05)

    p.add_argument("--mask_scorer", type=str2bool, default=True,
                    help="restrict pixel_l2 to the inpainted region (no effect on --scorer clip)")
    p.add_argument("--mask_segments", type=str2bool, default=True,
                    help="restrict superpixels/RISE masks to the garment silhouette")
    p.add_argument("--noise_seed", type=int, default=0,
                    help="seed for the per-image fixed x_T noise; keep constant across a comparison")
    p.add_argument("--skip_faithfulness", action="store_true")
    p.add_argument("--skip_consistency", action="store_true")
    p.add_argument("--skip_contrastivity", action="store_true")
    p.add_argument("--skip_continuity", action="store_true")
    return p.parse_args()


def main():
    args = build_args()
    np.random.seed(args.noise_seed)
    method_names = [m.strip() for m in args.methods.split(",") if m.strip()]
    for m in method_names:
        assert m in EXPLAINER_REGISTRY, f"unknown method '{m}', choose from {list(EXPLAINER_REGISTRY)}"

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
    csv_path = opj(args.save_dir, "xai_eval.csv")
    fieldnames = [
        "img_fn", "cloth_fn", "method", "n_segments", "n_perturb_samples",
        "localization_iou", "localization_ap", "completeness_recall",
        "complexity_sparsity", "complexity_normalized_entropy", "compactness_fraction_segments",
        "continuity_cosine", "continuity_pearson",
        "consistency_cosine",
        "contrastivity_dissimilarity",
        "faithfulness_positive_auc", "faithfulness_negative_auc",
    ]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for batch_idx, batch in enumerate(dataloader):
            gt_mask = batch["cloth_mask"] if args.controlnet else batch["agn_mask"]

            # One noise draw per IMAGE, shared by every method and every
            # perturbation of that image. Without this, q_sample() would draw
            # fresh noise per call and the perturbation signal would compete
            # with sampler randomness (see run_inference_on_batch's docstring).
            args.fixed_noise = make_fixed_noise(args.img_H, args.img_W, seed=args.noise_seed + batch_idx)

            # Restrict pixel scoring to the repainted region (no-op for --scorer clip).
            args.score_region_mask = get_inpaint_region_mask(batch) if args.mask_scorer else None

            for method_name in method_names:
                print(f"=== sample {batch_idx} ({batch['img_fn'][0]}) -- method: {method_name} ===")
                explainer = EXPLAINER_REGISTRY[method_name]()
                row = {
                    "img_fn": batch["img_fn"][0], "cloth_fn": batch["cloth_fn"][0], "method": method_name,
                    "n_segments": args.n_segments, "n_perturb_samples": args.n_perturb_samples,
                }

                result = explainer.explain(model, sampler, hook, batch, scorer, args)
                map_arr = result["map"]

                loc = metric_localization(map_arr, gt_mask, threshold=args.localization_threshold)
                comp = metric_completeness(map_arr, gt_mask, threshold=args.localization_threshold)
                cplx = metric_complexity(map_arr, sparsity_threshold=args.complexity_threshold)
                segments_for_compactness = get_superpixels(result["cloth_img"], n_segments=args.n_segments)
                compact = metric_compactness(map_arr, segments_for_compactness)
                row.update(loc)
                row.update(comp)
                row.update(cplx)
                row.update(compact)

                if not args.skip_continuity:
                    noisy_batch = add_gaussian_noise_to_cloth(batch, sigma=args.continuity_sigma)
                    result_noisy = explainer.explain(model, sampler, hook, noisy_batch, scorer, args)
                    row.update(metric_continuity(map_arr, result_noisy["map"]))

                if not args.skip_consistency:
                    result_rerun = explainer.explain(model, sampler, hook, batch, scorer, args)
                    row.update(metric_consistency(map_arr, result_rerun["map"]))

                if not args.skip_contrastivity:
                    data_type = "test"
                    mismatched_cloth = load_random_mismatched_cloth(
                        args.data_root_dir, data_type, batch["cloth_fn"][0], args.img_H, args.img_W
                    )
                    mismatched_batch = copy.deepcopy(batch)
                    mismatched_batch["cloth"] = mismatched_cloth
                    result_mismatched = explainer.explain(model, sampler, hook, mismatched_batch, scorer, args)
                    row.update(metric_contrastivity(map_arr, result_mismatched["map"]))

                if not args.skip_faithfulness:
                    row.update(metric_faithfulness(model, sampler, batch, map_arr, scorer, args))

                writer.writerow(row)
                f.flush()
                print(f"  {row}")

                # save a heatmap visualization per method
                cloth_img = result["cloth_img"]
                arr_norm = _normalize01(map_arr)
                heat_small = cv2.resize(arr_norm.astype(np.float32), (cloth_img.shape[1], cloth_img.shape[0]))
                heat = cv2.applyColorMap(np.uint8(heat_small * 255), cv2.COLORMAP_JET)
                overlay = cv2.addWeighted(cloth_img, 0.6, heat, 0.4, 0)
                save_path = opj(args.save_dir, f"{batch_idx}_{method_name}.jpg")
                cv2.imwrite(save_path, overlay[:, :, ::-1])

    print(f"\nSaved: {csv_path}")


if __name__ == "__main__":
    main()
