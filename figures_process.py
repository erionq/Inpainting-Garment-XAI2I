"""
Usage:

    python figures_process.py \
        --config_path ./configs/VITONHD.yaml \
        --model_load_path ./ckpts/VITONHD.ckpt \
        --data_root_dir ./zalando-hd-resized \
        --save_dir ./thesis_figures \
        --certain_data_idx 00273_00.jpg \
        --method i2am \
        --figures faithfulness,continuity,contrastivity
"""
import os
import copy
import argparse
from os.path import join as opj
from importlib import import_module

import cv2
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from omegaconf import OmegaConf

from cldm.model import create_model
from ldm.models.diffusion.ddim import DDIMSampler
from hook import CrossAttentionHook

import xai_eval as X


# --------------------------------------------------------------------------- #
# Figure labels
# --------------------------------------------------------------------------- #

LABELS = {
    "removed": "removed",
    "step": "step",
    "pos_title": "Positive perturbation (most important superpixels removed first)",
    "neg_title": "Negative perturbation (least important superpixels removed first)",
    "curve_x": "fraction of superpixels removed",
    "curve_y": "score",
    "pos_curve": "positive (most important first)",
    "neg_curve": "negative (least important first)",
    "orig_cloth": "original cloth",
    "noised_cloth": "noised cloth",
    "attr_orig": "attribution map (original)",
    "attr_noised": "attribution map (noised)",
    "paired_cloth": "correct cloth (paired)",
    "mismatched_cloth": "swapped cloth",
    "attr_paired": "attribution map (correct)",
    "attr_mismatched": "attribution map (swapped)",
}


def overlay_heatmap(cloth_img, map_arr):
    """Same visualization convention as xai_eval.py's per-method output, so the
    figures in the thesis match the heatmaps saved during evaluation."""
    arr_norm = X._normalize01(map_arr)
    heat_small = cv2.resize(arr_norm.astype(np.float32), (cloth_img.shape[1], cloth_img.shape[0]))
    heat = cv2.applyColorMap(np.uint8(heat_small * 255), cv2.COLORMAP_JET)
    return cv2.addWeighted(cloth_img, 0.6, heat, 0.4, 0)


# --------------------------------------------------------------------------- #
# Figure 1: Faithfulness -- perturbation sequence + score curves
# --------------------------------------------------------------------------- #

def figure_faithfulness(model, sampler, hook, batch, explainer, scorer, args, L):
    cloth_tensor = batch["cloth"][0].cpu().numpy()
    cloth_img = np.uint8(((cloth_tensor + 1.0) * 127.5))

    garment = X.get_garment_mask(batch) if args.mask_segments else None
    segments = X.get_superpixels(cloth_img, n_segments=args.n_segments, region_mask=garment)
    n_segs = int(segments.max()) + 1

    result = explainer.explain(model, sampler, hook, batch, scorer, args)
    map_arr = result["map"]

    map_resized = X.map_to_cloth_resolution(map_arr, cloth_img.shape[:2])
    seg_importance = np.array([map_resized[segments == s].mean() if (segments == s).any() else 0.0
                               for s in range(n_segs)])
    order_most_first = np.argsort(seg_importance)[::-1]
    order_least_first = order_most_first[::-1]

    original_generated = X.run_inference_on_batch(
        model, sampler, copy.deepcopy(batch), args.denoise_steps, args.img_H, args.img_W,
        fixed_noise=args.fixed_noise)

    region_mask = getattr(args, "score_region_mask", None)

    def run_curve(order, keep_snapshots):
        keep_mask = np.ones(n_segs, dtype=np.int32)
        curve = [scorer.score(original_generated, original_generated, region_mask=region_mask)]
        snapshots = {0: cloth_img.copy()}
        for step, seg_id in enumerate(order, 1):
            keep_mask[seg_id] = 0
            perturbed = X.perturb_image_segments(cloth_img, segments, keep_mask, baseline=args.baseline)
            perturbed_norm = (perturbed.astype(np.float32) / 127.5) - 1.0
            batch_i = copy.deepcopy(batch)
            batch_i["cloth"] = torch.from_numpy(perturbed_norm).unsqueeze(0)
            gen_img = X.run_inference_on_batch(
                model, sampler, batch_i, args.denoise_steps, args.img_H, args.img_W,
                fixed_noise=args.fixed_noise)
            curve.append(scorer.score(gen_img, original_generated, region_mask=region_mask))
            if step in keep_snapshots:
                snapshots[step] = perturbed.copy()
            print(f"  [faithfulness] step {step}/{n_segs} score={curve[-1]:.4f}")
        return curve, snapshots

    n_snap = min(args.n_snapshots, n_segs + 1)
    snap_steps = set(np.linspace(0, n_segs, n_snap).astype(int).tolist())

    print(" positive perturbation ...")
    pos_curve, pos_snaps = run_curve(order_most_first, snap_steps)
    print(" negative perturbation ...")
    neg_curve, neg_snaps = run_curve(order_least_first, snap_steps)

    pos_auc = float(np.trapz(pos_curve) / (len(pos_curve) - 1))
    neg_auc = float(np.trapz(neg_curve) / (len(neg_curve) - 1))

    sorted_steps = sorted(snap_steps)
    ncols = len(sorted_steps)
    fig = plt.figure(figsize=(2.0 * ncols, 8.2))
    gs = fig.add_gridspec(3, ncols, height_ratios=[1.35, 1.35, 1.5], hspace=0.35, wspace=0.08)

    for row, (snaps, title) in enumerate([(pos_snaps, L["pos_title"]), (neg_snaps, L["neg_title"])]):
        for col, step in enumerate(sorted_steps):
            ax = fig.add_subplot(gs[row, col])
            ax.imshow(snaps[step])
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"{step}/{n_segs} {L['removed']}", fontsize=8)
            if col == 0:
                ax.set_ylabel(title, fontsize=8, wrap=True)

    ax = fig.add_subplot(gs[2, :])
    xs = np.arange(len(pos_curve)) / (len(pos_curve) - 1)
    ax.plot(xs, pos_curve, marker="o", ms=3, label=f"{L['pos_curve']} (AUC={pos_auc:.3f})")
    ax.plot(xs, neg_curve, marker="s", ms=3, label=f"{L['neg_curve']} (AUC={neg_auc:.3f})")
    ax.set_xlabel(L["curve_x"]); ax.set_ylabel(L["curve_y"])
    ax.grid(alpha=0.3); ax.legend(fontsize=9)

    out = opj(args.save_dir, f"faithfulness_{explainer.name}.png")
    fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out}  (pos_auc={pos_auc:.4f}, neg_auc={neg_auc:.4f})")

    np.savetxt(opj(args.save_dir, f"faithfulness_{explainer.name}_curves.csv"),
               np.column_stack([xs, pos_curve, neg_curve]),
               delimiter=",", header="fraction_removed,positive,negative", comments="")
    return {"positive_auc": pos_auc, "negative_auc": neg_auc}


# --------------------------------------------------------------------------- #
# Figures 2 & 3: Continuity and Contrastivity -- 2x2 comparison panels
# --------------------------------------------------------------------------- #

def _comparison_figure(cloth_a, map_a, cloth_b, map_b, labels, metric_line, out_path, dpi):
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 9.0))
    panels = [
        (axes[0, 0], cloth_a, labels[0]),
        (axes[0, 1], cloth_b, labels[1]),
        (axes[1, 0], overlay_heatmap(cloth_a, map_a), labels[2]),
        (axes[1, 1], overlay_heatmap(cloth_b, map_b), labels[3]),
    ]
    for ax, img, title in panels:
        ax.imshow(img)
        ax.set_title(title, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(metric_line, fontsize=11, y=0.985)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out_path}")


def figure_continuity(model, sampler, hook, batch, explainer, scorer, args, L):
    res_orig = explainer.explain(model, sampler, hook, batch, scorer, args)
    noisy_batch = X.add_gaussian_noise_to_cloth(batch, sigma=args.continuity_sigma)
    res_noisy = explainer.explain(model, sampler, hook, noisy_batch, scorer, args)

    m = X.metric_continuity(res_orig["map"], res_noisy["map"])
    noisy_cloth = np.uint8(((noisy_batch["cloth"][0].cpu().numpy() + 1.0) * 127.5))

    _comparison_figure(
        res_orig["cloth_img"], res_orig["map"], noisy_cloth, res_noisy["map"],
        [L["orig_cloth"], f"{L['noised_cloth']} (sigma={args.continuity_sigma})",
         L["attr_orig"], L["attr_noised"]],
        f"Continuity: cosine={m['continuity_cosine']:.3f}, pearson={m['continuity_pearson']:.3f}",
        opj(args.save_dir, f"continuity_{explainer.name}.png"), args.dpi)
    return m


def figure_contrastivity(model, sampler, hook, batch, explainer, scorer, args, L):
    res_paired = explainer.explain(model, sampler, hook, batch, scorer, args)

    mismatched_cloth = X.load_random_mismatched_cloth(
        args.data_root_dir, "test", batch["cloth_fn"][0], args.img_H, args.img_W)
    mismatched_batch = copy.deepcopy(batch)
    mismatched_batch["cloth"] = mismatched_cloth
    res_mismatched = explainer.explain(model, sampler, hook, mismatched_batch, scorer, args)

    m = X.metric_contrastivity(res_paired["map"], res_mismatched["map"])
    mismatched_img = np.uint8(((mismatched_cloth[0].cpu().numpy() + 1.0) * 127.5))

    _comparison_figure(
        res_paired["cloth_img"], res_paired["map"], mismatched_img, res_mismatched["map"],
        [L["paired_cloth"], L["mismatched_cloth"], L["attr_paired"], L["attr_mismatched"]],
        f"Contrastivity: dissimilarity={m['contrastivity_dissimilarity']:.3f}",
        opj(args.save_dir, f"contrastivity_{explainer.name}.png"), args.dpi)
    return m


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def build_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config_path", type=str, required=True)
    p.add_argument("--model_load_path", type=str, required=True)
    p.add_argument("--data_root_dir", type=str, required=True)
    p.add_argument("--save_dir", type=str, default="./thesis_figures")
    p.add_argument("--certain_data_idx", type=str, default="00273_00.jpg")
    p.add_argument("--unpair", action="store_true")

    p.add_argument("--method", type=str, default="i2am",
                   help="one of: " + ",".join(X.EXPLAINER_REGISTRY))
    p.add_argument("--figures", type=str, default="faithfulness,continuity,contrastivity")

    p.add_argument("--img_H", type=int, default=512)
    p.add_argument("--img_W", type=int, default=384)
    p.add_argument("--denoise_steps", type=int, default=50)
    p.add_argument("--eta", type=float, default=0.0)
    p.add_argument("--generated_image", type=X.str2bool, default=True)
    p.add_argument("--controlnet", type=X.str2bool, default=False)

    p.add_argument("--n_segments", type=int, default=12)
    p.add_argument("--n_perturb_samples", type=int, default=150)
    p.add_argument("--rise_grid_size", type=int, default=8)
    p.add_argument("--rise_keep_prob", type=float, default=0.5)
    p.add_argument("--ig_steps", type=int, default=20)
    p.add_argument("--baseline", type=str, default="mean", choices=["mean", "black"])
    p.add_argument("--scorer", type=str, default="pixel_l2", choices=list(X.SCORER_REGISTRY))
    p.add_argument("--mask_scorer", type=X.str2bool, default=True)
    p.add_argument("--mask_segments", type=X.str2bool, default=True)
    p.add_argument("--noise_seed", type=int, default=0)
    p.add_argument("--continuity_sigma", type=float, default=0.05)

    p.add_argument("--n_snapshots", type=int, default=6,
                   help="how many perturbation steps to show in the faithfulness figure")
    p.add_argument("--dpi", type=int, default=300)
    return p.parse_args()


def main():
    args = build_args()
    L = LABELS
    figures = [f.strip() for f in args.figures.split(",") if f.strip()]
    assert args.method in X.EXPLAINER_REGISTRY, f"unknown method '{args.method}'"

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
        per_attention_head=False, controlnet=args.controlnet, new_attention_map=False)
    hook.take_module(sampler.model.model.diffusion_model)

    scorer = X.SCORER_REGISTRY[args.scorer]()
    explainer = X.EXPLAINER_REGISTRY[args.method]()

    dataset = getattr(import_module("dataset"), config.dataset_name)(
        data_root_dir=args.data_root_dir, img_H=args.img_H, img_W=args.img_W,
        is_paired=not args.unpair, is_test=True, is_sorted=True,
        only_one_data=True, certain_data_idx=args.certain_data_idx)
    dataloader = DataLoader(dataset, num_workers=0, shuffle=False, batch_size=1, pin_memory=True)

    os.makedirs(args.save_dir, exist_ok=True)

    for batch_idx, batch in enumerate(dataloader):
        # same seeding convention as xai_eval.py, so figures correspond to the
        # numbers in the results tables
        args.fixed_noise = X.make_fixed_noise(args.img_H, args.img_W, seed=args.noise_seed + batch_idx)
        args.score_region_mask = X.get_inpaint_region_mask(batch) if args.mask_scorer else None

        print(f"=== {batch['img_fn'][0]} -- method {args.method} ===")
        if "faithfulness" in figures:
            figure_faithfulness(model, sampler, hook, batch, explainer, scorer, args, L)
        if "continuity" in figures:
            figure_continuity(model, sampler, hook, batch, explainer, scorer, args, L)
        if "contrastivity" in figures:
            figure_contrastivity(model, sampler, hook, batch, explainer, scorer, args, L)


if __name__ == "__main__":
    main()
