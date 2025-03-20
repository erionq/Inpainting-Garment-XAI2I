"""
Usage:

    python shap_i2i.py \
        --config_path ./configs/VITONHD.yaml \
        --model_load_path ./ckpts/VITONHD.ckpt \
        --data_root_dir ./zalando-hd-resized \
        --save_dir ./results_shap \
        --only_one_data True \
        --certain_data_idx 00273_00.jpg \
        --n_segments 30 \
        --n_samples 150 \
        --scorers pixel_l2,clip
"""
import os
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
from utils import tensor2img

try:
    import shap
    _HAS_SHAP = True
except ImportError:
    _HAS_SHAP = False

try:
    import open_clip
    _HAS_OPEN_CLIP = True
except ImportError:
    _HAS_OPEN_CLIP = False

try:
    from skimage.metrics import structural_similarity as ssim
    _HAS_SSIM = True
except ImportError:
    _HAS_SSIM = False

from skimage.segmentation import slic


# --------------------------------------------------------------------------- #
# Scoring functions -- identical to lime_i2i.py, kept in sync deliberately so
# LIME and SHAP results are directly comparable in your thesis's evaluation.
# --------------------------------------------------------------------------- #

class Scorer:
    name = "base"

    def score(self, generated_img, original_generated_img, original_cloth_img):
        raise NotImplementedError


class PixelSimilarityScorer(Scorer):
    name = "pixel_similarity"

    def __init__(self, metric="l2"):
        assert metric in ("l2", "ssim")
        self.metric = metric
        if metric == "ssim" and not _HAS_SSIM:
            raise ImportError("scikit-image structural_similarity not available")

    def score(self, generated_img, original_generated_img, original_cloth_img):
        a = generated_img.astype(np.float32) / 255.0
        b = original_generated_img.astype(np.float32) / 255.0
        if self.metric == "l2":
            dist = float(np.mean((a - b) ** 2))
            return float(np.exp(-dist * 10.0))
        return float(ssim(a, b, channel_axis=2, data_range=1.0))


class CLIPSimilarityScorer(Scorer):
    name = "clip_similarity"

    def __init__(self, model_name="ViT-B-32", pretrained="openai", device="cuda"):
        if not _HAS_OPEN_CLIP:
            raise ImportError("open_clip_torch not installed")
        self.device = device
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.model = self.model.to(device).eval()

    @torch.no_grad()
    def score(self, generated_img, original_generated_img, original_cloth_img):
        from PIL import Image
        img_t = self.preprocess(Image.fromarray(generated_img)).unsqueeze(0).to(self.device)
        ref_t = self.preprocess(Image.fromarray(original_cloth_img)).unsqueeze(0).to(self.device)
        img_feat = self.model.encode_image(img_t)
        ref_feat = self.model.encode_image(ref_t)
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
        ref_feat = ref_feat / ref_feat.norm(dim=-1, keepdim=True)
        return float((img_feat @ ref_feat.T).item())


SCORER_REGISTRY = {
    "pixel_l2": lambda: PixelSimilarityScorer(metric="l2"),
    "pixel_ssim": lambda: PixelSimilarityScorer(metric="ssim"),
    "clip": lambda: CLIPSimilarityScorer(),
}


# --------------------------------------------------------------------------- #
# Superpixel perturbation -- identical to lime_i2i.py
# --------------------------------------------------------------------------- #

def get_superpixels(img_uint8_rgb, n_segments=50, compactness=10):
    return slic(img_uint8_rgb, n_segments=n_segments, compactness=compactness, start_label=0)


def perturb_image(img_uint8_rgb, segments, mask_vector, baseline="mean"):
    out = img_uint8_rgb.copy()
    if baseline == "mean":
        fill = img_uint8_rgb.reshape(-1, 3).mean(axis=0).astype(np.uint8)
    elif baseline == "black":
        fill = np.zeros(3, dtype=np.uint8)
    elif baseline == "blur":
        blurred = cv2.GaussianBlur(img_uint8_rgb, (31, 31), 0)
    else:
        raise ValueError(f"unknown baseline {baseline}")

    for seg_id, keep in enumerate(mask_vector):
        if not keep:
            region = segments == seg_id
            out[region] = blurred[region] if baseline == "blur" else fill
    return out


# --------------------------------------------------------------------------- #
# Model forward pass -- identical to lime_i2i.py
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


# --------------------------------------------------------------------------- #
# KernelSHAP explanation
# --------------------------------------------------------------------------- #

def shap_explain(model, sampler, base_batch, img_H, img_W, denoise_steps,
                  scorer_names, n_segments=50, n_samples=150,
                  baseline="mean", seed=0):
    """
    Returns: dict[scorer_name -> (segments, shap_values)],
             plus the reference cloth image and original generation for plotting.
    """
    if not _HAS_SHAP:
        raise ImportError("shap package not installed -- run: pip install shap")

    np.random.seed(seed)
    scorers = {name: SCORER_REGISTRY[name]() for name in scorer_names}

    cloth_tensor = base_batch["cloth"][0].cpu().numpy()
    cloth_img = np.uint8(((cloth_tensor + 1.0) * 127.5))

    segments = get_superpixels(cloth_img, n_segments=n_segments)
    n_segs = int(segments.max()) + 1

    original_generated = run_inference_on_batch(
        model, sampler, copy.deepcopy(base_batch), denoise_steps, img_H, img_W
    )

    call_counter = {"i": 0}

    def make_predict_fn(scorer_name):
        scorer = scorers[scorer_name]

        def predict_fn(mask_matrix):
            """mask_matrix: [n_coalitions, n_segs] binary array, as passed by shap.
            Returns: [n_coalitions] array of scores."""
            scores = np.zeros(mask_matrix.shape[0], dtype=np.float32)
            for row_idx, mask_vector in enumerate(mask_matrix):
                perturbed = perturb_image(cloth_img, segments, mask_vector, baseline=baseline)
                perturbed_norm = (perturbed.astype(np.float32) / 127.5) - 1.0

                batch_i = copy.deepcopy(base_batch)
                batch_i["cloth"] = torch.from_numpy(perturbed_norm).unsqueeze(0)

                gen_img = run_inference_on_batch(model, sampler, batch_i, denoise_steps, img_H, img_W)
                scores[row_idx] = scorer.score(gen_img, original_generated, cloth_img)

                call_counter["i"] += 1
                print(f"[SHAP:{scorer_name}] forward pass {call_counter['i']} "
                      f"(coalition size {int(mask_vector.sum())}/{n_segs}) -> score {scores[row_idx]:.3f}")
            return scores

        return predict_fn

    # Background: single all-off coalition (standard choice for image SHAP --
    # "removed" superpixels replaced by baseline). Instance to explain: all-on.
    background = np.zeros((1, n_segs))
    instance = np.ones((1, n_segs))

    results = {}
    for name in scorer_names:
        predict_fn = make_predict_fn(name)
        explainer = shap.KernelExplainer(predict_fn, background)
        # nsamples controls how many coalitions KernelSHAP queries;
        # each one costs a full diffusion forward pass, so this is your
        # main compute/quality knob, same role as n_samples in LIME.
        shap_values = explainer.shap_values(instance, nsamples=n_samples, silent=True)
        results[name] = (segments, np.asarray(shap_values).flatten())

    return results, cloth_img, original_generated


def visualize_shap(cloth_img, segments, values, save_path):
    importance = np.zeros(segments.shape, dtype=np.float32)
    for seg_id, val in enumerate(values):
        importance[segments == seg_id] = val
    # SHAP values can be negative (evidence against) as well as positive
    # (evidence for) -- keep sign information rather than min-max normalizing
    # away the zero point, unlike the LIME visualization.
    vmax = np.abs(importance).max() + 1e-8
    importance_norm = (importance / vmax) * 0.5 + 0.5  # map [-vmax,vmax] -> [0,1]

    heat = cv2.applyColorMap(np.uint8(importance_norm * 255), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(cloth_img, 0.6, heat, 0.4, 0)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, overlay[:, :, ::-1])


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
    p.add_argument("--save_dir", type=str, default="./results_shap")
    p.add_argument("--img_H", type=int, default=512)
    p.add_argument("--img_W", type=int, default=384)
    p.add_argument("--denoise_steps", type=int, default=50)
    p.add_argument("--eta", type=float, default=0.0)
    p.add_argument("--only_one_data", type=str2bool, default=True)
    p.add_argument("--certain_data_idx", type=str, default="00273_00.jpg")
    p.add_argument("--unpair", action="store_true")
    p.add_argument("--n_segments", type=int, default=30)
    p.add_argument("--n_samples", type=int, default=150,
                    help="number of coalitions KernelSHAP queries (~= forward passes)")
    p.add_argument("--baseline", type=str, default="mean", choices=["mean", "black", "blur"])
    p.add_argument("--scorers", type=str, default="pixel_l2,clip",
                    help="comma-separated subset of: pixel_l2,pixel_ssim,clip")
    return p.parse_args()


def main():
    args = build_args()
    scorer_names = [s.strip() for s in args.scorers.split(",") if s.strip()]
    for s in scorer_names:
        assert s in SCORER_REGISTRY, f"unknown scorer '{s}', choose from {list(SCORER_REGISTRY)}"

    config = OmegaConf.load(args.config_path)
    config.model.params.img_H = args.img_H
    config.model.params.img_W = args.img_W

    model = create_model(config_path=None, config=config)
    load_cp = torch.load(args.model_load_path, map_location="cpu")
    load_cp = load_cp["state_dict"] if "state_dict" in load_cp.keys() else load_cp
    model.load_state_dict(load_cp)
    model = model.cuda().eval()

    sampler = DDIMSampler(model)

    dataset = getattr(import_module("dataset"), config.dataset_name)(
        data_root_dir=args.data_root_dir,
        img_H=args.img_H,
        img_W=args.img_W,
        is_paired=not args.unpair,
        is_test=True,
        is_sorted=True,
        only_one_data=args.only_one_data,
        certain_data_idx=args.certain_data_idx,
    )
    dataloader = DataLoader(dataset, num_workers=0, shuffle=False, batch_size=1, pin_memory=True)

    os.makedirs(args.save_dir, exist_ok=True)

    for batch_idx, batch in enumerate(dataloader):
        print(f"--- sample {batch_idx} ({batch['img_fn'][0]}) ---")
        results, cloth_img, original_generated = shap_explain(
            model, sampler, batch,
            img_H=args.img_H, img_W=args.img_W, denoise_steps=args.denoise_steps,
            scorer_names=scorer_names,
            n_segments=args.n_segments, n_samples=args.n_samples,
            baseline=args.baseline,
        )

        cv2.imwrite(opj(args.save_dir, f"{batch_idx}_original_generated.jpg"),
                    original_generated[:, :, ::-1])
        cv2.imwrite(opj(args.save_dir, f"{batch_idx}_reference_cloth.jpg"),
                    cloth_img[:, :, ::-1])

        for name, (segments, values) in results.items():
            save_path = opj(args.save_dir, f"{batch_idx}_shap_{name}.jpg")
            visualize_shap(cloth_img, segments, values, save_path)
            print(f"  saved: {save_path}")


if __name__ == "__main__":
    main()
