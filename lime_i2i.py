"""
Usage (single sample, three scorers at once):

    python lime_i2i.py \
        --config_path ./configs/VITONHD.yaml \
        --model_load_path ./ckpts/VITONHD.ckpt \
        --data_root_dir ~/zalando-hd-resized \
        --save_dir ./results_lime \
        --only_one_data True \
        --certain_data_idx 00273_00.jpg \
        --n_segments 50 \
        --n_samples 150
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
from skimage.segmentation import slic
from sklearn.linear_model import Ridge

from cldm.model import create_model
from ldm.models.diffusion.ddim import DDIMSampler
from utils import tensor2img

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


# --------------------------------------------------------------------------- #
# Scoring functions (pluggable — run several per perturbation, one fwd pass)
# --------------------------------------------------------------------------- #

class LIMEScorer:
    """Base class. score() must return a float; higher = more similar/faithful
    to the reference behavior being explained."""
    name = "base"

    def score(self, generated_img, original_generated_img, original_cloth_img):
        raise NotImplementedError


class PixelSimilarityScorer(LIMEScorer):
    """How close the perturbed-input generation stays to the ORIGINAL
    (unperturbed) generation. High score = this superpixel subset was
    enough to reproduce the original output."""
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
        else:
            return float(ssim(a, b, channel_axis=2, data_range=1.0))


class CLIPSimilarityScorer(LIMEScorer):
    """CLIP similarity between the generated output and the ORIGINAL
    (unperturbed) cloth reference. High score = this superpixel subset
    was enough to keep the garment identity recognizable."""
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
# Superpixel perturbation
# --------------------------------------------------------------------------- #

def get_superpixels(img_uint8_rgb, n_segments=50, compactness=10):
    """img_uint8_rgb: HWC uint8 in [0,255]. Returns int label map [H,W]."""
    return slic(img_uint8_rgb, n_segments=n_segments, compactness=compactness, start_label=0)


def perturb_image(img_uint8_rgb, segments, mask_vector, baseline="mean"):
    """mask_vector: binary array, length = n_segments. 1 = keep original pixels."""
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
# Model forward pass (mirrors inference.py's main loop, batch size 1)
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
    return tensor2img(x_samples[0])  # uint8 HWC RGB


# --------------------------------------------------------------------------- #
# Core LIME loop
# --------------------------------------------------------------------------- #

def lime_explain(model, sampler, base_batch, img_H, img_W, denoise_steps,
                  scorer_names, n_segments=50, n_samples=150,
                  kernel_width=0.25, baseline="mean", seed=0):
    """
    base_batch: single-item collated batch dict from the DataLoader.
    Returns: dict[scorer_name -> (segments, ridge_coefficients)],
             plus the reference cloth image and original generation for plotting.
    """
    rng = np.random.RandomState(seed)
    scorers = {name: SCORER_REGISTRY[name]() for name in scorer_names}

    cloth_tensor = base_batch["cloth"][0].cpu().numpy()  # [-1,1] HWC
    cloth_img = np.uint8(((cloth_tensor + 1.0) * 127.5))

    segments = get_superpixels(cloth_img, n_segments=n_segments)
    n_segs = int(segments.max()) + 1

    original_generated = run_inference_on_batch(
        model, sampler, copy.deepcopy(base_batch), denoise_steps, img_H, img_W
    )

    masks = rng.randint(0, 2, size=(n_samples, n_segs))
    masks[0, :] = 1  # unperturbed reference sample

    all_scores = {name: [] for name in scorer_names}

    for i in range(n_samples):
        perturbed = perturb_image(cloth_img, segments, masks[i], baseline=baseline)
        perturbed_norm = (perturbed.astype(np.float32) / 127.5) - 1.0

        batch_i = copy.deepcopy(base_batch)
        batch_i["cloth"] = torch.from_numpy(perturbed_norm).unsqueeze(0)

        gen_img = run_inference_on_batch(model, sampler, batch_i, denoise_steps, img_H, img_W)

        for name in scorer_names:
            s = scorers[name].score(gen_img, original_generated, cloth_img)
            all_scores[name].append(s)

        print(f"[LIME] sample {i + 1}/{n_samples} scores: "
              f"{{{', '.join(f'{n}={all_scores[n][-1]:.3f}' for n in scorer_names)}}}")

    distances = np.linalg.norm(masks - masks[0], axis=1) / np.sqrt(n_segs)
    weights = np.exp(-(distances ** 2) / (kernel_width ** 2))

    results = {}
    for name in scorer_names:
        y = np.array(all_scores[name])
        surrogate = Ridge(alpha=1.0)
        surrogate.fit(masks, y, sample_weight=weights)
        results[name] = (segments, surrogate.coef_)

    return results, cloth_img, original_generated


def visualize_lime(cloth_img, segments, coefficients, save_path):
    importance = np.zeros(segments.shape, dtype=np.float32)
    for seg_id, coef in enumerate(coefficients):
        importance[segments == seg_id] = coef
    importance = (importance - importance.min()) / (importance.max() - importance.min() + 1e-8)

    heat = cv2.applyColorMap(np.uint8(importance * 255), cv2.COLORMAP_JET)
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
    p.add_argument("--save_dir", type=str, default="./results_lime")
    p.add_argument("--img_H", type=int, default=512)
    p.add_argument("--img_W", type=int, default=384)
    p.add_argument("--denoise_steps", type=int, default=50)
    p.add_argument("--eta", type=float, default=0.0)
    p.add_argument("--only_one_data", type=str2bool, default=True)
    p.add_argument("--certain_data_idx", type=str, default="00273_00.jpg")
    p.add_argument("--unpair", action="store_true")
    p.add_argument("--n_segments", type=int, default=50)
    p.add_argument("--n_samples", type=int, default=150)
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
        results, cloth_img, original_generated = lime_explain(
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

        for name, (segments, coefs) in results.items():
            save_path = opj(args.save_dir, f"{batch_idx}_lime_{name}.jpg")
            visualize_lime(cloth_img, segments, coefs, save_path)
            print(f"  saved: {save_path}")


if __name__ == "__main__":
    main()
