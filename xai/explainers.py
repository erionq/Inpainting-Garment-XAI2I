"""The six explanation methods and their registry.
"""
import copy

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from ldm.modules.attention import SpatialTransformer
from cldm.warping_cldm_network import CustomSpatialTransformer

from .deps import _HAS_SHAP, shap_lib
from .perturbation import get_superpixels, perturb_image_segments, get_garment_mask
from .forward import run_inference_on_batch, run_inference_and_get_i2am_attention
from .gradient_path import (
    single_step_noise_prediction,
    set_gradient_checkpointing,
    restore_gradient_checkpointing,
)

# =========================================================================== #
# Explainer implementations -- unified interface:
#   explain(model, sampler, hook, batch, scorer, args) -> dict with keys
#   "map" (2D numpy array, arbitrary resolution), "gen_img", "cloth_img"
# =========================================================================== #

class I2AMExplainer:
    name = "i2am"
    stochastic = False

    def explain(self, model, sampler, hook, batch, scorer, args):
        cloth_tensor = batch["cloth"][0].cpu().numpy()
        cloth_img = np.uint8(((cloth_tensor + 1.0) * 127.5))
        gen_img, attn = run_inference_and_get_i2am_attention(
            model, sampler, hook, copy.deepcopy(batch), args.denoise_steps, args.img_H, args.img_W,
            fixed_noise=getattr(args, 'fixed_noise', None)
        )
        attn_np = attn.detach().cpu().numpy() if torch.is_tensor(attn) else np.asarray(attn)
        return {"map": attn_np, "gen_img": gen_img, "cloth_img": cloth_img}


class LIMEExplainer:
    name = "lime"
    stochastic = True

    def explain(self, model, sampler, hook, batch, scorer, args):
        from sklearn.linear_model import Ridge

        cloth_tensor = batch["cloth"][0].cpu().numpy()
        cloth_img = np.uint8(((cloth_tensor + 1.0) * 127.5))
        segments = get_superpixels(cloth_img, n_segments=args.n_segments, region_mask=get_garment_mask(batch) if args.mask_segments else None)
        n_segs = int(segments.max()) + 1

        original_generated = run_inference_on_batch(model, sampler, copy.deepcopy(batch), args.denoise_steps, args.img_H, args.img_W, fixed_noise=getattr(args, 'fixed_noise', None))

        masks = np.random.randint(0, 2, size=(args.n_perturb_samples, n_segs))
        masks[0, :] = 1
        scores = np.zeros(args.n_perturb_samples)
        for i in range(args.n_perturb_samples):
            perturbed = perturb_image_segments(cloth_img, segments, masks[i], baseline=args.baseline)
            perturbed_norm = (perturbed.astype(np.float32) / 127.5) - 1.0
            batch_i = copy.deepcopy(batch)
            batch_i["cloth"] = torch.from_numpy(perturbed_norm).unsqueeze(0)
            gen_img = run_inference_on_batch(model, sampler, batch_i, args.denoise_steps, args.img_H, args.img_W, fixed_noise=getattr(args, 'fixed_noise', None))
            scores[i] = scorer.score(gen_img, original_generated, region_mask=getattr(args, 'score_region_mask', None))

        print(f"[lime] score spread: mean={scores.mean():.5f} std={scores.std():.5f} "
              f"min={scores.min():.5f} max={scores.max():.5f}")
        if scores.std() < 1e-4:
            print("[lime] WARNING: near-zero score variance -- the surrogate is fitting "
                  "noise. Attribution map will not be meaningful.")

        distances = np.linalg.norm(masks - masks[0], axis=1) / np.sqrt(n_segs)
        weights = np.exp(-(distances ** 2) / 0.25)
        surrogate = Ridge(alpha=1.0)
        surrogate.fit(masks, scores, sample_weight=weights)

        map_arr = np.zeros(segments.shape, dtype=np.float32)
        for seg_id, coef in enumerate(surrogate.coef_):
            map_arr[segments == seg_id] = coef
        return {"map": map_arr, "gen_img": original_generated, "cloth_img": cloth_img}


class SHAPExplainer:
    name = "shap"
    stochastic = True

    def explain(self, model, sampler, hook, batch, scorer, args):
        if not _HAS_SHAP:
            raise ImportError("shap package not installed -- pip install shap")

        cloth_tensor = batch["cloth"][0].cpu().numpy()
        cloth_img = np.uint8(((cloth_tensor + 1.0) * 127.5))
        segments = get_superpixels(cloth_img, n_segments=args.n_segments, region_mask=get_garment_mask(batch) if args.mask_segments else None)
        n_segs = int(segments.max()) + 1

        original_generated = run_inference_on_batch(model, sampler, copy.deepcopy(batch), args.denoise_steps, args.img_H, args.img_W, fixed_noise=getattr(args, 'fixed_noise', None))

        def predict_fn(mask_matrix):
            out = np.zeros(mask_matrix.shape[0], dtype=np.float32)
            for i, mv in enumerate(mask_matrix):
                perturbed = perturb_image_segments(cloth_img, segments, mv, baseline=args.baseline)
                perturbed_norm = (perturbed.astype(np.float32) / 127.5) - 1.0
                batch_i = copy.deepcopy(batch)
                batch_i["cloth"] = torch.from_numpy(perturbed_norm).unsqueeze(0)
                gen_img = run_inference_on_batch(model, sampler, batch_i, args.denoise_steps, args.img_H, args.img_W, fixed_noise=getattr(args, 'fixed_noise', None))
                out[i] = scorer.score(gen_img, original_generated, region_mask=getattr(args, 'score_region_mask', None))
            return out

        background = np.zeros((1, n_segs))
        instance = np.ones((1, n_segs))
        explainer = shap_lib.KernelExplainer(predict_fn, background)
        shap_values = np.asarray(explainer.shap_values(instance, nsamples=args.n_perturb_samples, silent=True)).flatten()

        map_arr = np.zeros(segments.shape, dtype=np.float32)
        for seg_id, val in enumerate(shap_values):
            map_arr[segments == seg_id] = val
        return {"map": map_arr, "gen_img": original_generated, "cloth_img": cloth_img}


class RISEExplainer:
    """Petsiuk et al. (2018), "RISE: Randomized Input Sampling for
    Explanation of Black-box Models" (BMVC). Adapted here from classification
    to a continuous scorer: importance = weighted average of random masks,
    weighted by the score each masked input produced."""
    name = "rise"
    stochastic = True

    def explain(self, model, sampler, hook, batch, scorer, args):
        cloth_tensor = batch["cloth"][0].cpu().numpy()
        cloth_img = np.uint8(((cloth_tensor + 1.0) * 127.5))
        H, W = cloth_img.shape[:2]

        original_generated = run_inference_on_batch(model, sampler, copy.deepcopy(batch), args.denoise_steps, args.img_H, args.img_W, fixed_noise=getattr(args, 'fixed_noise', None))

        grid_h, grid_w = args.rise_grid_size, int(args.rise_grid_size * W / H)
        baseline_fill = cloth_img.reshape(-1, 3).mean(axis=0).astype(np.uint8) if args.baseline == "mean" \
            else np.zeros(3, dtype=np.uint8)

        saliency = np.zeros((H, W), dtype=np.float64)
        mask_sum = np.zeros((H, W), dtype=np.float64)

        for i in range(args.n_perturb_samples):
            small_mask = (np.random.rand(grid_h, grid_w) < args.rise_keep_prob).astype(np.float32)
            full_mask = cv2.resize(small_mask, (W, H), interpolation=cv2.INTER_LINEAR)
            # keep background untouched, for the same reason superpixels are
            # garment-restricted (see get_superpixels docstring)
            if args.mask_segments:
                garment = get_garment_mask(batch).astype(np.float32)
                full_mask = full_mask * garment + (1.0 - garment)

            perturbed = (cloth_img.astype(np.float32) * full_mask[:, :, None] +
                         baseline_fill.astype(np.float32) * (1 - full_mask[:, :, None])).astype(np.uint8)
            perturbed_norm = (perturbed.astype(np.float32) / 127.5) - 1.0

            batch_i = copy.deepcopy(batch)
            batch_i["cloth"] = torch.from_numpy(perturbed_norm).unsqueeze(0)
            gen_img = run_inference_on_batch(model, sampler, batch_i, args.denoise_steps, args.img_H, args.img_W, fixed_noise=getattr(args, 'fixed_noise', None))
            s = scorer.score(gen_img, original_generated, region_mask=getattr(args, 'score_region_mask', None))

            saliency += s * full_mask
            mask_sum += full_mask

        map_arr = saliency / (mask_sum + 1e-8)
        return {"map": map_arr.astype(np.float32), "gen_img": original_generated, "cloth_img": cloth_img}


class GradCAMExplainer:
    """Single-timestep Grad-CAM, following Selvaraju et al. (2017), applied
    to the first non-middle-block SpatialTransformer layer in the diffusion
    U-Net (the same layer family I2AM's own hook targets) -- see module
    docstring for why this is single-timestep rather than full-trajectory."""
    name = "gradcam"
    stochastic = False

    def _pick_target_layer(self, model, controlnet):
        """
        Must target a layer in the DECODER (output_blocks), not the encoder.

        StableVITON.forward (cldm/warping_cldm_network.py) wraps its entire
        encoder path in a hardcoded `with torch.no_grad():` block:

            with torch.no_grad():
                h = x.type(self.dtype)
                for module in self.input_blocks:
                    h = module(h, emb, context)
                h = self.middle_block(h, emb, context)

        Any forward hook on an input_blocks / middle_block layer therefore sees
        requires_grad=False no matter what the caller does -- an inner no_grad
        context cannot be overridden by an outer enable_grad(). The output_blocks
        run outside that block and are grad-tracked, so Grad-CAM has to read from
        there. Note this means the CAM reflects decoder-side features, which is a
        genuine scope limitation worth stating in the methodology rather than a
        free choice.
        """
        target_cls = CustomSpatialTransformer if controlnet else SpatialTransformer
        candidates = [
            (name, module)
            for name, module in model.model.diffusion_model.named_modules()
            if isinstance(module, target_cls)
            and "output_blocks" in name
            and "warp_flow_blks" not in name
        ]
        if not candidates:
            raise RuntimeError(
                "no grad-tracked SpatialTransformer found in output_blocks. "
                "The encoder (input_blocks/middle_block) is inside a hardcoded "
                "torch.no_grad() in StableVITON.forward and cannot be used."
            )
        # last decoder transformer = closest to the output, highest spatial res
        name, module = candidates[-1]
        print(f"[gradcam] target layer: {name}")
        return module

    def explain(self, model, sampler, hook, batch, scorer, args):
        cloth_tensor = batch["cloth"][0].cpu().numpy()
        cloth_img = np.uint8(((cloth_tensor + 1.0) * 127.5))

        target_layer = self._pick_target_layer(model, args.controlnet)
        activation_store = {}

        def fwd_hook(module, inp, out):
            out.retain_grad()
            activation_store["act"] = out

        handle = target_layer.register_forward_hook(fwd_hook)
        ckpt_restore = set_gradient_checkpointing(model, False)
        try:
            with torch.enable_grad():
                model.zero_grad(set_to_none=True)
                for param in target_layer.parameters():
                    if not param.requires_grad:
                        param.requires_grad_(True)

                eps = single_step_noise_prediction(model, batch, fixed_noise=getattr(args, 'fixed_noise', None))
                target = eps.pow(2).mean()  # scalar target: overall noise-prediction magnitude
                target.backward()

                act = activation_store["act"]
                grad = act.grad  # same shape as act: [B, C, h, w] (or [B, seq, C] depending on layer -- see note below)
                if grad is None:
                    raise RuntimeError(
                        "no gradient reached the target layer. This usually means the "
                        "activation tensor itself had requires_grad=False at the point "
                        "the forward hook fired -- check that target_layer's parameters "
                        "have requires_grad=True (printed above) and that --controlnet "
                        "matches your model config."
                    )

                # SpatialTransformer output is [B, C, h, w] in this codebase
                weights = grad.mean(dim=(2, 3), keepdim=True)
                cam = F.relu((weights * act).sum(dim=1)).squeeze(0)
                map_arr = cam.detach().cpu().numpy()
        finally:
            handle.remove()
            restore_gradient_checkpointing(ckpt_restore)

        gen_img = run_inference_on_batch(model, sampler, copy.deepcopy(batch), args.denoise_steps, args.img_H, args.img_W, fixed_noise=getattr(args, 'fixed_noise', None))
        return {"map": map_arr, "gen_img": gen_img, "cloth_img": cloth_img}


class IntegratedGradientsExplainer:
    """Single-timestep Integrated Gradients (Sundararajan et al., 2017),
    integrating gradients of the noise-prediction target w.r.t. the cloth
    reference image pixels, from a baseline (black or mean-fill) to the
    actual input. See module docstring for the single-timestep rationale."""
    name = "ig"
    stochastic = False

    def explain(self, model, sampler, hook, batch, scorer, args):
        cloth_tensor_np = batch["cloth"][0].cpu().numpy()
        cloth_img = np.uint8(((cloth_tensor_np + 1.0) * 127.5))

        cloth_input = batch["cloth"].cuda().float()  # [1,H,W,3], range [-1,1]
        if args.baseline == "black":
            baseline = -torch.ones_like(cloth_input)
        else:
            baseline = torch.full_like(cloth_input, cloth_input.mean().item())

        steps = args.ig_steps
        grads_accum = torch.zeros_like(cloth_input)

        ckpt_restore = set_gradient_checkpointing(model, False)
        try:
            with torch.enable_grad():
                for k in range(1, steps + 1):
                    alpha = k / steps
                    interpolated = baseline + alpha * (cloth_input - baseline)
                    interpolated = interpolated.clone().requires_grad_(True)

                    model.zero_grad(set_to_none=True)
                    eps = single_step_noise_prediction(model, batch, cloth_override=interpolated, fixed_noise=getattr(args, 'fixed_noise', None))
                    target = eps.pow(2).mean()
                    grad, = torch.autograd.grad(target, interpolated)
                    grads_accum += grad.detach()
        finally:
            restore_gradient_checkpointing(ckpt_restore)

        avg_grad = grads_accum / steps
        ig = (cloth_input - baseline) * avg_grad  # [1,H,W,3]
        map_arr = ig.abs().sum(dim=-1).squeeze(0).detach().cpu().numpy()  # [H,W]

        gen_img = run_inference_on_batch(model, sampler, copy.deepcopy(batch), args.denoise_steps, args.img_H, args.img_W, fixed_noise=getattr(args, 'fixed_noise', None))
        return {"map": map_arr, "gen_img": gen_img, "cloth_img": cloth_img}


EXPLAINER_REGISTRY = {
    "i2am": I2AMExplainer,
    "lime": LIMEExplainer,
    "shap": SHAPExplainer,
    "rise": RISEExplainer,
    "gradcam": GradCAMExplainer,
    "ig": IntegratedGradientsExplainer,
}
