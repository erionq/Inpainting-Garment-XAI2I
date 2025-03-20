"""Single-timestep differentiable forward pass for GradCAM / Integrated Gradients.
"""
import torch
import torch.nn.functional as F

from .forward import build_conditioning

# =========================================================================== #
# Single-timestep forward pass for gradient-based methods (GradCAM / IG)
# =========================================================================== #

def single_step_noise_prediction(model, batch, cloth_override=None, fixed_noise=None):
    """
    Runs ONE differentiable forward call through model.apply_model() at the
    fixed timestep t=999 (matching the start_code used elsewhere in the
    pipeline), returning the predicted noise eps. If cloth_override is given
    (a tensor requiring grad), it replaces batch['cloth'] before building
    conditioning, so gradients w.r.t. it can be recovered afterward.
    """
    with torch.enable_grad():
        batch = dict(batch)  # shallow copy -- don't mutate caller's tensors
        if cloth_override is not None:
            batch["cloth"] = cloth_override
        batch = {k: (v.cuda() if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}

        # NOTE: LatentDiffusion.get_input (ddpm.py) is decorated @torch.no_grad(),
        # so EVERYTHING it returns is detached -- z, c_concat, first_stage_cond,
        # and the pre-CLIP cloth tensor. An outer torch.enable_grad() cannot undo
        # that, because the inner no_grad context always wins. Both gradient-based
        # explainers therefore have to re-introduce a grad-tracked path by hand.
        z, c = build_conditioning(model, batch)

        if cloth_override is not None:
            # --- Integrated Gradients path ---
            # Rebuild the CLIP conditioning from cloth_override OUTSIDE no_grad,
            # replicating get_input's own preprocessing (use_imageCLIP=True branch:
            # permute to NCHW -> resize to 224 -> imagenet_norm on [0,1] range),
            # so gradients can actually reach the cloth pixels.
            xc = cloth_override.cuda().permute(0, 3, 1, 2).float()  # [B,3,H,W] in [-1,1]
            xc = F.interpolate(xc, size=(224, 224), mode="bilinear", align_corners=False)
            xc = model.imagenet_norm((xc + 1) / 2)
            c["c_crossattn"] = [model.get_learned_conditioning(xc)]

        ts = torch.full((1,), 999, device=z.device, dtype=torch.long)

        # Force grad on x_noisy. For GradCAM this is what makes every downstream
        # activation grad-tracked (and so hookable with retain_grad) regardless of
        # which parameters happen to be frozen -- the U-Net's own weights are
        # trainable, but all of its *inputs* arrive detached from get_input.
        start_code = model.q_sample(z, ts, noise=fixed_noise).detach().requires_grad_(True)

        eps, _ = model.apply_model(start_code, ts, c)
        return eps


def set_gradient_checkpointing(module, value):
    restore_list = []
    for m in module.modules():
        for attr in ("use_checkpoint", "checkpoint"):
            if hasattr(m, attr):
                restore_list.append((m, attr, getattr(m, attr)))
                setattr(m, attr, value)
    return restore_list


def restore_gradient_checkpointing(restore_list):
    for m, attr, prev_value in restore_list:
        setattr(m, attr, prev_value)
