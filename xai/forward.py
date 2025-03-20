"""Full 50-step DDIM forward pass and seeded initial-latent helpers.

"""
import torch

from utils import tensor2img

# =========================================================================== #
# Full 50-step model forward pass (mirrors inference.py)
# =========================================================================== #

def build_conditioning(model, batch):
    """Returns (z, c) exactly as inference.py's main loop does, with
    c_crossattn already resolved through CLIP if needed."""
    z, c = model.get_input(batch, "image")
    bs = z.shape[0]
    c_crossattn = c["c_crossattn"][0][:bs]
    if c_crossattn.ndim == 4:
        c_crossattn = model.get_learned_conditioning(c_crossattn)
        c["c_crossattn"] = [c_crossattn]
    return z, c


@torch.no_grad()
def run_inference_on_batch(model, sampler, batch, denoise_steps, img_H, img_W, eta=0.0, fixed_noise=None):
    """
    fixed_noise: if given, used as the noise term in q_sample instead of a
    fresh torch.randn_like on every call. This matters for every
    perturbation-based method: q_sample() defaults to sampling new noise each
    time, so without a fixed tensor here each perturbed sample starts from a
    DIFFERENT x_T and the measured score difference mixes the perturbation's
    effect with sampling noise. With fixed_noise set and eta=0 (deterministic
    DDIM), the whole pipeline becomes a fixed function of the cloth input,
    which is the assumption LIME/SHAP/RISE all rely on.
    """
    batch = {k: (v.cuda() if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
    z, c = build_conditioning(model, batch)
    bs = z.shape[0]
    uc_cross = model.get_unconditional_conditioning(bs)
    uc_full = {"c_concat": c["c_concat"], "c_crossattn": [uc_cross]}
    uc_full["first_stage_cond"] = c["first_stage_cond"]

    sampler.model.batch = batch
    ts = torch.full((1,), 999, device=z.device, dtype=torch.long)
    start_code = model.q_sample(z, ts, noise=fixed_noise)

    shape = (4, img_H // 8, img_W // 8)
    samples, _, _ = sampler.sample(
        denoise_steps, bs, shape, c,
        x_T=start_code, verbose=False, eta=eta,
        unconditional_conditioning=uc_full,
    )
    x_samples = model.decode_first_stage(samples)
    return tensor2img(x_samples[0])


def make_fixed_noise(img_H, img_W, seed=0, device="cuda"):
    """
    Draws ONE noise tensor per image, reused across every perturbation of that
    image. Shape matches the latent (4, H/8, W/8) used everywhere in this
    pipeline. Seeded so a rerun of the same sample reproduces the same x_T --
    which is also what makes the Consistency metric measure the explainer's
    own stochasticity rather than the sampler's.
    """
    g = torch.Generator(device=device).manual_seed(seed)
    return torch.randn((1, 4, img_H // 8, img_W // 8), generator=g, device=device)


@torch.no_grad()
def run_inference_and_get_i2am_attention(model, sampler, hook, batch, denoise_steps, img_H, img_W, eta=0.0, fixed_noise=None):
    gen_img = run_inference_on_batch(model, sampler, batch, denoise_steps, img_H, img_W, eta, fixed_noise=fixed_noise)
    agn_mask = batch["agn_mask"].cuda() if isinstance(batch["agn_mask"], torch.Tensor) else batch["agn_mask"]
    attention_map = hook.make_attention_maps(agn_mask)  # also clears hook state
    return gen_img, attention_map
