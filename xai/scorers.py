"""Scoring functions: the scalarisation f: X -> R that defines what is explained.
"""
import numpy as np
import torch

from .deps import _HAS_OPEN_CLIP, open_clip

# =========================================================================== #
# Scoring functions (target for perturbation-based methods + faithfulness)
# =========================================================================== #

class PixelSimilarityScorer:
    name = "pixel_l2"

    def score(self, generated_img, original_generated_img, region_mask=None):
        a = generated_img.astype(np.float32) / 255.0
        b = original_generated_img.astype(np.float32) / 255.0
        sq = (a - b) ** 2
        if region_mask is not None:
            m = region_mask.astype(np.float32)[:, :, None]
            dist = float((sq * m).sum() / (m.sum() * sq.shape[-1] + 1e-8))
        else:
            dist = float(np.mean(sq))
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
    def score(self, generated_img, original_generated_img, region_mask=None):
        # region_mask is accepted for interface parity but intentionally unused:
        # CLIP embeds a whole resized image, so there is no meaningful way to
        # restrict its similarity to a spatial sub-region without cropping,
        # which would change the semantics of the embedding.
        from PIL import Image
        a_t = self.preprocess(Image.fromarray(generated_img)).unsqueeze(0).to(self.device)
        b_t = self.preprocess(Image.fromarray(original_generated_img)).unsqueeze(0).to(self.device)
        a_f = self.model.encode_image(a_t)
        b_f = self.model.encode_image(b_t)
        a_f = a_f / a_f.norm(dim=-1, keepdim=True)
        b_f = b_f / b_f.norm(dim=-1, keepdim=True)
        return float((a_f @ b_f.T).item())


SCORER_REGISTRY = {"pixel_l2": lambda: PixelSimilarityScorer(), "clip": lambda: CLIPSimilarityScorer()}
