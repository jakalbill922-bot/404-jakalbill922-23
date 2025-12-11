#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trellis image->3D pipeline (VGGT-conditioned SS-Flow)

This rewrite keeps your original SLAT flow conditioning on DINOv2 the same,
**but replaces only the Sparse-Structure (SS) flow conditioning** with
multi‑view features from VGGT, passed through your ModulatedMultiViewCond
("Atten.") to produce the stereo tokens used by SS-Flow.

Key changes vs original:
- Load VGGT (facebookresearch/vggt) and extract multi‑view features with the
  Aggregator.
- Build a 3072‑dim context per patch by concat([patch_tokens(1024),
  aggregated_tokens(2048)]).
- Feed that context to ModulatedMultiViewCond (your "Atten.") to obtain
  [B, num_tokens, 1024] stereo tokens for the SS-Flow sampler.
- SLAT flow still uses the original DINOv2 image_cond_model conditioning.
"""

from __future__ import annotations
from typing import *
from contextlib import contextmanager
import math
import gc

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from PIL import Image
import rembg
from easydict import EasyDict as edict

from .base import Pipeline
from . import samplers
from ..modules import sparse as sp
from ..representations import Gaussian

# --------------------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------------------

def downsample_coords(coords: torch.Tensor, voxel_budget: int = 20_000) -> torch.Tensor:
    N = coords.shape[0]
    if N <= voxel_budget:
        return coords
    ratio = N / voxel_budget
    stride = max(1, 1 << round(math.log2(ratio ** (1 / 3))))
    mask = (
        (coords[:, 1] & (stride - 1) == 0)
        & (coords[:, 2] & (stride - 1) == 0)
        & (coords[:, 3] & (stride - 1) == 0)
    )
    ds = coords[mask]
    if ds.shape[0] > voxel_budget:
        idx = torch.randperm(ds.shape[0], device=coords.device)[:voxel_budget]
        ds = ds[idx]
    return ds


# --------------------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------------------

class TrellisVGGTTo3DPipeline(Pipeline):
    """
    Pipeline for inferring Trellis image-to-3D models.

    Args:
        models (dict[str, nn.Module]): Models to use (flow models, decoders, Atten module, etc.).
        sparse_structure_sampler (samplers.Sampler): Sampler for sparse structure latent.
        slat_sampler (samplers.Sampler): Sampler for structured latent (SLAT).
        slat_normalization (dict): Normalization params for SLAT latent.
        image_cond_model (str): DINOv2 name for SLAT conditioning (unchanged).
    """

    # ------------------------------ init / load -------------------------------------
    def __init__(
        self,
        models: dict[str, nn.Module] | None = None,
        sparse_structure_sampler: samplers.Sampler | None = None,
        slat_sampler: samplers.Sampler | None = None,
        slat_normalization: dict | None = None,
        image_cond_model: str | None = None,
    ):
        if models is None:
            return
        super().__init__(models)
        self.sparse_structure_sampler = sparse_structure_sampler
        self.slat_sampler = slat_sampler
        self.sparse_structure_sampler_params: dict = {}
        self.slat_sampler_params: dict = {}
        self.slat_normalization = slat_normalization
        self.rembg_session = None

        # For SLAT (unchanged): DINOv2 encoder
        self._init_image_cond_model(image_cond_model)
        # For SS-Flow (new): VGGT aggregator
        self._init_vggt_model()

    @staticmethod
    def from_pretrained(path: str) -> "TrellisVGGTTo3DPipeline":
        """Load a pretrained pipeline and construct samplers from config."""
        pipeline = super(TrellisVGGTTo3DPipeline, TrellisVGGTTo3DPipeline).from_pretrained(path)
        print("Loaded base pipeline:", type(pipeline), pipeline)
        new_pipeline = TrellisVGGTTo3DPipeline()
        new_pipeline.__dict__ = pipeline.__dict__
        args = pipeline._pretrained_args

        new_pipeline.sparse_structure_sampler = getattr(samplers, args['sparse_structure_sampler']['name'])(**args['sparse_structure_sampler']['args'])
        new_pipeline.sparse_structure_sampler_params = args['sparse_structure_sampler']['params']

        new_pipeline.slat_sampler = getattr(samplers, args['slat_sampler']['name'])(**args['slat_sampler']['args'])
        new_pipeline.slat_sampler_params = args['slat_sampler']['params']
        new_pipeline.slat_normalization = args['slat_normalization']

        new_pipeline._init_image_cond_model(args['image_cond_model'])  # SLAT cond
        new_pipeline._init_vggt_model("/root/vggt-object-v0-1")  # SS cond
        
        print(f"Loaded TrellisVGGTTo3DPipeline from {path}")

        return new_pipeline

    # ------------------------------ encoders ----------------------------------------
    def _init_image_cond_model(self, name: str):
        """DINOv2 encoder for SLAT flow (kept as-is)."""
        dinov2_model = torch.hub.load('facebookresearch/dinov2', name, pretrained=True)
        dinov2_model.eval()
        self.models['image_cond_model'] = dinov2_model
        self.image_cond_model_transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def _init_vggt_model(self, model_id: str | None = None):
        """Load VGGT (aggregator + heads). We only need the aggregator here.
        If `model_id` is None, default to Stable-X finetune: "Stable-X/vggt-object-v0-1".
        """
        try:
            from vggt.models.vggt import VGGT  # type: ignore
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "Cannot import VGGT. Please `pip install git+https://github.com/facebookresearch/vggt` "
                "or add it to your environment."
            ) from e

        model_id = model_id or "Stable-X/vggt-object-v0-1"

        # # Choose autocast dtype following official quickstart (bf16 on Ampere+ else fp16)
        # if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
        #     self.vggt_dtype = torch.bfloat16
        # else:
        #     self.vggt_dtype = torch.float16

        # Load the finetuned weights from Hugging Face and move to device
        self.models['vggt'] = VGGT.from_pretrained(model_id).to(self.device).eval()

    # ------------------------------ image preproc -----------------------------------
    def preprocess_image(self, input: Image.Image, resolution: int = 512) -> Image.Image:
        """Remove background (if needed), square‑pad, resize to `resolution` and keep RGBA.
        Returns an RGBA PIL image where RGB is premultiplied by alpha and alpha kept in A.
        """
        has_alpha = False
        if input.mode == 'RGBA':
            alpha = np.array(input)[:, :, 3]
            if not np.all(alpha == 255):
                has_alpha = True
        if has_alpha:
            output = input
        else:
            input = input.convert('RGB')
            max_size = max(input.size)
            scale = min(1, 1024 / max_size)
            if scale < 1:
                input = input.resize((int(input.width * scale), int(input.height * scale)), Image.Resampling.LANCZOS)
            if getattr(self, 'rembg_session', None) is None:
                self.rembg_session = rembg.new_session('u2net')
            output = rembg.remove(input, session=self.rembg_session)

        output_np = np.array(output)
        alpha = output_np[:, :, 3]
        bbox = np.argwhere(alpha > 0.8 * 255)
        if len(bbox) == 0:
            return input.convert('RGB')
        x0, y0, x1, y1 = np.min(bbox[:, 1]), np.min(bbox[:, 0]), np.max(bbox[:, 1]), np.max(bbox[:, 0])
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        size = int(max(x1 - x0, y1 - y0) * 1.2)
        bx0, by0, bx1, by1 = int(cx - size // 2), int(cy - size // 2), int(cx + size // 2), int(cy + size // 2)
        bx0, by0, bx1, by1 = max(0, bx0), max(0, by0), min(output.width, bx1), min(output.height, by1)
        output = output.crop((bx0, by0, bx1, by1))

        # square pad
        w, h = output.size
        if w > h:
            padded = Image.new('RGBA', (w, w), (0, 0, 0, 0))
            padded.paste(output, (0, (w - h) // 2))
        else:
            padded = Image.new('RGBA', (h, h), (0, 0, 0, 0))
            padded.paste(output, ((h - w) // 2, 0))

        padded = padded.resize((resolution, resolution), Image.Resampling.LANCZOS)
        arr = np.array(padded).astype(np.float32) / 255.0
        arr = np.dstack((arr[:, :, :3] * arr[:, :, 3:4], arr[:, :, 3]))  # premultiply
        return Image.fromarray((arr * 255).astype(np.uint8), mode='RGBA')

    # ------------------------------ SLAT cond (unchanged) ---------------------------
    @torch.no_grad()
    def encode_image(self, image: torch.Tensor | list[Image.Image]) -> torch.Tensor:
        """Encode images with DINOv2 for SLAT flow cond (kept from your original)."""
        if isinstance(image, torch.Tensor):
            assert image.ndim == 4, "Image tensor should be batched (B, C, H, W)"
            x = image
        elif isinstance(image, list):
            assert all(isinstance(i, Image.Image) for i in image)
            imgs = [i.resize((518, 518), Image.LANCZOS) for i in image]
            imgs = [np.array(i.convert('RGB')).astype(np.float32) / 255 for i in imgs]
            x = torch.stack([torch.from_numpy(i).permute(2, 0, 1).float() for i in imgs]).to(self.device)
        else:
            raise ValueError(f"Unsupported type of image: {type(image)}")
        x = self.image_cond_model_transform(x).to(self.device)
        feats = self.models['image_cond_model'](x, is_training=True)['x_prenorm']
        patchtokens = F.layer_norm(feats, feats.shape[-1:])
        return patchtokens

    def get_cond(self, image: torch.Tensor | list[Image.Image]) -> dict:
        cond = self.encode_image(image)
        return { 'cond': cond, 'neg_cond': torch.zeros_like(cond) }

    # ------------------------------ VGGT cond (new for SS-Flow) ---------------------
    def _pil_rgba_to_rgb_tensor(self, imgs: list[Image.Image], resize_to: int = 518) -> torch.Tensor:
        """Convert RGBA PIL images to [S, 3, H, W] float32 in [0,1] for VGGT.
        - Composite over transparent background (already premultiplied -> just drop A).
        - Resize to 518 like VGGT default.
        """
        out: list[torch.Tensor] = []
        for im in imgs:
            if im.mode != 'RGBA':
                im = im.convert('RGBA')
            im = im.resize((resize_to, resize_to), Image.Resampling.LANCZOS)
            arr = np.array(im).astype(np.float32) / 255.0
            rgb = arr[..., :3]
            out.append(torch.from_numpy(rgb).permute(2, 0, 1))
        x = torch.stack(out, dim=0)
        return x

    @torch.no_grad()
    def _vggt_multiview_features(self, images_rgba: list[Image.Image]) -> torch.Tensor:
        """Build ctx features of shape [B, S*P, 3072] from VGGT."""
        assert 'vggt' in self.models, "VGGT model is not initialized"
        vggt = self.models['vggt']
        x = self._pil_rgba_to_rgb_tensor(images_rgba).to(self.device).unsqueeze(0)
        # Enable FP16 autocast for VGGT aggregator
        vggt = vggt.to(torch.float16)
        x = x.to(torch.float16)
        with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
            agg_list, patch_start_idx = vggt.aggregator(x)

        agg = agg_list[-1]
        B, S, P_total, _ = agg.shape
        agg_patch = agg[:, :, patch_start_idx:, :]
        mean = getattr(vggt.aggregator, "_resnet_mean")
        std = getattr(vggt.aggregator, "_resnet_std")
        x_norm = (x - mean) / std
        BS, C, H, W = (x_norm.shape[0] * x_norm.shape[1],) + tuple(x_norm.shape[2:])
        x_flat = x_norm.view(BS, C, H, W)
        patch_tokens = vggt.aggregator.patch_embed(x_flat)
        if isinstance(patch_tokens, dict):
            patch_tokens = patch_tokens["x_norm_patchtokens"]
        P = patch_tokens.shape[1]
        patch_tokens = patch_tokens.view(B, S, P, -1)
        if agg_patch.shape[2] != P:
            P_common = min(P, agg_patch.shape[2])
            patch_tokens = patch_tokens[:, :, :P_common]
            agg_patch = agg_patch[:, :, :P_common]
        ctx = torch.cat([patch_tokens, agg_patch], dim=-1)
        ctx = ctx.view(B, S * ctx.shape[2], ctx.shape[3])
        return ctx

    @torch.no_grad()
    def _get_ssflow_cond_from_vggt(self, images_rgba: list[Image.Image]) -> dict:
        # 1. Get VGGT features
        ctx = self._vggt_multiview_features(images_rgba)  # [B, L, 3072]
        ctx = ctx.to(torch.float16)

        # 2. Force cond module to FP16
        cond_mod = self.models['sparse_structure_vggt_cond']
        cond_mod = cond_mod.to(torch.float16)
        cond_mod.use_fp16 = True
        cond_mod.dtype = torch.float16

        # 3. Run forward in autocast
        with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
            stereo_tokens = cond_mod(ctx)  # [B, num_tokens, 1024]

        neg = torch.zeros_like(stereo_tokens)
        return {'cond': stereo_tokens, 'neg_cond': neg}

    # ------------------------------ sampling / decode --------------------------------
    def sample_sparse_structure(
        self,
        cond: dict,
        num_samples: int = 1,
        sampler_params: dict = {},
    ) -> torch.Tensor:
        # Occupancy latent
        flow_model = self.models['sparse_structure_flow_model']
        reso = flow_model.resolution
        noise = torch.randn(num_samples, flow_model.in_channels, reso, reso, reso).to(self.device)
        params = {**self.sparse_structure_sampler_params, **sampler_params}
        z_s = self.sparse_structure_sampler.sample(
            flow_model, noise, **cond, **params, verbose=True
        ).samples
        # Decode to coords
        decoder = self.models['sparse_structure_decoder']
        coords = torch.argwhere(decoder(z_s) > 0)[:, [0, 2, 3, 4]].int()
        return coords

    def decode_slat(self, slat: sp.SparseTensor, formats: List[str] = ['mesh', 'gaussian']) -> dict:
        ret: dict[str, Any] = {}
        if 'gaussian' in formats:
            ret['gaussian'] = self.models['slat_decoder_gs'](slat)
        return ret

    def sample_slat(self, cond: dict, coords: torch.Tensor, sampler_params: dict = {}) -> sp.SparseTensor:
        flow_model = self.models['slat_flow_model']
        noise = sp.SparseTensor(
            feats=torch.randn(coords.shape[0], flow_model.in_channels).to(self.device),
            coords=coords,
        )
        params = {**self.slat_sampler_params, **sampler_params}
        slat = self.slat_sampler.sample(
            flow_model, noise, **cond, **params, verbose=True
        ).samples
        std = torch.tensor(self.slat_normalization['std'])[None].to(slat.device)
        mean = torch.tensor(self.slat_normalization['mean'])[None].to(slat.device)
        slat = slat * std + mean
        return slat

    # ------------------------------ top-level runs -----------------------------------
    @torch.no_grad()
    def run(
        self,
        image: Image.Image,
        num_samples: int = 1,
        seed: int = 42,
        sparse_structure_sampler_params: dict = {},
        slat_sampler_params: dict = {},
        formats: List[str] = ['mesh', 'gaussian'],
        preprocess_image: bool = True,
    ) -> dict:
        # Preprocess once (RGBA)
        img_rgba = self.preprocess_image(image) if preprocess_image else image

        # SLAT cond (unchanged)
        slat_cond = self.get_cond([img_rgba])
        if num_samples > 1:
            slat_cond['cond'] = slat_cond['cond'].repeat(num_samples, 1, 1)
            slat_cond['neg_cond'] = slat_cond['neg_cond'].repeat(num_samples, 1, 1)

        # SS-Flow cond from VGGT + Atten
        ss_cond = self._get_ssflow_cond_from_vggt([img_rgba])
        if num_samples > 1:
            ss_cond['cond'] = ss_cond['cond'].repeat(num_samples, 1, 1)
            ss_cond['neg_cond'] = ss_cond['neg_cond'].repeat(num_samples, 1, 1)

        torch.manual_seed(seed)
        coords = self.sample_sparse_structure(ss_cond, num_samples, sparse_structure_sampler_params)
        torch.cuda.empty_cache(); gc.collect()

        slat = self.sample_slat(slat_cond, coords, slat_sampler_params)
        torch.cuda.empty_cache(); gc.collect()
        return self.decode_slat(slat, formats)

    @contextmanager
    def inject_sampler_multi_image(
        self,
        sampler_name: str,
        num_images: int,
        num_steps: int | None,
        mode: Literal['stochastic', 'multidiffusion'] = 'stochastic',
    ):
        sampler = getattr(self, sampler_name)
        setattr(sampler, f'_old_inference_model', sampler._inference_model)
        if mode == 'stochastic':
            if num_images > (num_steps or 0):
                print(f"\033[93mWarning: num_images > num_steps for {sampler_name}.\033[0m")
            cond_indices = (np.arange(num_steps or num_images) % num_images).tolist()
            def _new_inference_model(self, model, x_t, t, cond, **kwargs):
                cond_idx = cond_indices.pop(0)
                cond_i = cond[cond_idx:cond_idx+1]
                return self._old_inference_model(model, x_t, t, cond=cond_i, **kwargs)
        elif mode == 'multidiffusion':
            from .samplers import FlowEulerSampler
            def _new_inference_model(self, model, x_t, t, cond, neg_cond, cfg_strength, cfg_interval, **kwargs):
                if cfg_interval[0] <= t <= cfg_interval[1]:
                    preds = [FlowEulerSampler._inference_model(self, model, x_t, t, cond[i:i+1], **kwargs)
                             for i in range(len(cond))]
                    pred = sum(preds) / len(preds)
                    neg_pred = FlowEulerSampler._inference_model(self, model, x_t, t, neg_cond, **kwargs)
                    return (1 + cfg_strength) * pred - cfg_strength * neg_pred
                else:
                    preds = [FlowEulerSampler._inference_model(self, model, x_t, t, cond[i:i+1], **kwargs)
                             for i in range(len(cond))]
                    return sum(preds) / len(preds)
        else:
            raise ValueError(f"Unsupported mode: {mode}")
        sampler._inference_model = _new_inference_model.__get__(sampler, type(sampler))
        try:
            yield
        finally:
            sampler._inference_model = sampler._old_inference_model
            delattr(sampler, f'_old_inference_model')

    @torch.no_grad()
    def run_multi_image(
        self,
        images: List[Image.Image],
        normalized_images: List[Image.Image],  # kept for API parity; not used by VGGT branch directly
        num_samples: int = 1,
        seed: int = 42,
        sparse_structure_sampler_params: dict = {},
        slat_sampler_params: dict = {},
        formats: List[str] = ['mesh', 'gaussian', 'radiance_field'],
        preprocess_image: bool = True,
        mode: Literal['stochastic', 'multidiffusion'] = 'stochastic',
    ) -> dict:
        # Preprocess to RGBA
        if preprocess_image:
            images_rgba = [self.preprocess_image(im) for im in images]
        else:
            images_rgba = images

        # SLAT cond (still DINOv2 on *images*, as before)
        slat_cond = self.get_cond(images_rgba)
        slat_cond['neg_cond'] = slat_cond['neg_cond'][:1]

        # SS-Flow cond via VGGT + Atten on the (optionally) normalized set
        ss_cond = self._get_ssflow_cond_from_vggt(images_rgba)
        ss_cond['neg_cond'] = ss_cond['neg_cond'][:1]

        torch.manual_seed(seed)
        ss_steps = {**self.sparse_structure_sampler_params, **sparse_structure_sampler_params}.get('steps')
        with self.inject_sampler_multi_image('sparse_structure_sampler', len(images_rgba), ss_steps, mode=mode):
            coords = self.sample_sparse_structure(ss_cond, num_samples, sparse_structure_sampler_params)
        torch.cuda.empty_cache(); gc.collect()

        slat_steps = {**self.slat_sampler_params, **slat_sampler_params}.get('steps')
        with self.inject_sampler_multi_image('slat_sampler', len(images_rgba), slat_steps, mode=mode):
            slat = self.sample_slat(slat_cond, coords, slat_sampler_params)
        torch.cuda.empty_cache(); gc.collect()

        return self.decode_slat(slat, formats)


# # For pylance / direct run
# if __name__ == '__main__':
#     pass
