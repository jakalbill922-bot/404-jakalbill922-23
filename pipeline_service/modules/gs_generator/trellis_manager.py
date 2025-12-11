from __future__ import annotations

import base64
import os
import time
from pathlib import Path
from typing import Optional
import io

import torch
import numpy as np
from scipy.spatial.transform import Rotation
from PIL import Image, ImageStat

from config import Settings
from logger_config import logger
from libs.trellis.pipelines import TrellisImageTo3DPipeline 
from schemas import TrellisResult, TrellisRequest, TrellisParams

class TrellisService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.pipeline: Optional[TrellisImageTo3DPipeline] = None
        self.gpu = settings.trellis_gpu
        self.default_params = TrellisParams.from_settings(self.settings)

    async def startup(self) -> None:
        logger.info("Loading Trellis pipeline...")
        os.environ.setdefault("ATTN_BACKEND", "flash-attn")
        os.environ.setdefault("SPCONV_ALGO", "native")

        if torch.cuda.is_available():
            torch.cuda.set_device(self.gpu)

        self.pipeline = TrellisImageTo3DPipeline.from_pretrained(
            self.settings.trellis_model_id
        )
        self.pipeline.cuda()
        logger.success("Trellis pipeline ready.")

    async def shutdown(self) -> None:
        self.pipeline = None
        logger.info("Trellis pipeline closed.")

    def is_ready(self) -> bool:
        return self.pipeline is not None

    def generate(
        self,
        trellis_request: TrellisRequest,
    ) -> TrellisResult:
        if not self.pipeline:
            raise RuntimeError("Trellis pipeline not loaded.")

        image_rgb = trellis_request.image.convert("RGB")
        logger.info(f"Generating Trellis {trellis_request.seed=} and image size {trellis_request.image.size}")

        params = self.default_params.overrided(trellis_request.params)

        start = time.time()
        buffer = None
        try:
            outputs = self.pipeline.run_multi_image(
                [image_rgb],
                seed=trellis_request.seed,
                sparse_structure_sampler_params={
                    "steps": params.sparse_structure_steps,
                    "cfg_strength": params.sparse_structure_cfg_strength,
                },
                slat_sampler_params={
                    "steps": params.slat_steps,
                    "cfg_strength": params.slat_cfg_strength,
                },
                preprocess_image=False,
                formats=["gaussian"],
                mode="multidiffusion"
            )

            generation_time = time.time() - start
            gaussian = outputs["gaussian"][0]

            # Apply initial rotation (same as in gaussian_processor.py)
            T_initial = np.array([0, 0, 0], dtype=np.float32)
            R_initial_obj = Rotation.from_euler('xyz', [90.0, 0.0, 0.0], degrees=True)
            R_initial = R_initial_obj.as_matrix().astype(np.float32)
            gaussian.transform_data(T_initial, R_initial)

            # Save ply to buffer
            buffer = io.BytesIO()
            gaussian.save_ply(buffer)
            buffer.seek(0)           

            result = TrellisResult(
                ply_file=buffer.getvalue() if buffer else None # bytes
            )

            logger.success(f"Trellis finished generation in {generation_time:.2f}s.")
            return result
        finally:
            if buffer:
                buffer.close()

