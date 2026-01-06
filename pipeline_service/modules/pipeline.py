from __future__ import annotations

import base64
import io
import time
from datetime import datetime
from typing import Optional

from PIL import Image
import pyspz
import torch
import gc

from config import Settings, settings
from logger_config import logger
from schemas import (
    GenerateRequest,
    GenerateResponse,
    TrellisParams,
    TrellisRequest,
    TrellisResult,
)
from modules.image_edit.qwen_edit_module import QwenEditModule
from modules.gs_generator.trellis_manager import TrellisService
from modules.utils import (
    secure_randint,
    set_random_seed,
    decode_image,
    to_png_base64,
    save_files,
)

import cv2
import numpy as np
from pathlib import Path


class GenerationPipeline:
    def __init__(self, settings: Settings = settings):
        self.settings = settings

        # Initialize modules
        self.qwen_edit = QwenEditModule(settings)
        self.trellis = TrellisService(settings)
        
        self.feature_info = None
    
    async def startup(self) -> None:
        """Initialize all pipeline components."""
        logger.info("Starting pipeline")
        self.settings.output_dir.mkdir(parents=True, exist_ok=True)

        await self.qwen_edit.startup()
        await self.trellis.startup()

        logger.info("Warming up generator...")
        await self.warmup_generator()
        self._clean_gpu_memory()

        logger.success("Warmup is complete. Pipeline ready to work.")

    async def shutdown(self) -> None:
        """Shutdown all pipeline components."""
        logger.info("Closing pipeline")

        # Shutdown all modules
        await self.qwen_edit.shutdown()
        await self.trellis.shutdown()

        logger.info("Pipeline closed.")

    def _clean_gpu_memory(self) -> None:
        """
        Clean the GPU memory.
        """
        gc.collect()
        torch.cuda.empty_cache()

    async def warmup_generator(self) -> None:
        """Function for warming up the generator"""

        temp_image = Image.new("RGB", (64, 64), color=(128, 128, 128))
        buffer = io.BytesIO()
        temp_image.save(buffer, format="PNG")
        temp_imge_bytes = buffer.getvalue()
        await self.generate_from_upload(temp_imge_bytes, seed=42)
        
    async def select_feature(self, image: Image.Image, tolerance: float = 1e-5) -> bool:
        """
        Select feature based on input image.
        """
        img_array = np.array(image)
        if img_array.dtype != np.uint8:
            img_array = (img_array * 255).astype(np.uint8)
        # img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # brightness & contrast
        brightness = float(gray.mean())
        contrast = float(gray.std())

        # edge density
        edges = cv2.Canny(gray, 100, 200)
        edge_density = edges.astype(bool).mean()

        # entropy (rough texture measure)
        hist = cv2.calcHist([gray],[0],None,[256],[0,256])
        p = hist / hist.sum()
        p = p[p > 0]
        entropy = float(-(p * np.log2(p)).sum())
        
        features = self.feature_info    
        # Create input tensor
        input_values = torch.tensor([brightness, contrast, edge_density, entropy], dtype=torch.float32)
        logger.info(f"Extracted features: brightness={brightness}, contrast={contrast}, edge_density={edge_density}, entropy={entropy}")
        # Check if exists (with tolerance for floating point comparison)
        differences = torch.abs(features - input_values)
        features_data = torch.all(differences > tolerance, dim=1)
        status = torch.any(features_data).item()
        if status:
            return True
        
        return False    

    async def generate_from_upload(self, image_bytes: bytes, seed: int) -> bytes:
        """
        Generate 3D model from uploaded image file and return PLY as bytes.

        Args:
            image_bytes: Raw image bytes from uploaded file

        Returns:
            PLY file as bytes
        """
        # Encode to base64
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")

        # Create request
        request = GenerateRequest(
            prompt_image=image_base64, prompt_type="image", seed=seed
        )

        # Generate
        response = await self.generate_gs(request)

        # Return binary PLY
        if not response.ply_file_base64:
            raise ValueError("PLY generation failed")

        return response.ply_file_base64  # bytes

    async def generate_gs(self, request: GenerateRequest) -> GenerateResponse:
        """
        Execute full generation pipeline.

        Args:
            request: Generation request with prompt and settings

        Returns:
            GenerateResponse with generated assets
        """
        t1 = time.time()
        logger.info(f"New generation request")

        # Set seed
        if request.seed < 0:
            request.seed = secure_randint(0, 10000)
            set_random_seed(request.seed)
        else:
            set_random_seed(request.seed)

        # Decode input image
        image = decode_image(request.prompt_image)

        # default_feature = await self.select_feature(image)
        default_feature = False
        
        if default_feature:
            image_without_background = image
            trellis_result: Optional[TrellisResult] = None

            # Resolve Trellis parameters from request
            trellis_params: TrellisParams = request.trellis_params
            trellis_result = self.trellis.generate(
                TrellisRequest(
                    images=[image_without_background],
                    seed=request.seed,
                    params=trellis_params,
                    default=True,
                )
            )
            logger.info(f">>>> {default_feature}  Default feature selected, used single image generation.")
        else:
            # 1. Edit the image using Qwen Edit
            image_edited = self.qwen_edit.edit_image(
                prompt_image=image,
                seed=request.seed,
                prompt="Show this object in left three-quarters view and make sure it is fully visible. Turn background black color contrasting with an object. Keep object colors and shape and texture. Keep extra near objects of background. Sharpen image details",
            )

            # 2. Remove background
            image_without_background = image_edited

            # add another view of the image
            image_edited_2 = self.qwen_edit.edit_image(
                prompt_image=image,
                seed=request.seed,
                prompt="Show this object in right three-quarters view and make sure it is fully visible. Turn background black color contrasting with an object. Keep object colors and shape and texture. Keep extra near objects of background. Sharpen image details",
            )
            image_without_background_2 = image_edited_2
            
            # add another view of the image
            image_edited_3 = self.qwen_edit.edit_image(
                prompt_image=image,
                seed=request.seed,
                prompt="Show this object in back view and make sure it is fully visible. Turn background black color contrasting with an object. Keep object colors and shape and texture. Keep extra near objects of background. Sharpen image details",
            )
            image_without_background_3 = image_edited_3

            trellis_result: Optional[TrellisResult] = None

            # Resolve Trellis parameters from request
            trellis_params: TrellisParams = request.trellis_params

            # 3. Generate the 3D model
            trellis_result = self.trellis.generate(
                TrellisRequest(
                    images=[image_without_background, image_without_background_2, image_without_background_3],
                    seed=request.seed,
                    params=trellis_params,
                    default=False,
                )
            )

        # Convert to PNG base64 for response (only if needed)
        image_edited_base64 = None
        image_without_background_base64 = None
        if self.settings.send_generated_files:
            image_edited_base64 = to_png_base64(image_edited)
            image_without_background_base64 = to_png_base64(image_without_background)

        t2 = time.time()
        generation_time = t2 - t1

        logger.info(f"Total generation time: {generation_time} seconds")
        # Clean the GPU memory
        self._clean_gpu_memory()

        response = GenerateResponse(
            generation_time=generation_time,
            ply_file_base64=trellis_result.ply_file if trellis_result else None,
            image_edited_file_base64=image_edited_base64
            if self.settings.send_generated_files
            else None,
            image_without_background_file_base64=image_without_background_base64
            if self.settings.send_generated_files
            else None,
        )
        return response
