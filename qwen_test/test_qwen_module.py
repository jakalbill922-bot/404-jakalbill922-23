"""
Test script for Qwen Edit module with 2511 model
"""
import asyncio
import sys
from pathlib import Path
from PIL import Image

# Add pipeline_service to path
sys.path.insert(0, str(Path(__file__).parent / "pipeline_service"))

from config import settings
from modules.image_edit.qwen_edit_module import QwenEditModule
from logger_config import logger


async def test_qwen_edit():
    """Test Qwen Edit module with various prompts"""
    
    # Initialize module
    qwen = QwenEditModule(settings)
    
    # Startup
    logger.info("Starting Qwen module...")
    await qwen.startup()
    
    # Load test image
    image_path = "qwen_test/image.png"  # Adjust path as needed
    if not Path(image_path).exists():
        logger.error(f"Image not found: {image_path}")
        return
    
    prompt_image = Image.open(image_path)
    logger.info(f"Loaded image: {prompt_image.size}")
    
    # Test prompts (same as pipeline)
    test_prompts = [
        ("left_view", "Show this object in left three-quarters view and make sure it is fully visible. Turn background neutral solid color contrasting with an object. Delete background details. Delete watermarks. Keep object colors. Sharpen image details"),
        ("right_view", "Show this object in right three-quarters view and make sure it is fully visible. Turn background neutral solid color contrasting with an object. Delete background details. Delete watermarks. Keep object colors. Sharpen image details"),
        ("back_view", "Show this object in back view and make sure it is fully visible. Turn background neutral solid color contrasting with an object. Delete background details. Delete watermarks. Keep object colors. Sharpen image details"),
    ]
    
    seed = 42
    
    # Generate and save results
    for name, prompt in test_prompts:
        logger.info(f"\nTesting: {name}")
        logger.info(f"Prompt: {prompt[:80]}...")
        
        result = qwen.edit_image(
            prompt_image=prompt_image,
            seed=seed,
            prompt=prompt
        )
        
        output_path = f"output_qwen_{name}_2511.png"
        result.save(output_path)
        logger.success(f"Saved: {output_path}")
    
    # Shutdown
    await qwen.shutdown()
    logger.success("\nTest completed!")


if __name__ == "__main__":
    asyncio.run(test_qwen_edit())
