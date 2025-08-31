"""
GPU-accelerated metrics calculator for perceptual quality assessment
"""

import asyncio
from typing import Optional, Tuple
import logging

import torch
import torch.nn.functional as F
import numpy as np
from skimage.metrics import structural_similarity
import kornia
import lpips

logger = logging.getLogger(__name__)

class MetricsCalculator:
    """Calculate perceptual quality metrics using GPU acceleration"""
    
    def __init__(self, device: torch.device):
        self.device = device
        
        # Initialize LPIPS model for perceptual distance
        try:
            self.lpips_model = lpips.LPIPS(net='alex').to(device)
            self.lpips_model.eval()
            logger.info("LPIPS model loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load LPIPS model: {e}")
            self.lpips_model = None
    
    async def calculate_ssim(
        self,
        image1: torch.Tensor,
        image2: torch.Tensor,
        window_size: int = 11
    ) -> float:
        """Calculate Structural Similarity Index (SSIM)"""
        
        try:
            # Ensure images are in the right format
            if image1.dim() == 3:
                image1 = image1.unsqueeze(0)
            if image2.dim() == 3:
                image2 = image2.unsqueeze(0)
            
            # Use kornia for GPU-accelerated SSIM
            ssim_val = kornia.metrics.ssim(
                image1,
                image2,
                window_size=window_size,
                reduction='mean'
            )
            
            return float(ssim_val.cpu().item())
            
        except Exception as e:
            logger.error(f"SSIM calculation failed: {e}")
            # Fallback to CPU calculation
            return await self._calculate_ssim_cpu(image1, image2)
    
    async def _calculate_ssim_cpu(
        self,
        image1: torch.Tensor,
        image2: torch.Tensor
    ) -> float:
        """Fallback CPU SSIM calculation"""
        
        # Convert to numpy
        img1_np = image1.cpu().numpy()
        img2_np = image2.cpu().numpy()
        
        # Handle different dimensions
        if img1_np.ndim == 4:
            img1_np = img1_np[0]
            img2_np = img2_np[0]
        
        # Transpose to HWC format
        if img1_np.shape[0] in [1, 3]:
            img1_np = np.transpose(img1_np, (1, 2, 0))
            img2_np = np.transpose(img2_np, (1, 2, 0))
        
        # Calculate SSIM
        ssim_val = structural_similarity(
            img1_np,
            img2_np,
            multichannel=True if img1_np.ndim == 3 else False,
            data_range=1.0
        )
        
        return float(ssim_val)
    
    async def calculate_psnr(
        self,
        image1: torch.Tensor,
        image2: torch.Tensor,
        max_val: float = 1.0
    ) -> float:
        """Calculate Peak Signal-to-Noise Ratio (PSNR)"""
        
        try:
            # Ensure images are in the right format
            if image1.dim() == 3:
                image1 = image1.unsqueeze(0)
            if image2.dim() == 3:
                image2 = image2.unsqueeze(0)
            
            # Calculate MSE
            mse = F.mse_loss(image1, image2)
            
            # Calculate PSNR
            if mse == 0:
                return float('inf')
            
            psnr = 20 * torch.log10(torch.tensor(max_val).to(self.device) / torch.sqrt(mse))
            
            return float(psnr.cpu().item())
            
        except Exception as e:
            logger.error(f"PSNR calculation failed: {e}")
            return 0.0
    
    async def calculate_lpips(
        self,
        image1: torch.Tensor,
        image2: torch.Tensor
    ) -> Optional[float]:
        """Calculate Learned Perceptual Image Patch Similarity (LPIPS)"""
        
        if self.lpips_model is None:
            return None
        
        try:
            # Ensure images are in the right format [B, C, H, W]
            if image1.dim() == 3:
                image1 = image1.unsqueeze(0)
            if image2.dim() == 3:
                image2 = image2.unsqueeze(0)
            
            # Normalize to [-1, 1] range for LPIPS
            image1 = 2 * image1 - 1
            image2 = 2 * image2 - 1
            
            # Calculate LPIPS distance
            with torch.no_grad():
                lpips_val = self.lpips_model(image1, image2)
            
            return float(lpips_val.cpu().item())
            
        except Exception as e:
            logger.error(f"LPIPS calculation failed: {e}")
            return None
    
    async def calculate_all_metrics(
        self,
        image1: torch.Tensor,
        image2: torch.Tensor
    ) -> Tuple[float, float, Optional[float]]:
        """Calculate all metrics in parallel"""
        
        # Run calculations concurrently
        tasks = [
            self.calculate_ssim(image1, image2),
            self.calculate_psnr(image1, image2),
            self.calculate_lpips(image1, image2)
        ]
        
        results = await asyncio.gather(*tasks)
        return results[0], results[1], results[2]
    
    def validate_images(
        self,
        image1: torch.Tensor,
        image2: torch.Tensor
    ) -> bool:
        """Validate that images are compatible for comparison"""
        
        # Check dimensions
        if image1.shape != image2.shape:
            logger.warning(f"Image shapes don't match: {image1.shape} vs {image2.shape}")
            return False
        
        # Check value range
        if image1.min() < 0 or image1.max() > 1:
            logger.warning(f"Image1 values out of range: [{image1.min()}, {image1.max()}]")
            return False
        
        if image2.min() < 0 or image2.max() > 1:
            logger.warning(f"Image2 values out of range: [{image2.min()}, {image2.max()}]")
            return False
        
        return True