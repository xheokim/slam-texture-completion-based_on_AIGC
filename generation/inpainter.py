"""
模块④ 修复器 — ControlNet + Stable Diffusion 深度条件修复
"""

import torch
import numpy as np
from PIL import Image, ImageFilter


class DepthControlNetInpainter:
    def __init__(self, model_id, controlnet_id, device="cuda", use_fp16=True):
        self.device = device

        print("Loading ControlNet...")
        from diffusers import ControlNetModel

        dtype = torch.float16 if use_fp16 else torch.float32
        self.controlnet = ControlNetModel.from_pretrained(
            controlnet_id, torch_dtype=dtype
        )

        print("Loading Stable Diffusion inpainting pipeline...")
        from diffusers import StableDiffusionControlNetInpaintPipeline

        self.pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
            model_id,
            controlnet=self.controlnet,
            torch_dtype=dtype,
            safety_checker=None,
        )
        self.pipe.to(device)

        # 8GB 显存优化
        self.pipe.enable_attention_slicing()
        try:
            self.pipe.vae.enable_slicing()
            self.pipe.vae.enable_tiling()
        except Exception:
            pass

        self.use_fp16 = use_fp16
        self.target_size = 512
        print("Model loaded.")

    def inpaint(self, rgb_image, mask_image, depth_image,
                prompt, negative_prompt="",
                num_steps=20, guidance_scale=7.5,
                controlnet_conditioning_scale=0.85, strength=1.0):
        """
        rgb_image:   PIL RGB, 原始分辨率 (640x480)
        mask_image:  PIL "L", 255=需要修复的区域, 0=保留
        depth_image: PIL RGB, 深度条件图 (640x480)

        Returns: PIL RGB, 修复后的图像 (640x480)
        """
        orig_w, orig_h = rgb_image.size

        # 预处理 → 512x512
        rgb_512, mask_512, depth_512 = self._preprocess(
            rgb_image, mask_image, depth_image
        )

        # 检查是否有有效 mask
        mask_arr = np.array(mask_512)
        if mask_arr.max() < 128:
            # 无缺陷区域，直接返回原图
            return rgb_image

        # 运行扩散修复
        with torch.inference_mode():
            result = self.pipe(
                image=rgb_512,
                mask_image=mask_512,
                control_image=depth_512,
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_steps,
                guidance_scale=guidance_scale,
                controlnet_conditioning_scale=controlnet_conditioning_scale,
                strength=strength,
                height=512,
                width=512,
            ).images[0]

        # 后处理 → 原始分辨率
        return self._postprocess(result, orig_w, orig_h)

    def _preprocess(self, rgb, mask, depth):
        """Resize + pad 到 512x512"""
        target = self.target_size

        # 保持宽高比，缩放使长边=512
        rgb_resized = rgb.resize((target, int(target * rgb.size[1] / rgb.size[0])))
        mask_resized = mask.resize((target, int(target * mask.size[1] / mask.size[0])),
                                   Image.NEAREST)
        depth_resized = depth.resize((target, int(target * depth.size[1] / depth.size[0])),
                                     Image.NEAREST)

        # Pad 到 512x512
        w, h = rgb_resized.size
        pad_bottom = target - h

        rgb_512 = Image.new("RGB", (target, target), (0, 0, 0))
        rgb_512.paste(rgb_resized, (0, 0))
        mask_512 = Image.new("L", (target, target), 0)
        mask_512.paste(mask_resized, (0, 0))
        depth_512 = Image.new("RGB", (target, target), (0, 0, 0))
        depth_512.paste(depth_resized, (0, 0))

        return rgb_512, mask_512, depth_512

    def _postprocess(self, result_pil, orig_w, orig_h):
        """从 512x512 裁剪回原始尺寸"""
        target = self.target_size
        h = int(target * orig_h / orig_w)
        result_pil = result_pil.crop((0, 0, target, h))
        return result_pil.resize((orig_w, orig_h), Image.LANCZOS)


def create_defect_mask_image(defect_mask_2d, dilate_pixels=2):
    """
    将 (H, W) bool 缺陷掩膜转为 PIL mask image (255=需修复)
    可选膨胀以覆盖缺陷边界的锯齿
    """
    mask = (defect_mask_2d * 255).astype(np.uint8)
    pil = Image.fromarray(mask, mode="L")
    if dilate_pixels > 0:
        pil = pil.filter(ImageFilter.MaxFilter(size=2 * dilate_pixels + 1))
    return pil
