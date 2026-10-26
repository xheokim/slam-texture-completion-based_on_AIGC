"""
模块⑥ 指标 — PSNR, SSIM, ICP
"""

import numpy as np
from scipy.ndimage import uniform_filter


def psnr(img1, img2, max_val=255.0):
    """Peak Signal-to-Noise Ratio"""
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse < 1e-10:
        return 100.0
    return 20.0 * np.log10(max_val) - 10.0 * np.log10(mse)


def ssim(img1, img2, max_val=255.0, K1=0.01, K2=0.03, win_size=11):
    """
    Structural Similarity Index (简化实现)

    基于: Wang et al. "Image Quality Assessment: From Error Visibility
    to Structural Similarity" (TIP 2004)
    """
    C1 = (K1 * max_val) ** 2
    C2 = (K2 * max_val) ** 2

    # 灰度化
    if img1.ndim == 3:
        g1 = np.mean(img1, axis=2)
        g2 = np.mean(img2, axis=2)
    else:
        g1, g2 = img1.astype(np.float64), img2.astype(np.float64)

    # 均值
    mu1 = uniform_filter(g1, size=win_size)
    mu2 = uniform_filter(g2, size=win_size)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    # 方差
    sigma1_sq = uniform_filter(g1 ** 2, size=win_size) - mu1_sq
    sigma2_sq = uniform_filter(g2 ** 2, size=win_size) - mu2_sq
    sigma12 = uniform_filter(g1 * g2, size=win_size) - mu1_mu2

    sigma1_sq = np.maximum(sigma1_sq, 0)
    sigma2_sq = np.maximum(sigma2_sq, 0)

    # SSIM
    numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)

    ssim_map = numerator / denominator
    return np.mean(ssim_map)


def compute_metrics(img_gt, img_render, mask=None):
    """
    计算两张图像之间的所有指标。

    img_gt:      (H, W, 3) uint8, ground truth
    img_render:  (H, W, 3) uint8, rendered/completed
    mask:        (H, W) bool, 评估区域 (None = 全图)

    Returns dict with PSNR, SSIM, num_pixels
    """
    if mask is not None and mask.sum() < 10:
        return {"psnr": None, "ssim": None, "pixels": mask.sum()}

    if mask is not None:
        gt = img_gt[mask]
        rd = img_render[mask]
    else:
        gt = img_gt.reshape(-1, 3)
        rd = img_render.reshape(-1, 3)

    # 全图 SSIM 用滑动窗口
    if mask is None:
        ssim_val = ssim(img_gt, img_render, max_val=255.0)
    else:
        # 掩膜区域：逐像素 PSNR, 简化 SSIM
        ssim_val = ssim_patch(gt, rd, max_val=255.0)

    psnr_val = psnr(gt, rd, max_val=255.0)

    return {
        "psnr": round(psnr_val, 2),
        "ssim": round(ssim_val, 4),
        "pixels": len(gt) if mask is not None else img_gt.size // 3,
    }


def ssim_patch(pixels_gt, pixels_rd, max_val=255.0, K1=0.01, K2=0.03):
    """简化 SSIM：对像素集合计算"""
    C1 = (K1 * max_val) ** 2
    C2 = (K2 * max_val) ** 2

    gt = pixels_gt.astype(np.float64)
    rd = pixels_rd.astype(np.float64)

    mu1 = gt.mean()
    mu2 = rd.mean()
    sigma1_sq = gt.var()
    sigma2_sq = rd.var()
    sigma12 = np.mean((gt - mu1) * (rd - mu2))

    numerator = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2)

    return numerator / denominator
