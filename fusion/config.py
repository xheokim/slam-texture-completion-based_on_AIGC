"""
模块⑤ 纹理融合 — 配置
"""

import os

# ===== 路径 =====
OUTPUT_DIR = r"D:\LearnPython\finalproject\outputs"
TEXTURED_MESH_PATH = os.path.join(OUTPUT_DIR, "scene_textured.ply")
DEFECT_MASK_PATH = os.path.join(OUTPUT_DIR, "defect_mask.npy")
ACCUMULATOR_PATH = os.path.join(OUTPUT_DIR, "texture_accum_final.npz")
FINAL_MESH_PATH = os.path.join(OUTPUT_DIR, "scene_final.ply")

# ===== 离群值过滤 =====
MIN_OBSERVATIONS = 3         # 至少被观察到这么多次才信任颜色
OUTLIER_STD_THRESH = 2.5     # 颜色偏离邻域均值超过N倍标准差 → 离群

# ===== 双边滤波 =====
BILATERAL_ITERS = 3          # 迭代次数
SIGMA_SPATIAL = 0.8          # 空间核宽度（面片邻域跳数）
SIGMA_COLOR = 0.12           # 颜色核宽度（RGB空间）
BILATERAL_RING = 2           # 邻域环数

# ===== 梯度域边界优化 =====
POISSON_BOUNDARY_RINGS = 3   # 优化区域：边界外N环
POISSON_LAMBDA = 0.5         # 正则化强度

# ===== 颜色协调 =====
HARMONIZE_STRENGTH = 0.4     # 协调混合强度（0=不变, 1=完全匹配）
