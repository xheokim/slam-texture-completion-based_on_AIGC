"""
模块⑥ 评估 — 配置
"""

import os

# ===== 路径 =====
OUTPUT_DIR = r"D:\LearnPython\finalproject\outputs"
KF_DIR = r"D:\LearnPython\finalproject\data\fr2_desk\rgbd_dataset_freiburg2_desk\keyframes"
DEFECT_MASK_PATH = os.path.join(OUTPUT_DIR, "defect_mask.npy")
REPORT_PATH = os.path.join(OUTPUT_DIR, "evaluation_report.txt")

# 需要评估的网格
MESH_PATHS = {
    "original": os.path.join(OUTPUT_DIR, "scene_mesh.ply"),
    "completed": os.path.join(OUTPUT_DIR, "scene_final.ply"),
}

# ===== 相机内参 =====
FX, FY = 520.908620, 521.007327
CX, CY = 325.141442, 249.701764
WIDTH, HEIGHT = 640, 480

# ===== 评估参数 =====
SAMPLE_EVERY_N = 10     # 每N帧评估一次
MIN_DEFECT_PIXELS = 50  # 跳过缺陷像素太少的视角

# SSIM
SSIM_K1, SSIM_K2 = 0.01, 0.03
SSIM_WIN_SIZE = 11
