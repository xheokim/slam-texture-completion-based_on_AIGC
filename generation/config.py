"""
模块④ 纹理生成 — 配置
"""

import os

# ===== 路径 =====
DATA_DIR = r"D:\LearnPython\finalproject\data\fr2_desk\rgbd_dataset_freiburg2_desk"
KF_DIR = os.path.join(DATA_DIR, "keyframes")
OUTPUT_DIR = r"D:\LearnPython\finalproject\outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MESH_PATH = os.path.join(OUTPUT_DIR, "scene_mesh.ply")
DEFECT_MASK_PATH = os.path.join(OUTPUT_DIR, "defect_mask.npy")
TEXTURED_MESH_PATH = os.path.join(OUTPUT_DIR, "scene_textured.ply")

# ===== 相机内参 (TUM2) =====
FX, FY = 520.908620, 521.007327
CX, CY = 325.141442, 249.701764
WIDTH, HEIGHT = 640, 480

# ===== 扩散模型 =====
MODEL_ID = "runwayml/stable-diffusion-v1-5"
CONTROLNET_ID = "lllyasviel/sd-controlnet-depth"

# Stable Diffusion ControlNet Inpaint pipeline
PROMPT = (
    "photorealistic indoor office desk scene, wooden table surface, "
    "computer monitor, keyboard, mouse, papers, books, cluttered workspace, "
    "fluorescent ceiling lighting, sharp focus, high detail"
)
NEGATIVE_PROMPT = (
    "blurry, distorted, unrealistic, cartoon, painting, watermark, text, "
    "low quality, jpeg artifacts, distorted perspective"
)
NUM_INFERENCE_STEPS = 20
GUIDANCE_SCALE = 7.5
CONTROLNET_CONDITIONING_SCALE = 0.85
GENERATION_RESOLUTION = 512  # process at 512x384 then pad to 512x512

# ===== 视角选择 =====
MAX_VIEWPOINTS = 60
VIEWPOINT_SAMPLE_STRIDE = 4  # fallback: every 4th keyframe
MIN_DEFECT_PIXELS = 100       # skip viewpoint if fewer defect pixels visible

# ===== 融合参数 =====
ANGLE_WEIGHT_POWER = 1.0      # pow(cos_angle, power) for frontal-view weighting
DISTANCE_EPS = 0.1            # epsilon for distance weighting 1/(dist+eps)^2
CHECKPOINT_EVERY = 10         # save accumulator state every N views

# ===== 边界平滑 =====
BOUNDARY_SMOOTH_ITERATIONS = 5
BOUNDARY_RING_COUNT = 2       # smooth faces within N rings of defect boundary
