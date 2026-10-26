"""
模块④ 主流程 — 纹理生成 Pipeline
"""

import os
import sys
import time
import numpy as np
import open3d as o3d

# 确保项目根目录在 Python 路径中
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# 修复 Windows Anaconda SSL 证书路径问题
import certifi
os.environ.setdefault("SSL_CERT_FILE", certifi.where())
# 使用 HF 镜像（国内加速）
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from generation import config
from generation.renderer import MeshRenderer, load_poses
from generation.inpainter import DepthControlNetInpainter, create_defect_mask_image
from generation.backproject import ColorAccumulator
from generation.blender import TextureBlender


def select_viewpoints(poses, n_frames, max_views, sample_stride):
    """均匀采样视角（简单策略）"""
    indices = list(range(0, n_frames, sample_stride))
    if len(indices) > max_views:
        step = max(1, len(indices) // max_views)
        indices = indices[::step][:max_views]
    return indices


def run_texture_generation(
    mesh_path=config.MESH_PATH,
    defect_mask_path=config.DEFECT_MASK_PATH,
    kf_dir=config.KF_DIR,
    output_path=config.TEXTURED_MESH_PATH,
):
    print("=" * 60)
    print("Module 4: Texture Generation - View-Based Depth-Conditioned Inpainting")
    print("=" * 60)

    # ── 1. Load data ──
    print("\n[1/6] Loading mesh and defect mask...")
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    mesh.compute_vertex_normals()
    defect_mask = np.load(defect_mask_path)
    defect_face_array = defect_mask  # (n_faces,) bool
    n_faces = len(defect_mask)
    print(f"  Total faces: {n_faces}, defect faces: {defect_mask.sum()} "
          f"({100 * defect_mask.sum() / n_faces:.1f}%)")

    # ── 2. Load poses and keyframes ──
    print("\n[2/6] Loading poses and keyframes...")
    poses = load_poses(os.path.join(kf_dir, "poses.txt"))
    rgb_files = sorted(os.listdir(os.path.join(kf_dir, "rgb")))
    depth_files = sorted(os.listdir(os.path.join(kf_dir, "depth")))
    n_frames = min(len(rgb_files), len(depth_files), len(poses))
    print(f"  Available frames: {n_frames}")

    # ── 3. Select viewpoints ──
    print("\n[3/6] Selecting viewpoints...")
    cam_intrinsic = np.array([
        [config.FX, 0, config.CX],
        [0, config.FY, config.CY],
        [0, 0, 1],
    ])
    viewpoint_indices = select_viewpoints(
        poses, n_frames,
        config.MAX_VIEWPOINTS, config.VIEWPOINT_SAMPLE_STRIDE
    )
    print(f"  Selected {len(viewpoint_indices)} viewpoints")

    # ── 4. Initialize components ──
    print("\n[4/6] Initializing components...")
    renderer = MeshRenderer(mesh, cam_intrinsic, config.WIDTH, config.HEIGHT)
    inpainter = DepthControlNetInpainter(
        config.MODEL_ID, config.CONTROLNET_ID,
        use_fp16=True
    )
    accumulator = ColorAccumulator(n_faces)

    # ── 5. Process per viewpoint ──
    print(f"\n[5/6] Processing {len(viewpoint_indices)} viewpoints...")
    total_pixels = 0
    skipped = 0
    t_start = time.time()

    for v_idx, frame_i in enumerate(viewpoint_indices):
        T_c2w = poses[frame_i]
        cam_pos = T_c2w[:3, 3]

        # 渲染
        render_result = renderer.render(T_c2w, defect_face_array)

        defect_2d = render_result["defect_mask"]
        n_defect_pixels = defect_2d.sum()

        if n_defect_pixels < config.MIN_DEFECT_PIXELS:
            skipped += 1
            continue

        # 准备输入图像
        from PIL import Image
        rgb_pil = Image.fromarray(render_result["color"], mode="RGB")
        mask_pil = create_defect_mask_image(defect_2d, dilate_pixels=2)
        depth_pil = renderer.create_depth_condition(render_result["depth"])

        # Inpaint
        print(f"  View {v_idx + 1}/{len(viewpoint_indices)} "
              f"(frame {frame_i}): defect pixels {n_defect_pixels} → inpainting...",
              end=" ", flush=True)

        try:
            inpainted = inpainter.inpaint(
                rgb_pil, mask_pil, depth_pil,
                prompt=config.PROMPT,
                negative_prompt=config.NEGATIVE_PROMPT,
                num_steps=config.NUM_INFERENCE_STEPS,
                guidance_scale=config.GUIDANCE_SCALE,
                controlnet_conditioning_scale=config.CONTROLNET_CONDITIONING_SCALE,
            )
        except Exception as e:
            print(f"skipped (error: {e})")
            continue

        inpainted_np = np.array(inpainted, dtype=np.uint8)

        # Back-project and accumulate
        n_accum = accumulator.accumulate(
            render_result["face_id"],
            inpainted_np,
            defect_2d,
            render_result["depth"],
            render_result["normal"],
            cam_pos,
        )
        total_pixels += n_accum
        print(f"accumulated {n_accum} pixels")

        # Incremental checkpoint
        if (v_idx + 1) % config.CHECKPOINT_EVERY == 0:
            ckpt_path = os.path.join(config.OUTPUT_DIR,
                                     "texture_accum_checkpoint.npz")
            accumulator.save(ckpt_path)
            print(f"    [checkpoint saved: {ckpt_path}]")

    t_elapsed = time.time() - t_start
    print(f"\n  Done: {len(viewpoint_indices) - skipped} viewpoints processed, "
          f"skipped {skipped}, total pixels {total_pixels}, "
          f"elapsed {t_elapsed:.1f}s")

    # Save final accumulator
    acc_path = os.path.join(config.OUTPUT_DIR, "texture_accum_final.npz")
    accumulator.save(acc_path)
    print(f"  Accumulator saved: {acc_path}")

    # ── 6. Fusion + save ──
    print("\n[6/6] Multi-view fusion + boundary smoothing...")
    blender = TextureBlender(mesh, defect_mask)
    textured_mesh = blender.apply(accumulator)

    o3d.io.write_triangle_mesh(output_path, textured_mesh)
    print(f"\nTextured mesh saved: {output_path}")

    # Statistics
    observed_defects = (accumulator.weight_sum > 0).sum()
    print(f"Repaired defect faces: {observed_defects} / "
          f"{defect_mask.sum()}")

    return textured_mesh


if __name__ == "__main__":
    run_texture_generation()
