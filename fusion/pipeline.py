"""
模块⑤ 主流程 — 纹理融合优化
"""

import os
import sys
import numpy as np
import open3d as o3d

# 项目根目录
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from fusion import config
from fusion.optimize import (build_face_adjacency, remove_outliers,
                              bilateral_filter_mesh)
from fusion.boundary import gradient_domain_blend
from fusion.harmonize import color_harmonize


def run_fusion_pipeline(
    mesh_path=config.TEXTURED_MESH_PATH,
    defect_mask_path=config.DEFECT_MASK_PATH,
    accumulator_path=config.ACCUMULATOR_PATH,
    output_path=config.FINAL_MESH_PATH,
):
    print("=" * 60)
    print("Module 5: Texture Fusion - Mesh-Based Optimization Pipeline")
    print("=" * 60)

    # ── 1. Load data ──
    print("\n[1/5] Loading mesh and masks...")
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    mesh.compute_vertex_normals()
    defect_mask = np.load(defect_mask_path)
    n_faces = len(defect_mask)

    # 加载累积器以获得权重
    accum_data = np.load(accumulator_path)
    weight_sum = accum_data["weight_sum"]
    sample_count = accum_data["sample_count"]

    print(f"  Faces: {n_faces}, defects: {defect_mask.sum()}")
    print(f"  Observed defect faces: {(weight_sum > 0).sum()}")
    print(f"  Mean observations: {sample_count[sample_count > 0].mean():.1f}")

    # ── Build adjacency ──
    print("\n[2/5] Building face adjacency...")
    triangles = np.asarray(mesh.triangles)
    adjacency = build_face_adjacency(triangles, n_faces)

    # 计算面片中心
    vertices = np.asarray(mesh.vertices)
    face_centers = np.mean(vertices[triangles], axis=1)

    # 当前面片颜色（从顶点颜色平均）
    vert_colors = np.asarray(mesh.vertex_colors)
    face_colors = np.zeros((n_faces, 3))
    for fi, tri in enumerate(triangles):
        face_colors[fi] = np.mean(vert_colors[list(tri)], axis=0)
    print(f"  Face color range: {face_colors.min():.3f} ~ {face_colors.max():.3f}")

    # ── 3. Outlier removal + bilateral filter ──
    print("\n[3/5] Removing outliers...")
    face_colors = remove_outliers(
        face_colors, adjacency, weight_sum, defect_mask,
        min_obs=config.MIN_OBSERVATIONS,
        std_thresh=config.OUTLIER_STD_THRESH,
    )

    print("Bilateral filtering...")
    face_colors = bilateral_filter_mesh(
        face_colors, adjacency, face_centers, defect_mask,
        iters=config.BILATERAL_ITERS,
        sigma_spatial=config.SIGMA_SPATIAL,
        sigma_color=config.SIGMA_COLOR,
        ring=config.BILATERAL_RING,
    )

    # ── 4. Gradient-domain boundary optimization ──
    print("\n[4/5] Gradient-domain boundary optimization...")
    face_colors = gradient_domain_blend(
        face_colors, adjacency, defect_mask,
        boundary_rings=config.POISSON_BOUNDARY_RINGS,
        lambda_reg=config.POISSON_LAMBDA,
    )

    # ── 5. Color harmonization ──
    print("\n[5/5] Color harmonization...")
    face_colors = color_harmonize(
        face_colors, defect_mask, adjacency,
        strength=config.HARMONIZE_STRENGTH,
    )

    # ── Save ──
    print("\nSaving final mesh...")
    face_colors = np.clip(face_colors, 0.0, 1.0)

    # 面片颜色 → 顶点颜色
    vert_sum = np.zeros((len(vertices), 3), dtype=np.float64)
    vert_count = np.zeros(len(vertices), dtype=np.int32)
    for fi, tri in enumerate(triangles):
        for vi in tri:
            vert_sum[vi] += face_colors[fi]
            vert_count[vi] += 1
    vert_count[vert_count == 0] = 1
    vert_colors = vert_sum / vert_count[:, None]

    mesh.vertex_colors = o3d.utility.Vector3dVector(vert_colors)
    mesh.compute_vertex_normals()

    o3d.io.write_triangle_mesh(output_path, mesh)
    print(f"  Final mesh saved: {output_path}")

    # ── Statistics ──
    gen_colors = face_colors[defect_mask]
    orig_colors = face_colors[~defect_mask]
    print(f"\nGenerated region: mean RGB=({gen_colors.mean(axis=0)[0]:.3f}, "
          f"{gen_colors.mean(axis=0)[1]:.3f}, {gen_colors.mean(axis=0)[2]:.3f})")
    print(f"Original region: mean RGB=({orig_colors.mean(axis=0)[0]:.3f}, "
          f"{orig_colors.mean(axis=0)[1]:.3f}, {orig_colors.mean(axis=0)[2]:.3f})")
    print(f"Color diff L2: {np.linalg.norm(gen_colors.mean(axis=0) - orig_colors.mean(axis=0)):.4f}")

    return mesh


if __name__ == "__main__":
    run_fusion_pipeline()
