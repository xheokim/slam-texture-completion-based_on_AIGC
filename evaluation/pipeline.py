"""
模块⑥ 主流程 — 评估 Pipeline
"""

import os
import sys
import time
import cv2
import numpy as np
import open3d as o3d

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from evaluation import config
from evaluation.render_compare import (load_and_prepare_meshes,
                                        evaluate_viewpoint,
                                        aggregate_results)
from generation.renderer import load_poses


def run_evaluation(
    mesh_paths=None,
    defect_mask_path=config.DEFECT_MASK_PATH,
    kf_dir=config.KF_DIR,
    report_path=config.REPORT_PATH,
):
    if mesh_paths is None:
        mesh_paths = config.MESH_PATHS

    print("=" * 60)
    print("Module 6: Evaluation - PSNR / SSIM / Region Analysis")
    print("=" * 60)

    # ── 1. Load data ──
    print("\n[1/4] Loading meshes...")
    renderers = load_and_prepare_meshes(mesh_paths)

    print("\n[2/4] Loading defect mask and keyframes...")
    defect_mask = np.load(defect_mask_path)
    poses = load_poses(os.path.join(kf_dir, "poses.txt"))
    rgb_files = sorted(os.listdir(os.path.join(kf_dir, "rgb")))
    n_frames = min(len(rgb_files), len(poses))
    print(f"  Keyframes: {n_frames}")

    # ── 2. Evaluate per viewpoint ──
    print(f"\n[3/4] Evaluating (every {config.SAMPLE_EVERY_N} frames)...")

    all_results = []
    t_start = time.time()
    sample_indices = list(range(0, n_frames, config.SAMPLE_EVERY_N))

    for idx, frame_i in enumerate(sample_indices):
        T_c2w = poses[frame_i]

        # 加载 ground truth RGB
        rgb_path = os.path.join(kf_dir, "rgb", rgb_files[frame_i])
        rgb_gt = cv2.imread(rgb_path)
        if rgb_gt is None:
            continue
        rgb_gt = cv2.cvtColor(rgb_gt, cv2.COLOR_BGR2RGB)

        # 评估
        view_results = evaluate_viewpoint(
            frame_i, T_c2w, renderers, rgb_gt, defect_mask
        )

        # 保存帧级结果
        view_results["frame_index"] = frame_i
        all_results.append(view_results)

        if (idx + 1) % 5 == 0:
            elapsed = time.time() - t_start
            print(f"  {idx + 1}/{len(sample_indices)} frames "
                  f"({elapsed:.1f}s)")

    elapsed = time.time() - t_start
    print(f"  Completed {len(all_results)} frames in {elapsed:.1f}s")

    # ── 3. Aggregate ──
    print("\n[4/4] Generating report...")
    summary = aggregate_results(all_results)

    # ── 4. 输出报告 ──
    report = format_report(summary, mesh_paths.keys(), len(all_results))
    print(report)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\nReport saved: {report_path}")

    return summary


def format_report(summary, mesh_names, n_views):
    """格式化评估报告"""
    lines = []
    lines.append("=" * 72)
    lines.append("  SLAM Map Texture Completion - Evaluation Report")
    lines.append("=" * 72)
    lines.append(f"  Viewpoints evaluated: {n_views}")
    lines.append(f"  Meshes evaluated:   {', '.join(mesh_names)}")
    lines.append("")

    # Table header
    header = f"{'Metric':<35} {'PSNR(dB)':>10} {'+/-std':>8}  {'SSIM':>8} {'+/-std':>8}"
    lines.append("-" * 72)
    lines.append(header)
    lines.append("-" * 72)

    for key, stats in sorted(summary.items()):
        if stats["count"] == 0:
            continue

        psnr_s = f"{stats['psnr_mean']:.1f}" if stats['psnr_mean'] else "N/A"
        psnr_std = f"{stats['psnr_std']:.1f}" if stats.get('psnr_std') else ""
        ssim_s = f"{stats['ssim_mean']:.4f}" if stats['ssim_mean'] else "N/A"
        ssim_std = f"{stats['ssim_std']:.4f}" if stats.get('ssim_std') else ""

        lines.append(
            f"{key:<35} {psnr_s:>10} {psnr_std:>8}  {ssim_s:>8} {ssim_std:>8}"
        )

    lines.append("-" * 72)
    lines.append("")

    # Key findings
    lines.append("[Key Comparisons]")
    lines.append("")

    # original vs completed full image PSNR
    for mesh in mesh_names:
        key_full = f"{mesh}_full"
        key_defect = f"{mesh}_defect"
        key_normal = f"{mesh}_normal"

        if key_full in summary and summary[key_full]["psnr_mean"]:
            psnr_full = summary[key_full]["psnr_mean"]
            ssim_full = summary[key_full]["ssim_mean"]
            lines.append(
                f"  {mesh} full:      PSNR={psnr_full:.1f} dB, "
                f"SSIM={ssim_full:.4f}"
            )

        if key_defect in summary and summary[key_defect]["psnr_mean"]:
            psnr_d = summary[key_defect]["psnr_mean"]
            ssim_d = summary[key_defect]["ssim_mean"]
            lines.append(
                f"  {mesh} defect:    PSNR={psnr_d:.1f} dB, "
                f"SSIM={ssim_d:.4f}"
            )

        if key_normal in summary and summary[key_normal]["psnr_mean"]:
            psnr_n = summary[key_normal]["psnr_mean"]
            ssim_n = summary[key_normal]["ssim_mean"]
            lines.append(
                f"  {mesh} non-defect: PSNR={psnr_n:.1f} dB, "
                f"SSIM={ssim_n:.4f}"
            )
        lines.append("")

    lines.append("=" * 72)

    return "\n".join(lines)


if __name__ == "__main__":
    run_evaluation()
