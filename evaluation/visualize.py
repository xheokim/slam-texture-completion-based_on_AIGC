"""
模块⑥ 可视化 — 生成论文级对比图
"""

import os
import sys
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from evaluation import config
from evaluation.render_compare import load_and_prepare_meshes
from generation.renderer import load_poses


def create_comparison_figure(rgb_gt, rendered_orig, rendered_completed,
                              defect_2d, zoom_bbox=None):
    """
    创建单帧对比图（4 列或 5 列）：
      GT | 原始网格 | 补全网格 | 缺陷掩膜 | [放大区域]

    Returns: PIL Image
    """
    h, w = rgb_gt.shape[:2]

    # 统一尺寸
    def to_pil(img):
        if img.ndim == 3:
            return Image.fromarray(img, mode="RGB")
        return Image.fromarray(img, mode="L")

    img_gt = to_pil(rgb_gt)
    img_orig = to_pil(rendered_orig)
    img_comp = to_pil(rendered_completed)

    # 缺陷掩膜：半透明红色覆盖在 GT 上
    defect_overlay = to_pil(rgb_gt.copy())
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    for yy in range(0, h, 2):
        for xx in range(0, w, 2):
            if defect_2d[yy, xx]:
                draw.rectangle([xx, yy, xx + 1, yy + 1],
                              fill=(255, 50, 50, 120))
    defect_overlay = defect_overlay.convert("RGBA")
    defect_overlay.paste(overlay, (0, 0), overlay)

    # 拼接: GT | 缺陷标记 | 原始 | 补全
    gap = 4
    total_w = w * 4 + gap * 3
    canvas = Image.new("RGB", (total_w, h + 30), (255, 255, 255))

    canvas.paste(img_gt, (0, 0))
    canvas.paste(defect_overlay.convert("RGB"), (w + gap, 0))
    canvas.paste(img_orig, (2 * (w + gap), 0))
    canvas.paste(img_comp, (3 * (w + gap), 0))

    # 标签
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except Exception:
        font = ImageFont.load_default()

    for i, label in enumerate(["GT Keyframe", "Defect Region", "Original Mesh", "Completed Mesh"]):
        x = i * (w + gap) + w // 2 - 50
        canvas_draw = ImageDraw.Draw(canvas)
        canvas_draw.text((x, h + 5), label, fill=(0, 0, 0), font=font)

    if zoom_bbox is not None:
        # 在对比图底部追加放大区域
        x1, y1, x2, y2 = zoom_bbox
        zoom_w, zoom_h = x2 - x1, y2 - y1
        zoom_cols = []
        for src in [img_gt, defect_overlay.convert("RGB"), img_orig, img_comp]:
            crop = src.crop((x1, y1, x2, y2))
            crop = crop.resize((zoom_w * 2, zoom_h * 2), Image.LANCZOS)
            zoom_cols.append(crop)

        zoom_canvas = Image.new("RGB", (total_w, zoom_h * 2 + 25), (255, 255, 255))
        for i, zc in enumerate(zoom_cols):
            zoom_canvas.paste(zc, (i * (w + gap), 0))
        zoom_draw = ImageDraw.Draw(zoom_canvas)
        for i, label in enumerate(["GT (Zoom)", "Defect (Zoom)", "Original (Zoom)", "Completed (Zoom)"]):
            x = i * (w + gap) + w // 2 - 60
            zoom_draw.text((x, zoom_h * 2 + 3), label, fill=(0, 0, 0), font=font)

        # 垂直拼接
        final = Image.new("RGB", (total_w, h + 30 + zoom_h * 2 + 30), (255, 255, 255))
        final.paste(canvas, (0, 0))
        final.paste(zoom_canvas, (0, h + 40))
        canvas = final

    return canvas


def find_defect_bbox(defect_2d, margin=20):
    """找到缺陷掩膜的最大连通区域边界框"""
    rows, cols = np.where(defect_2d)
    if len(rows) < 100:
        return None
    y1, y2 = rows.min(), rows.max()
    x1, x2 = cols.min(), cols.max()
    # 扩展 margin
    h, w = defect_2d.shape
    y1 = max(0, y1 - margin)
    y2 = min(h, y2 + margin)
    x1 = max(0, x1 - margin)
    x2 = min(w, x2 + margin)
    return (x1, y1, x2, y2)


def create_summary_figure(all_results, output_path):
    """
    创建 3 行 × 4 列的论文级总结对比图。
    每行一个视角：GT | 缺陷标记 | 原始 | 补全
    """
    n_rows = min(3, len(all_results))
    n_cols = 4
    w, h = config.WIDTH, config.HEIGHT
    gap = 4
    label_h = 24

    total_w = w * n_cols + gap * (n_cols - 1)
    total_h = h * n_rows + gap * (n_rows - 1) + label_h * n_rows
    canvas = Image.new("RGB", (total_w, total_h), (255, 255, 255))

    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except Exception:
        font = ImageFont.load_default()

    for row_idx, data in enumerate(all_results[:n_rows]):
        y_offset = row_idx * (h + label_h + gap)

        # 放置 4 列
        imgs = [
            Image.fromarray(data["gt"], "RGB"),
            Image.fromarray(data["defect_overlay"], "RGB"),
            Image.fromarray(data["original"]),
            Image.fromarray(data["completed"]),
        ]
        labels = ["GT Keyframe", "Defect Mask", "Original Mesh", "Completed Mesh"]

        for col_idx, (img, label) in enumerate(zip(imgs, labels)):
            x = col_idx * (w + gap)
            canvas.paste(img, (x, y_offset))
            draw = ImageDraw.Draw(canvas)
            draw.text((x + 5, y_offset + h + 2), label, fill=(0, 0, 0), font=font)

    canvas.save(output_path)
    print(f"  Summary saved: {output_path}")


def run_visualization(
    mesh_paths=None,
    defect_mask_path=config.DEFECT_MASK_PATH,
    kf_dir=config.KF_DIR,
    output_dir=config.OUTPUT_DIR,
):
    if mesh_paths is None:
        mesh_paths = config.MESH_PATHS

    print("=" * 60)
    print("Module 6: Visualization - Generate Thesis Comparison Figures")
    print("=" * 60)

    # Load
    print("\nLoading meshes...")
    renderers = load_and_prepare_meshes(mesh_paths)
    defect_mask = np.load(defect_mask_path)
    poses = load_poses(os.path.join(kf_dir, "poses.txt"))
    rgb_files = sorted(os.listdir(os.path.join(kf_dir, "rgb")))
    n_frames = min(len(rgb_files), len(poses))

    # Select best viewpoints (those with most defect pixels visible)
    print("\nSelecting best viewpoints...")
    # Use the original renderer to find views with many defect pixels
    orig_renderer = renderers["original"]
    view_scores = []
    for frame_i in range(0, n_frames, config.SAMPLE_EVERY_N):
        T_c2w = poses[frame_i]
        result = orig_renderer.render(T_c2w, defect_mask)
        n_defect = result["defect_mask"].sum()
        view_scores.append((frame_i, n_defect, T_c2w))

    # Top views by defect pixel count
    view_scores.sort(key=lambda x: -x[1])
    best_views = view_scores[:8]  # select top 8
    print(f"  Selected {len(best_views)} best viewpoints")

    # Generate comparison images for each
    print("\nGenerating comparison figures...")
    all_data = []
    best_defect_2d = None
    for rank, (frame_i, n_defect, T_c2w) in enumerate(best_views):
        rgb_path = os.path.join(kf_dir, "rgb", rgb_files[frame_i])
        rgb_gt = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)

        # Render original mesh
        orig_result = renderers["original"].render(T_c2w, defect_mask)
        orig_img = orig_result["color"]
        defect_2d = orig_result["defect_mask"]

        # Save best defect mask for best comparison
        if best_defect_2d is None:
            best_defect_2d = defect_2d.copy()

        # Render completed mesh
        comp_result = renderers["completed"].render(T_c2w, defect_mask)
        comp_img = comp_result["color"]

        # Create defect overlay
        defect_overlay = rgb_gt.copy()
        defect_overlay[defect_2d] = (
            defect_overlay[defect_2d] * 0.5 + np.array([255, 50, 50]) * 0.5
        ).astype(np.uint8)

        # Find zoom bbox
        zoom_bbox = find_defect_bbox(defect_2d)

        # Create comparison
        comparison = create_comparison_figure(
            rgb_gt, orig_img, comp_img, defect_2d, zoom_bbox
        )

        # Save individual comparison
        save_path = os.path.join(output_dir, f"compare_view_{frame_i}.png")
        comparison.save(save_path)

        # Collect for summary
        all_data.append({
            "gt": rgb_gt,
            "defect_overlay": defect_overlay,
            "original": orig_img,
            "completed": comp_img,
        })

        print(f"  [{rank + 1}/{len(best_views)}] Frame {frame_i}: "
              f"defect pixels {n_defect}, saved")

    # Create summary figure
    print("\nGenerating summary figure...")
    summary_path = os.path.join(output_dir, "comparison_summary.png")
    create_summary_figure(all_data, summary_path)

    # Also save the best before/after single comparison
    # Best = first in sorted list
    best_data = all_data[0]
    best_comparison = create_comparison_figure(
        best_data["gt"], best_data["original"], best_data["completed"],
        defect_2d=best_defect_2d
    )
    best_path = os.path.join(output_dir, "comparison_best.png")
    best_comparison.save(best_path)

    print(f"\nAll comparison figures saved to: {output_dir}/")
    print(f"  comparison_summary.png  - Multi-view summary figure")
    print(f"  comparison_best.png     - Best viewpoint comparison")
    for frame_i, _, _ in best_views:
        print(f"  compare_view_{frame_i}.png")


if __name__ == "__main__":
    run_visualization()
