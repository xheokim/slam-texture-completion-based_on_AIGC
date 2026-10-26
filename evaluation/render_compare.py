"""
模块⑥ 渲染比较 — 从关键帧视角渲染网格并与真实图像对比
"""

import os
import sys
import cv2
import numpy as np

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from generation.renderer import MeshRenderer, load_poses


def load_and_prepare_meshes(mesh_paths):
    """加载所有要评估的网格，返回渲染器字典"""
    import open3d as o3d

    cam_intrinsic = np.array([
        [520.908620, 0, 325.141442],
        [0, 521.007327, 249.701764],
        [0, 0, 1],
    ])

    renderers = {}
    for name, path in mesh_paths.items():
        mesh = o3d.io.read_triangle_mesh(path)
        mesh.compute_vertex_normals()
        renderers[name] = MeshRenderer(mesh, cam_intrinsic, 640, 480)
        print(f"  Loaded: {name} ({len(mesh.vertices)} vertices)")

    return renderers


def evaluate_viewpoint(frame_i, T_c2w, renderers, rgb_gt, defect_mask_2d):
    """
    评估单个视角：

    Returns: dict with metrics for each mesh variant, per region
    """
    results = {}

    for mesh_name, renderer in renderers.items():
        # 渲染
        render_result = renderer.render(T_c2w, defect_mask_2d)
        rendered_img = render_result["color"]

        # 全图指标
        full = {
            f"{mesh_name}_full": {"psnr": None, "ssim": None, "pixels": 0}
        }

        # 缺陷区域指标
        defect = {
            f"{mesh_name}_defect": {"psnr": None, "ssim": None, "pixels": 0}
        }

        # 非缺陷区域
        normal = {
            f"{mesh_name}_normal": {"psnr": None, "ssim": None, "pixels": 0}
        }

        # 全图
        from evaluation.metrics import compute_metrics
        full_metrics = compute_metrics(rgb_gt, rendered_img)
        full[f"{mesh_name}_full"] = full_metrics

        # 缺陷区域
        if defect_mask_2d.sum() >= 10:
            # 使用渲染时返回的缺陷掩膜（该mesh在该视角下的缺陷面投影）
            render_defect = render_result["defect_mask"]
            if render_defect.sum() >= 10:
                defect_metrics = compute_metrics(
                    rgb_gt, rendered_img, render_defect
                )
                defect[f"{mesh_name}_defect"] = defect_metrics

            # 非缺陷区域
            normal_mask = ~render_defect & (render_result["face_id"] != 0xFFFFFFFF)
            if normal_mask.sum() >= 10:
                normal_metrics = compute_metrics(
                    rgb_gt, rendered_img, normal_mask
                )
                normal[f"{mesh_name}_normal"] = normal_metrics

        results.update(full)
        results.update(defect)
        results.update(normal)

    return results


def aggregate_results(all_view_results):
    """汇总所有视角的结果 → 平均值 ± 标准差"""
    # 收集指标
    metric_keys = set()
    for vr in all_view_results:
        metric_keys.update(vr.keys())

    summary = {}
    for key in sorted(metric_keys):
        vals = [vr[key] for vr in all_view_results if isinstance(vr.get(key), dict)]
        if not vals:
            continue
        psnrs = [v["psnr"] for v in vals if v.get("psnr") is not None]
        ssims = [v["ssim"] for v in vals if v.get("ssim") is not None]
        total_pix = sum(v.get("pixels", 0) for v in vals)

        summary[key] = {
            "count": len(psnrs),
            "psnr_mean": round(np.mean(psnrs), 2) if psnrs else None,
            "psnr_std": round(np.std(psnrs), 2) if len(psnrs) > 1 else None,
            "ssim_mean": round(np.mean(ssims), 4) if ssims else None,
            "ssim_std": round(np.std(ssims), 4) if len(ssims) > 1 else None,
            "total_pixels": total_pix,
        }

    return summary
