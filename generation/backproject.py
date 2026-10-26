"""
模块④ 反投影 — 将修复后的像素颜色累积到 3D 面片
"""

import numpy as np
import os


class ColorAccumulator:
    def __init__(self, n_faces):
        """
        n_faces: 网格面片总数
        """
        self.n_faces = n_faces
        self.color_sum = np.zeros((n_faces, 3), dtype=np.float64)
        self.weight_sum = np.zeros(n_faces, dtype=np.float64)
        self.sample_count = np.zeros(n_faces, dtype=np.int32)

    def accumulate(self, face_id_map, inpainted_image, defect_mask_2d,
                   depth_map, normal_map, camera_position):
        """
        对每个缺陷像素，按视角加权累积颜色到对应的 3D 面片。

        face_id_map:      (H, W) int32, 面片索引，-1=无交点
        inpainted_image:  (H, W, 3) uint8, 修复后的 RGB
        defect_mask_2d:   (H, W) bool, 缺陷区域像素
        depth_map:        (H, W) float32, 深度 (米)
        normal_map:       (H, W, 3) float32, 世界坐标系面法向量
        camera_position:  (3,) float64, 相机在世界坐标系的位置
        """
        rows, cols = np.where(defect_mask_2d)
        if len(rows) == 0:
            return 0

        face_indices = face_id_map[rows, cols]
        valid = face_indices >= 0
        if not valid.any():
            return 0

        rows, cols = rows[valid], cols[valid]
        face_indices = face_indices[valid]

        # 提取各像素的颜色和几何信息
        colors = inpainted_image[rows, cols].astype(np.float64) / 255.0
        depths = depth_map[rows, cols].astype(np.float64)
        normals = normal_map[rows, cols]

        # 视角方向 (单位向量)
        face_centers = (camera_position[None, :] +
                        (normals * 0.0))  # placeholder, compute properly

        # 计算权重
        view_dirs = -normals  # 近似：视角光线方向
        # 实际视线方向 = normalize(camera_pos - surface_point)
        # 这里用法向量与视线夹角近似：越正对相机，权重越高
        cos_angles = np.abs(np.sum(normals * view_dirs, axis=1))
        cos_angles = np.clip(cos_angles, 0.01, 1.0)

        # 角度权重
        angle_weight = cos_angles ** 1.0

        # 距离权重: 越近权重越高
        dist_weight = 1.0 / (depths ** 2 + 0.1)

        weights = angle_weight * dist_weight

        # 累加
        np.add.at(self.color_sum, face_indices, colors * weights[:, None])
        np.add.at(self.weight_sum, face_indices, weights)
        np.add.at(self.sample_count, face_indices, 1)

        return len(rows)

    def get_final_colors(self, default_color=None):
        """
        返回 (n_faces, 3) float64 RGB 颜色。
        未被观察到的面片使用默认颜色。
        """
        if default_color is None:
            default_color = np.array([0.5, 0.5, 0.5], dtype=np.float64)

        colors = np.tile(default_color[None, :], (self.n_faces, 1))
        valid = self.weight_sum > 0
        colors[valid] = (self.color_sum[valid] /
                         self.weight_sum[valid, None])
        return np.clip(colors, 0.0, 1.0)

    def save(self, path):
        np.savez_compressed(path,
                            color_sum=self.color_sum,
                            weight_sum=self.weight_sum,
                            sample_count=self.sample_count)

    def load(self, path):
        data = np.load(path)
        self.color_sum = data["color_sum"]
        self.weight_sum = data["weight_sum"]
        self.sample_count = data["sample_count"]


def compute_surface_points(face_id_map, depth_map, camera_position,
                           camera_rays_dir):
    """
    计算每个像素对应的 3D 表面点位置。

    camera_rays_dir: (H, W, 3) 相机射线方向（世界坐标系，单位向量）

    Returns: (H, W, 3) 表面点世界坐标
    """
    points = (camera_position[None, None, :] +
              camera_rays_dir * depth_map[:, :, None])
    return points


def compute_view_directions(surface_points, camera_position):
    """
    计算从表面点指向相机的归一化方向。

    surface_points: (N, 3)
    camera_position: (3,)
    """
    to_cam = camera_position[None, :] - surface_points
    dist = np.linalg.norm(to_cam, axis=1, keepdims=True)
    dist = np.maximum(dist, 1e-8)
    return to_cam / dist
