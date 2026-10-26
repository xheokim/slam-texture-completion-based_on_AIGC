"""
模块④ 渲染器 — Open3D RaycastingScene 封装
从相机位姿渲染 face_id / depth / normal / color 图像
"""

import numpy as np
import open3d as o3d
from PIL import Image


class MeshRenderer:
    def __init__(self, mesh, intrinsic_matrix, width, height):
        """
        mesh: o3d.geometry.TriangleMesh（legacy）
        intrinsic_matrix: 3x3 numpy array
        width, height: 图像尺寸
        """
        self.width = width
        self.height = height
        self.intrinsic = o3d.core.Tensor(intrinsic_matrix, dtype=o3d.core.float64)

        # 转为 tensor mesh 并构建 RaycastingScene
        mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
        self.scene = o3d.t.geometry.RaycastingScene()
        self.scene.add_triangles(mesh_t)

        # 缓存顶点颜色为 numpy
        vert_colors = np.asarray(mesh.vertex_colors, dtype=np.float64)
        self.triangles = np.asarray(mesh.triangles, dtype=np.int64)  # (n_faces, 3)
        self.vert_colors = vert_colors  # (n_verts, 3) float64

    def render(self, T_c2w, defect_face_array):
        """
        T_c2w: 4x4 numpy array, camera-to-world 变换矩阵
        defect_face_array: (n_faces,) bool array, 缺陷面片标记

        Returns dict:
            face_id:    (H, W) int32,  -1 = 无交点
            depth:      (H, W) float32, 米
            normal:     (H, W, 3) float32, 世界坐标系法向量
            color:      (H, W, 3) uint8, RGB
            defect_mask:(H, W) bool, 缺陷面片覆盖的像素
        """
        # 世界→相机 外参 (Open3D RaycastingScene 要求 w2c)
        T_w2c = np.linalg.inv(T_c2w)
        extrinsic = o3d.core.Tensor(T_w2c, dtype=o3d.core.float64)

        # 创建射线
        rays = self.scene.create_rays_pinhole(
            self.intrinsic, extrinsic, self.width, self.height
        )

        # 射线求交
        ans = self.scene.cast_rays(rays)
        # ans keys: 't_hit', 'primitive_ids', 'primitive_uvs', 'primitive_normals', 'geometry_ids'

        # 提取结果
        t_hit = ans["t_hit"].numpy()                    # (H, W) float32, inf if miss
        prim_ids = ans["primitive_ids"].numpy()          # (H, W) uint32, UINT32_MAX if miss
        prim_uvs = ans["primitive_uvs"].numpy()          # (H, W, 2) float32, [u, v] barycentric
        prim_normals = ans["primitive_normals"].numpy()  # (H, W, 3) float32

        # Open3D RaycastingScene 用 UINT32_MAX 表示 miss
        MISS_SENTINEL = 0xFFFFFFFF
        hit_mask = prim_ids != MISS_SENTINEL

        # 转为有符号以兼容后续索引
        prim_ids_signed = prim_ids.astype(np.int64)

        # 逐像素插值顶点颜色
        colors = self._interpolate_colors(prim_ids_signed, prim_uvs, hit_mask)

        # 缺陷掩膜 — 向量化布尔索引
        defect_mask_2d = np.zeros((self.height, self.width), dtype=bool)
        if defect_face_array is not None and hit_mask.any():
            defect_mask_2d[hit_mask] = defect_face_array[prim_ids_signed[hit_mask]]

        return {
            "face_id": prim_ids,
            "depth": np.where(hit_mask, t_hit, 0.0).astype(np.float32),
            "normal": np.where(hit_mask[..., None], prim_normals, 0.0).astype(np.float32),
            "color": colors,
            "defect_mask": defect_mask_2d,
        }

    def _interpolate_colors(self, prim_ids, prim_uvs, hit_mask):
        """用重心坐标插值面片三顶点的颜色"""
        colors = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        if not hit_mask.any():
            return colors

        hit_rows, hit_cols = np.where(hit_mask)
        face_indices = prim_ids[hit_rows, hit_cols]

        # 获取每个命中像素的面片顶点索引
        v0 = self.triangles[face_indices, 0]
        v1 = self.triangles[face_indices, 1]
        v2 = self.triangles[face_indices, 2]

        u = prim_uvs[hit_rows, hit_cols, 0]
        v = prim_uvs[hit_rows, hit_cols, 1]
        w_coord = 1.0 - u - v

        # 获取顶点颜色 (numpy)
        vc = self.vert_colors

        c0 = vc[v0]
        c1 = vc[v1]
        c2 = vc[v2]

        # 重心坐标插值
        interp = (c0 * u[:, None] + c1 * v[:, None] + c2 * w_coord[:, None])
        interp = np.clip(interp * 255, 0, 255).astype(np.uint8)

        colors[hit_rows, hit_cols] = interp
        return colors

    def create_depth_condition(self, depth_map):
        """将深度图归一化到 [0, 255] 三通道 PIL Image (ControlNet 输入)"""
        d = depth_map.copy()
        valid = d > 0
        if valid.any():
            d_min, d_max = d[valid].min(), d[valid].max()
            if d_max - d_min > 1e-8:
                d[valid] = (d[valid] - d_min) / (d_max - d_min) * 255.0
        d = d.astype(np.uint8)
        return Image.fromarray(np.stack([d, d, d], axis=-1), mode="RGB")


# ===== 工具函数 =====

def load_poses(poses_path):
    """加载位姿 (timestamp tx ty tz qx qy qz qw) → 4x4 相机到世界矩阵列表"""
    poses = []
    with open(poses_path) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.strip().split()
            tx, ty, tz = float(parts[1]), float(parts[2]), float(parts[3])
            qx, qy, qz, qw = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])
            T = np.eye(4)
            T[:3, :3] = o3d.geometry.get_rotation_matrix_from_quaternion([qw, qx, qy, qz])
            T[:3, 3] = [tx, ty, tz]
            poses.append(T)
    return poses
