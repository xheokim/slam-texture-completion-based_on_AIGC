"""
模块④ 融合器 — 多视角加权融合 + 边界平滑
"""

import numpy as np
import open3d as o3d


class TextureBlender:
    def __init__(self, mesh, defect_mask):
        """
        mesh: o3d.geometry.TriangleMesh
        defect_mask: (n_faces,) bool, 缺陷面片标记
        """
        self.mesh = mesh
        self.defect_mask = defect_mask.copy()
        self.n_faces = len(defect_mask)
        self.triangles = np.asarray(mesh.triangles)

        # 保存原始顶点颜色
        self.original_vert_colors = np.asarray(mesh.vertex_colors).copy()

        # 构建面片邻接关系
        self.face_adjacency = self._build_face_adjacency()

    def _build_face_adjacency(self):
        """构建面片邻接表（共享至少一条边的面片）"""
        # 边 → 面片列表
        edge_to_faces = {}
        for fi, tri in enumerate(self.triangles):
            edges = [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])]
            for e in edges:
                key = tuple(sorted(e))
                edge_to_faces.setdefault(key, []).append(fi)

        adjacency = {fi: set() for fi in range(self.n_faces)}
        for faces in edge_to_faces.values():
            if len(faces) > 1:
                for i in range(len(faces)):
                    for j in range(i + 1, len(faces)):
                        adjacency[faces[i]].add(faces[j])
                        adjacency[faces[j]].add(faces[i])
        return adjacency

    def compute_final_face_colors(self, accumulator, default_color=None):
        """
        融合最终面片颜色：
        - 被观察到的缺陷面 → 加权平均颜色
        - 未被观察到的缺陷面 → 从邻近已修复面片传播
        - 非缺陷面 → 保留原始颜色
        """
        if default_color is None:
            default_color = np.array([0.5, 0.5, 0.5], dtype=np.float64)

        final = np.zeros((self.n_faces, 3), dtype=np.float64)

        # 从原始 mesh 计算面片颜色
        original_face_colors = self._compute_original_face_colors()

        # 缺陷面：优先用累积颜色，否则传播
        for fi in range(self.n_faces):
            if not self.defect_mask[fi]:
                final[fi] = original_face_colors[fi]
            elif accumulator.weight_sum[fi] > 0:
                final[fi] = (accumulator.color_sum[fi] /
                             accumulator.weight_sum[fi])
                final[fi] = np.clip(final[fi], 0.0, 1.0)
            else:
                final[fi] = default_color

        # 传播：未观察到的缺陷面从邻接已修复面片取平均
        self._propagate_to_unobserved(final, accumulator.weight_sum > 0)

        return np.clip(final, 0.0, 1.0)

    def _compute_original_face_colors(self):
        """从顶点颜色计算面片中心颜色"""
        vc = self.original_vert_colors
        face_colors = np.zeros((self.n_faces, 3))
        for fi, tri in enumerate(self.triangles):
            face_colors[fi] = np.mean(vc[list(tri)], axis=0)
        return face_colors

    def _propagate_to_unobserved(self, face_colors, observed_mask, max_iters=5):
        """将颜色从已观察面片传播到未观察面片（迭代平滑）"""
        unobserved = self.defect_mask & (~observed_mask)
        if not unobserved.any():
            return

        for _ in range(max_iters):
            updated = face_colors.copy()
            changed = False
            for fi in np.where(unobserved)[0]:
                neighbors = list(self.face_adjacency[fi])
                if not neighbors:
                    continue
                neighbor_colors = face_colors[neighbors]
                avg = neighbor_colors.mean(axis=0)
                if not np.allclose(avg, face_colors[fi]):
                    updated[fi] = avg
                    changed = True
            face_colors[:] = updated
            if not changed:
                break

    def smooth_boundary(self, face_colors, iterations=5, ring_count=2):
        """
        Laplacian 平滑：只平滑缺陷边界附近的面片，
        避免模糊整个纹理。
        """
        if iterations <= 0:
            return face_colors

        boundary = self._find_boundary_faces(ring_count)
        if not boundary:
            return face_colors

        smoothed = face_colors.copy()
        for _ in range(iterations):
            updated = smoothed.copy()
            for fi in boundary:
                neighbors = list(self.face_adjacency[fi])
                if len(neighbors) >= 2:
                    avg = np.mean(smoothed[neighbors], axis=0)
                    # 在原色和邻域平均之间插值 (0.5 blend)
                    updated[fi] = smoothed[fi] * 0.6 + avg * 0.4
            smoothed = updated
        return smoothed

    def _find_boundary_faces(self, ring_count=2):
        """
        找到 defect/normal 边界附近的面片。
        ring_count=1: 与 normal 面相邻的 defect 面
        ring_count=2: ring 1 的邻接面也加入
        """
        # 直接相邻边界
        boundary = set()
        for fi in range(self.n_faces):
            if not self.defect_mask[fi]:
                continue
            has_normal_neighbor = any(
                not self.defect_mask[nb]
                for nb in self.face_adjacency[fi]
            )
            if has_normal_neighbor:
                boundary.add(fi)

        # 扩展 ring
        for r in range(1, ring_count):
            expanded = set(boundary)
            for fi in boundary:
                expanded.update(self.face_adjacency[fi])
            boundary = expanded

        return list(boundary)

    def face_colors_to_vertex_colors(self, face_colors):
        """面片颜色 → 顶点颜色（平均）"""
        vert_sum = np.zeros((len(self.mesh.vertices), 3), dtype=np.float64)
        vert_count = np.zeros(len(self.mesh.vertices), dtype=np.int32)
        for fi, tri in enumerate(self.triangles):
            for vi in tri:
                vert_sum[vi] += face_colors[fi]
                vert_count[vi] += 1
        vert_count[vert_count == 0] = 1
        return vert_sum / vert_count[:, None]

    def apply(self, accumulator):
        """Complete fusion pipeline"""
        print("Computing final face colors...")
        face_colors = self.compute_final_face_colors(accumulator)

        print("Boundary smoothing...")
        face_colors = self.smooth_boundary(
            face_colors,
            iterations=5,
            ring_count=2
        )

        print("Face colors → vertex colors...")
        vert_colors = self.face_colors_to_vertex_colors(face_colors)

        self.mesh.vertex_colors = o3d.utility.Vector3dVector(vert_colors)
        self.mesh.compute_vertex_normals()
        return self.mesh
