"""
模块⑤ 优化 — 离群值过滤 + 网格双边滤波
"""

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve


def build_face_adjacency(triangles, n_faces):
    """构建面片邻接表"""
    edge_to_faces = {}
    for fi, tri in enumerate(triangles):
        edges = [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])]
        for e in edges:
            key = tuple(sorted(e))
            edge_to_faces.setdefault(key, []).append(fi)

    adjacency = {fi: set() for fi in range(n_faces)}
    for faces in edge_to_faces.values():
        if len(faces) > 1:
            for i in range(len(faces)):
                for j in range(i + 1, len(faces)):
                    adjacency[faces[i]].add(faces[j])
                    adjacency[faces[j]].add(faces[i])
    return adjacency


def remove_outliers(face_colors, adjacency, weight_sum, defect_mask,
                    min_obs=3, std_thresh=2.5):
    """
    移除颜色离群值：与邻域颜色偏差过大的面片。
    仅处理缺陷面。
    """
    cleaned = face_colors.copy()
    outlier_mask = np.zeros(len(face_colors), dtype=bool)

    for fi in np.where(defect_mask)[0]:
        if weight_sum[fi] < min_obs * 1e-6:
            continue

        neighbors = list(adjacency[fi])
        if len(neighbors) < 2:
            continue

        neighbor_colors = face_colors[neighbors]
        mean_color = neighbor_colors.mean(axis=0)
        std_color = neighbor_colors.std(axis=0)

        # 检查是否离群
        diff = np.abs(face_colors[fi] - mean_color)
        if np.any(diff > std_thresh * np.maximum(std_color, 1e-3)):
            # 用邻域中值替换
            cleaned[fi] = np.median(neighbor_colors, axis=0)
            outlier_mask[fi] = True

    print(f"  Outliers: {outlier_mask.sum()} faces")
    return cleaned


def bilateral_filter_mesh(face_colors, adjacency, face_centers,
                          defect_mask, iters=3, sigma_spatial=0.8,
                          sigma_color=0.12, ring=2):
    """
    网格上的双边滤波 — 保留边缘的同时平滑颜色。

    空间核：基于面片中心欧氏距离
    颜色核：基于 RGB 差异
    仅在缺陷面及其邻域上操作。
    """
    # 确定滤波区域：缺陷面 + 周围 ring 环
    filter_set = set(np.where(defect_mask)[0])
    for _ in range(ring):
        expanded = set(filter_set)
        for fi in filter_set:
            expanded.update(adjacency[fi])
        filter_set = expanded
    filter_indices = np.array(sorted(filter_set))

    filtered = face_colors.copy()

    for iteration in range(iters):
        updated = filtered.copy()
        for fi in filter_indices:
            neighbors = list(adjacency[fi])
            if not neighbors:
                continue

            nb_colors = filtered[neighbors]
            curr_color = filtered[fi]

            # 空间权重：面片中心距离
            curr_center = face_centers[fi]
            nb_centers = face_centers[neighbors]
            spatial_dist = np.linalg.norm(nb_centers - curr_center, axis=1)
            w_spatial = np.exp(-0.5 * (spatial_dist / (sigma_spatial + 1e-8)) ** 2)

            # 颜色权重
            color_dist = np.linalg.norm(nb_colors - curr_color, axis=1)
            w_color = np.exp(-0.5 * (color_dist / (sigma_color + 1e-8)) ** 2)

            weights = w_spatial * w_color + 1e-8
            weights /= weights.sum()

            updated[fi] = np.dot(weights, nb_colors)

        filtered = updated
        if (iteration + 1) % 2 == 0:
            print(f"  Bilateral filter {iteration + 1}/{iters}")

    return filtered
