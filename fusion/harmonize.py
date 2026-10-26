"""
模块⑤ 协调 — 颜色直方图协调（生成↔原始区域）
"""

import numpy as np


def color_harmonize(face_colors, defect_mask, adjacency,
                    strength=0.4, n_clusters=1):
    """
    协调生成区域与原始区域的颜色分布。

    方法：线性颜色变换
        x' = A * (x - mean_gen) + mean_orig
    其中 A 基于协方差匹配。

    strength=0 → 不变, strength=1 → 完全匹配原始分布
    """
    orig_colors = face_colors[~defect_mask]
    gen_colors = face_colors[defect_mask]

    if len(orig_colors) == 0 or len(gen_colors) == 0:
        return face_colors

    if n_clusters > 1:
        return _harmonize_per_cluster(
            face_colors, defect_mask, adjacency, strength, n_clusters
        )

    # 单全局变换
    harmonized = face_colors.copy()
    gen_transformed = _apply_color_transform(gen_colors, orig_colors, strength)
    harmonized[defect_mask] = gen_transformed

    return harmonized


def _harmonize_per_cluster(face_colors, defect_mask, adjacency,
                            strength, n_clusters):
    """按空间聚类分别协调（保留局部颜色变化）"""
    from sklearn.cluster import KMeans

    harmonized = face_colors.copy()

    # 对缺陷面空间聚类
    defect_indices = np.where(defect_mask)[0]

    # 简化：用面片索引近似空间位置（实际应使用 face_centers）
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(harmonized[defect_indices])

    for c in range(n_clusters):
        cluster_mask = np.zeros(len(defect_mask), dtype=bool)
        cluster_mask[defect_indices[labels == c]] = True

        # 该 cluster 的邻域原始面
        nearby_orig = set()
        for fi in defect_indices[labels == c]:
            nearby_orig.update(
                nb for nb in adjacency[fi] if not defect_mask[nb]
            )
        if not nearby_orig:
            continue

        nearby_indices = np.array(sorted(nearby_orig))
        orig_ref = face_colors[nearby_indices]
        gen_in = face_colors[cluster_mask]

        harmonized[cluster_mask] = _apply_color_transform(
            gen_in, orig_ref, strength
        )

    return harmonized


def _apply_color_transform(gen_colors, ref_colors, strength):
    """
    线性颜色变换：匹配均值和协方差。

    gen_colors: (N, 3) 生成区域颜色
    ref_colors: (M, 3) 参考区域颜色
    """
    if strength <= 0:
        return gen_colors

    mean_gen = gen_colors.mean(axis=0)
    mean_ref = ref_colors.mean(axis=0)

    # 协方差矩阵
    cov_gen = np.cov(gen_colors.T) + 1e-6 * np.eye(3)
    cov_ref = np.cov(ref_colors.T) + 1e-6 * np.eye(3)

    # 对称 Cholesky: A = cov_ref^(1/2) * cov_gen^(-1/2)
    try:
        L_gen = np.linalg.cholesky(cov_gen)
        L_ref = np.linalg.cholesky(cov_ref)
        A = L_ref @ np.linalg.inv(L_gen)
    except np.linalg.LinAlgError:
        # 退化：仅均值匹配
        A = np.eye(3)

    # 应用变换
    centered = gen_colors - mean_gen
    transformed = centered @ A.T + mean_ref

    # 插值
    blend = strength
    return gen_colors * (1 - blend) + transformed * blend
