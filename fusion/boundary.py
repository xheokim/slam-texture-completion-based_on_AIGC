"""
模块⑤ 边界优化 — 梯度域融合（Poisson 网格编辑）
"""

import numpy as np
from scipy.sparse import csr_matrix, eye
from scipy.sparse.linalg import spsolve


def gradient_domain_blend(face_colors, adjacency, defect_mask,
                          boundary_rings=3, lambda_reg=0.5):
    """
    梯度域网格颜色融合。

    最小化: ||grad(x) - grad(generated)||^2
    约束:   在非缺陷区域 x = original

    等价于求解 Poisson 方程:
        L * x = div(grad(generated))
    其中 L 是图 Laplacian，边界条件为 Dirichlet。

    在边界 rings 内进行，内部缺陷面不受边界约束。
    """
    n_faces = len(face_colors)

    # 1. 划分区域
    interior = _find_interior_faces(defect_mask, adjacency, boundary_rings)
    boundary_set = set()
    for fi in np.where(defect_mask & ~interior)[0]:
        for nb in adjacency[fi]:
            if not defect_mask[nb]:
                boundary_set.add(fi)
                break
    boundary = np.array(sorted(boundary_set))

    # Dirichlet 边界：非缺陷面 + 缺陷边界外部
    dirichlet = np.zeros(n_faces, dtype=bool)
    dirichlet[~defect_mask] = True  # 非缺陷面 → 固定原色
    # 不在 interior 也不在 boundary 的缺陷面维持原色
    dirichlet[defect_mask & ~interior & ~np.isin(
        np.arange(n_faces), boundary)] = True
    free = ~dirichlet

    if free.sum() == 0:
        print("  No free variables to solve")
        return face_colors

    print(f"  Free faces: {free.sum()}, boundary: {len(boundary)}, "
          f"interior: {interior.sum()}")

    # 2. 构建图 Laplacian
    row, col, data = [], [], []
    for fi in range(n_faces):
        neighbors = list(adjacency[fi])
        if not neighbors:
            continue
        degree = len(neighbors)
        row.append(fi)
        col.append(fi)
        data.append(degree)
        for nb in neighbors:
            row.append(fi)
            col.append(nb)
            data.append(-1.0)

    L = csr_matrix((data, (row, col)), shape=(n_faces, n_faces))

    # 3. 计算散度 b = div(grad(generated))
    b = np.zeros((n_faces, 3))
    for fi in range(n_faces):
        neighbors = list(adjacency[fi])
        if not neighbors:
            continue
        g_sum = np.zeros(3)
        for nb in neighbors:
            g_sum += face_colors[nb] - face_colors[fi]
        b[fi] = g_sum

    # 4. Dirichlet 边界条件
    # 对每个自由变量，减去 L[free, dirichlet] @ x[dirichlet]
    x_fixed = face_colors.copy()

    blended = face_colors.copy()
    for ch in range(3):
        rhs = b[:, ch].copy()

        # 修正 rhs: b' = b - L[:, dirichlet] * x[dirichlet]
        L_free_fixed = L[free, :][:, dirichlet]
        rhs[free] -= L_free_fixed @ x_fixed[dirichlet, ch]

        # 正则化: 添加 lambda * I 防止奇异
        A = L[free, :][:, free]
        if lambda_reg > 0:
            A = A + lambda_reg * eye(A.shape[0])

        # 求解
        x_free = spsolve(A, rhs[free])

        blended[free, ch] = np.clip(x_free, 0.0, 1.0)

    return blended


def _find_interior_faces(defect_mask, adjacency, boundary_rings):
    """找到在缺陷区域内部（距离边界 > boundary_rings）的面片"""
    # BFS: 从边界开始标记距离
    boundary = set()
    for fi in np.where(defect_mask)[0]:
        for nb in adjacency[fi]:
            if not defect_mask[nb]:
                boundary.add(fi)
                break

    dist = np.full(len(defect_mask), -1, dtype=int)
    queue = list(boundary)
    for bf in queue:
        dist[bf] = 0

    while queue:
        fi = queue.pop(0)
        for nb in adjacency[fi]:
            if defect_mask[nb] and dist[nb] == -1:
                dist[nb] = dist[fi] + 1
                queue.append(nb)

    interior = np.zeros(len(defect_mask), dtype=bool)
    for fi in np.where(defect_mask)[0]:
        if dist[fi] >= boundary_rings:
            interior[fi] = True

    return interior
