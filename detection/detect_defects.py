"""
模块③ 缺陷检测
检测网格上的低纹理区域 —— 融合 2D Laplacian 纹理 + 3D 深度几何
"""

import os, sys
import numpy as np
import open3d as o3d
import cv2

# ===== 路径 =====
DATA_DIR = r"D:\LearnPython\finalproject\data\fr2_desk\rgbd_dataset_freiburg2_desk"
KF_DIR = os.path.join(DATA_DIR, "keyframes")
OUTPUT_DIR = r"D:\LearnPython\finalproject\outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===== 相机内参 =====
FX, FY = 520.908620, 521.007327
CX, CY = 325.141442, 249.701764
WIDTH, HEIGHT = 640, 480

# ===== 参数 =====
LAPLACIAN_THRESH = 10.0       # Laplacian方差低于此值→低纹理
DEPTH_GRAD_THRESH = 0.05      # 深度梯度低于此值→几何平坦
ALPHA = 0.7                    # 纹理/几何融合权重
VISIBILITY_ANGLE_DEG = 75      # 面片法向量与视线夹角阈值
SAMPLE_EVERY_N = 5             # 每N帧采样一次（减少计算量）


def load_poses(path):
    poses = []
    with open(path) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.strip().split()
            poses.append([float(x) for x in parts])
    return poses


def quat_to_matrix(tx, ty, tz, qx, qy, qz, qw):
    T = np.eye(4)
    R = o3d.geometry.get_rotation_matrix_from_quaternion([qw, qx, qy, qz])
    T[:3, :3] = R
    T[:3, 3] = [tx, ty, tz]
    return T


def is_visible(face_center, face_normal, T_cam):
    """检查面片是否对相机可见（正面面向相机 + 在视野内）"""
    cam_pos = T_cam[:3, 3]
    cam_dir = T_cam[:3, :3] @ np.array([0, 0, 1])

    to_face = face_center - cam_pos
    dist = np.linalg.norm(to_face)
    if dist < 0.01:
        return False, None

    view_dir = -to_face / dist
    cos_angle = np.dot(face_normal, view_dir)
    if cos_angle < np.cos(np.radians(VISIBILITY_ANGLE_DEG)):
        return False, None

    # 投影到图像平面
    p_cam = np.linalg.inv(T_cam[:3, :3]) @ to_face
    if p_cam[2] <= 0:
        return False, None

    u = FX * p_cam[0] / p_cam[2] + CX
    v = FY * p_cam[1] / p_cam[2] + CY

    if u < 0 or u >= WIDTH - 1 or v < 0 or v >= HEIGHT - 1:
        return False, None

    return True, (int(u), int(v))


def compute_laplacian_score(img_gray):
    """Laplacian方差 → 纹理丰富度。越低=纹理越少"""
    lap = cv2.Laplacian(img_gray, cv2.CV_64F)
    return lap.var()


def compute_depth_gradient_score(depth_img):
    """深度梯度模长均值 → 几何复杂度。越低=越平坦"""
    gx = cv2.Sobel(depth_img, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(depth_img, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx ** 2 + gy ** 2)
    return np.mean(grad_mag)


def main():
    # 1. Load mesh
    print("Loading mesh...")
    mesh = o3d.io.read_triangle_mesh(os.path.join(OUTPUT_DIR, "scene_mesh.ply"))
    mesh.compute_vertex_normals()
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    face_normals = np.asarray(mesh.triangle_normals)
    n_faces = len(triangles)
    print(f"Faces: {n_faces}")

    # 每个面片的累积分数
    tex_scores = np.zeros(n_faces)
    geo_scores = np.zeros(n_faces)
    visibility_count = np.zeros(n_faces, dtype=int)

    # 2. 加载位姿和关键帧
    poses = load_poses(os.path.join(KF_DIR, "poses.txt"))
    rgb_files = sorted(os.listdir(os.path.join(KF_DIR, "rgb")))
    depth_files = sorted(os.listdir(os.path.join(KF_DIR, "depth")))
    n_frames = min(len(rgb_files), len(depth_files), len(poses))

    # 3. 遍历关键帧，计算面片可见性 + 纹理分数
    face_centers = np.mean(vertices[triangles], axis=1)

    print(f"Iterating {n_frames} keyframes (sampling every {SAMPLE_EVERY_N})...")
    processed = 0

    for i in range(0, n_frames, SAMPLE_EVERY_N):
        ts, tx, ty, tz, qx, qy, qz, qw = poses[i]

        rgb = cv2.imread(os.path.join(KF_DIR, "rgb", rgb_files[i]))
        depth = cv2.imread(os.path.join(KF_DIR, "depth", depth_files[i]),
                           cv2.IMREAD_UNCHANGED)
        if rgb is None or depth is None:
            continue

        rgb_gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        T = quat_to_matrix(tx, ty, tz, qx, qy, qz, qw)
        depth_norm = depth.astype(np.float64) / 5208.0

        # 计算整图分数
        lap_val = compute_laplacian_score(rgb_gray)
        depth_val = compute_depth_gradient_score(depth_norm)

        # 对所有面片检查可见性
        for fi in range(n_faces):
            visible, uv = is_visible(face_centers[fi], face_normals[fi], T)
            if visible:
                tex_scores[fi] += lap_val
                geo_scores[fi] += depth_val
                visibility_count[fi] += 1

        processed += 1
        if processed % 10 == 0:
            print(f"  {processed} frames processed")

    # 4. 对未被看到的面对进行分数传播
    seen_mask = visibility_count > 0
    unseen_mask = ~seen_mask
    print(f"Visible faces: {seen_mask.sum()}, unseen faces: {unseen_mask.sum()}")

    # 归一化
    tex_scores[seen_mask] /= visibility_count[seen_mask]
    geo_scores[seen_mask] /= visibility_count[seen_mask]

    # 5. 融合分数（论文公式）
    tex_norm = np.zeros_like(tex_scores)
    geo_norm = np.zeros_like(geo_scores)
    if tex_scores[seen_mask].max() > 0:
        tex_norm[seen_mask] = tex_scores[seen_mask] / tex_scores[seen_mask].max()
    if geo_scores[seen_mask].max() > 0:
        geo_norm[seen_mask] = geo_scores[seen_mask] / geo_scores[seen_mask].max()

    # Score = α * (1 - tex_norm) + (1-α) * (1 - geo_norm)
    # 高分数 = 低纹理 + 平坦区域 → 需要补全
    defect_score = ALPHA * (1.0 - tex_norm) + (1 - ALPHA) * (1.0 - geo_norm)

    # 6. Otsu 自适应阈值
    valid_scores = defect_score[seen_mask]
    if len(valid_scores) > 0:
        # 取前 30% 高分面片作为缺陷区域
        threshold = np.percentile(valid_scores, 70)
        defect_mask = defect_score >= threshold
    print(f"Defect threshold: {threshold:.4f}, defect faces: {defect_mask.sum()} / {n_faces}")

    # 7. Propagate to unseen faces (KNN)
    if unseen_mask.any():
        print("Propagating defect labels to unseen faces...")
        unseen_indices = np.where(unseen_mask)[0]
        seen_indices = np.where(seen_mask)[0]
        seen_centers = face_centers[seen_mask]
        for ui in unseen_indices:
            dists = np.linalg.norm(seen_centers - face_centers[ui], axis=1)
            nn = seen_indices[np.argmin(dists)]
            defect_mask[ui] = defect_mask[nn]

    # 8. 保存结果
    # 低纹理区域 → 红色，正常区域 → 灰色
    colors = np.ones((n_faces, 3)) * 0.7
    colors[defect_mask] = [1.0, 0.2, 0.2]
    np.save(os.path.join(OUTPUT_DIR, "defect_mask.npy"), defect_mask)
    np.save(os.path.join(OUTPUT_DIR, "defect_score.npy"), defect_score)

    # 可视化：给网格上色
    vert_colors = np.zeros((len(vertices), 3))
    vert_counts = np.zeros(len(vertices))
    for fi in range(n_faces):
        for vi in triangles[fi]:
            vert_colors[vi] += colors[fi]
            vert_counts[vi] += 1
    vert_colors[vert_counts > 0] /= vert_counts[vert_counts > 0, np.newaxis]
    mesh.vertex_colors = o3d.utility.Vector3dVector(vert_colors)

    viz_path = os.path.join(OUTPUT_DIR, "scene_defect.ply")
    o3d.io.write_triangle_mesh(viz_path, mesh)
    print(f"\nVisualization mesh saved: {viz_path}")
    print(f"Red = low-texture defects ({defect_mask.sum()}/{n_faces} faces)")

    return mesh, defect_mask


if __name__ == "__main__":
    main()
