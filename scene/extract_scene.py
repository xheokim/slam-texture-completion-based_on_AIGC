"""
模块② 场景提取
从 ORB-SLAM3 关键帧 (RGB-D + 位姿) → Open3D 网格重建
"""

import os
import numpy as np
import open3d as o3d
import cv2

# ===== 路径 =====
DATA_DIR = r"D:\LearnPython\finalproject\data\fr2_desk\rgbd_dataset_freiburg2_desk"
KF_DIR = os.path.join(DATA_DIR, "keyframes")
OUTPUT_DIR = r"D:\LearnPython\finalproject\outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===== 相机内参 (from TUM2.yaml) =====
FX, FY = 520.908620, 521.007327
CX, CY = 325.141442, 249.701764
WIDTH, HEIGHT = 640, 480
DEPTH_SCALE = 5208.0  # TUM depth: raw_value / scale = meters

intrinsic = o3d.camera.PinholeCameraIntrinsic(WIDTH, HEIGHT, FX, FY, CX, CY)


def load_poses(poses_path):
    """加载关键帧位姿 (timestamp tx ty tz qx qy qz qw)"""
    poses = []
    with open(poses_path) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.strip().split()
            ts = float(parts[0])
            tx, ty, tz = float(parts[1]), float(parts[2]), float(parts[3])
            qx, qy, qz, qw = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])
            poses.append((ts, tx, ty, tz, qx, qy, qz, qw))
    return poses


def quat_to_matrix(tx, ty, tz, qx, qy, qz, qw):
    """四元数+平移 → 4x4 变换矩阵"""
    T = np.eye(4)
    # 四元数 → 旋转矩阵
    R = o3d.geometry.get_rotation_matrix_from_quaternion([qw, qx, qy, qz])
    T[:3, :3] = R
    T[:3, 3] = [tx, ty, tz]
    return T


def main():
    # 1. 加载位姿
    poses = load_poses(os.path.join(KF_DIR, "poses.txt"))
    print(f"Loaded poses: {len(poses)}")

    # 2. 按序号配对 RGB 和 Depth（它们来自不同的时间戳匹配）
    rgb_files = sorted(os.listdir(os.path.join(KF_DIR, "rgb")))
    depth_files = sorted(os.listdir(os.path.join(KF_DIR, "depth")))
    n_frames = min(len(rgb_files), len(depth_files), len(poses))
    print(f"Frame pairs available: {n_frames}")

    # 3. 逐个处理 RGB-D → 点云 → 融合
    merged_pcd = o3d.geometry.PointCloud()
    skipped = 0

    for i in range(n_frames):
        _, tx, ty, tz, qx, qy, qz, qw = poses[i]

        rgb_path = os.path.join(KF_DIR, "rgb", rgb_files[i])
        depth_path = os.path.join(KF_DIR, "depth", depth_files[i])

        rgb = cv2.imread(rgb_path)
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if rgb is None or depth is None:
            skipped += 1
            continue

        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        # 创建 Open3D RGBD 图像
        rgb_o3d = o3d.geometry.Image(rgb)
        depth_o3d = o3d.geometry.Image(depth)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            rgb_o3d, depth_o3d,
            depth_scale=DEPTH_SCALE,
            depth_trunc=3.0,
            convert_rgb_to_intensity=False,
        )

        # 反投影 → 当前帧点云
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)

        # 用位姿变换到世界坐标
        T = quat_to_matrix(tx, ty, tz, qx, qy, qz, qw)
        pcd.transform(T)

        merged_pcd += pcd

        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{n_frames} frames, points: {len(merged_pcd.points)}")

    print(f"Skipped {skipped} frames")

    print(f"Merged point cloud: {len(merged_pcd.points)} points")

    # 3. Downsample + denoise
    print("Voxel downsampling (voxel=0.01m)...")
    merged_pcd = merged_pcd.voxel_down_sample(voxel_size=0.01)
    print(f"After downsampling: {len(merged_pcd.points)} points")

    print("Statistical outlier removal...")
    merged_pcd, _ = merged_pcd.remove_statistical_outlier(nb_neighbors=30, std_ratio=2.0)
    print(f"After denoising: {len(merged_pcd.points)} points")

    # 4. Save raw point cloud
    pcd_path = os.path.join(OUTPUT_DIR, "scene_raw.ply")
    o3d.io.write_point_cloud(pcd_path, merged_pcd)
    print(f"Raw point cloud saved: {pcd_path}")

    # 5. Estimate normals
    print("Estimating normals...")
    merged_pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30)
    )
    # 法向量一致性定向
    merged_pcd.orient_normals_towards_camera_location()

    # 6. Poisson surface reconstruction → mesh
    print("Poisson surface reconstruction (depth=10)...")
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        merged_pcd, depth=10, width=0, scale=1.1, linear_fit=False
    )

    # 7. Remove low-density faces
    print("Removing low-density regions...")
    densities = np.asarray(densities)
    density_threshold = np.quantile(densities, 0.05)
    vertices_to_remove = densities < density_threshold
    mesh.remove_vertices_by_mask(vertices_to_remove)
    print(f"After density filter: {len(mesh.vertices)} vertices, {len(mesh.triangles)} faces")

    # 8. Save full mesh
    mesh_path = os.path.join(OUTPUT_DIR, "scene_mesh_full.ply")
    o3d.io.write_triangle_mesh(mesh_path, mesh)
    print(f"Full mesh saved: {mesh_path}")

    # 9. Mesh simplification — Quadric error metric decimation
    target_triangles = 100000
    ratio = target_triangles / len(mesh.triangles)
    if ratio < 1.0:
        print(f"Quadric decimation → target {target_triangles} faces...")
        mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=target_triangles)
        print(f"After decimation: {len(mesh.vertices)} vertices, {len(mesh.triangles)} faces")

    # 10. Laplacian smoothing
    print("Laplacian smoothing (3 iterations)...")
    mesh = mesh.filter_smooth_laplacian(number_of_iterations=3)

    # 11. Recompute normals
    mesh.compute_vertex_normals()

    # 12. Save simplified mesh
    simple_path = os.path.join(OUTPUT_DIR, "scene_mesh.ply")
    o3d.io.write_triangle_mesh(simple_path, mesh)
    print(f"Simplified mesh saved: {simple_path}")
    print(f"Final: {len(mesh.vertices)} vertices, {len(mesh.triangles)} faces")

    return mesh


if __name__ == "__main__":
    mesh = main()
