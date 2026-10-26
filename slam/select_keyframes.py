"""
Module 1: Coverage-Aware Keyframe Selection

Optimizes ORB-SLAM3's keyframe set for the texture completion task.
ORB-SLAM3 selects keyframes for tracking quality; this script re-ranks
them for geometric coverage — ensuring defect regions are observed from
diverse viewpoints.

Algorithm: Farthest Point Sampling (FPS) on camera positions.
  - Starts from a random viewpoint
  - Iteratively adds the viewpoint farthest from all selected ones
  - Optionally weights by defect region visibility score

This is a downstream optimization — it does NOT modify ORB-SLAM3 source.
It post-processes the exported keyframes.

Reference:
  Eldar et al., "The Farthest Point Strategy for Progressive Image Sampling"
  IEEE Trans. Image Processing, 1997.

Usage:
  # Basic FPS (select 100 most diverse viewpoints from 270 keyframes)
  python slam/select_keyframes.py --n_views 100

  # With defect-aware weighting (requires defect_mask.npy and mesh)
  python slam/select_keyframes.py --n_views 100 --defect_aware
"""

import os
import sys
import argparse
import numpy as np

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def load_poses(poses_path):
    """
    Load poses.txt: [(timestamp, tx, ty, tz, qx, qy, qz, qw), ...]
    First column can be frame_id (int) or timestamp (float) — auto-detected.
    """
    entries = []
    with open(poses_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            ts = float(parts[0])
            tx, ty, tz = float(parts[1]), float(parts[2]), float(parts[3])
            qx, qy, qz, qw = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])
            entries.append((ts, tx, ty, tz, qx, qy, qz, qw))
    return entries


def quat_to_rotation(qx, qy, qz, qw):
    """Quaternion to 3x3 rotation matrix."""
    R = np.array([
        [1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw,     2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw,     1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw,     2*qy*qz + 2*qx*qw,     1 - 2*qx**2 - 2*qy**2],
    ])
    return R


def fps_sampling(positions, n_views, start_idx=0):
    """
    Farthest Point Sampling on camera positions.

    positions: (N, 3) camera world coordinates
    n_views:   number of viewpoints to select
    start_idx: index of the first viewpoint (0 = random first)

    Returns: list of selected indices (in selection order)
    """
    n = len(positions)
    if n_views >= n:
        return list(range(n))

    selected = [start_idx]
    # Distance from each point to the nearest selected point
    min_dists = np.full(n, np.inf)

    for _ in range(1, n_views):
        # Update distances with the latest selected point
        last_selected = positions[selected[-1]]
        dists = np.linalg.norm(positions - last_selected, axis=1)
        min_dists = np.minimum(min_dists, dists)

        # Pick the farthest
        min_dists[selected] = -1.0  # exclude already selected
        next_idx = int(np.argmax(min_dists))
        selected.append(next_idx)

    return selected


def compute_view_scores(poses, mesh_path, defect_mask_path, cam_intrinsic,
                        width=640, height=480):
    """
    Score each viewpoint by how many defect faces are visible.

    Uses the MeshRenderer from Module 4 to raycast and count visible
    defect face pixels for each candidate viewpoint.
    """
    import open3d as o3d
    from generation.renderer import MeshRenderer

    mesh = o3d.io.read_triangle_mesh(mesh_path)
    mesh.compute_vertex_normals()
    defect_mask = np.load(defect_mask_path)
    renderer = MeshRenderer(mesh, cam_intrinsic, width, height)

    scores = np.zeros(len(poses))
    for i, (_, tx, ty, tz, qx, qy, qz, qw) in enumerate(poses):
        R = quat_to_rotation(qx, qy, qz, qw)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = [tx, ty, tz]

        result = renderer.render(T, defect_mask)
        scores[i] = result["defect_mask"].sum()

    return scores


def fps_weighted(positions, scores, n_views, coverage_weight=0.7,
                 start_idx=None):
    """
    Weighted FPS: balances geometric coverage with defect visibility.

    positions:       (N, 3) camera positions
    scores:          (N,) defect visibility scores per view
    n_views:         number to select
    coverage_weight: 1.0=full FPS, 0.0=only defect scores
    """
    n = len(positions)
    if n_views >= n:
        return list(range(n))

    # Normalize positions and scores to [0, 1]
    pos_range = positions.max(axis=0) - positions.min(axis=0)
    pos_range[pos_range < 1e-6] = 1.0
    pos_norm = (positions - positions.min(axis=0)) / pos_range

    score_norm = scores / (scores.max() + 1e-8)

    if start_idx is None:
        start_idx = int(np.argmax(scores))  # start with best defect view

    selected = [start_idx]
    min_dists = np.full(n, np.inf)

    for _ in range(1, n_views):
        last_pos = pos_norm[selected[-1]]
        geo_dists = np.linalg.norm(pos_norm - last_pos, axis=1)

        # Combined metric: blend geometric distance with defect score
        combined = (
            coverage_weight * geo_dists / (geo_dists.max() + 1e-8)
            + (1 - coverage_weight) * score_norm
        )
        min_dists = np.maximum(min_dists, combined)
        min_dists[selected] = -1.0

        next_idx = int(np.argmax(min_dists))
        selected.append(next_idx)

    return selected


def compute_coverage_metric(positions, selected_indices):
    """
    Compute coverage quality: mean nearest-neighbor distance between
    selected viewpoints. Higher = better spatial spread.
    """
    selected_pos = positions[selected_indices]
    n = len(selected_pos)
    if n < 2:
        return 0.0

    # Pairwise distances (upper triangle)
    total_min_dist = 0.0
    for i in range(n):
        dists = np.linalg.norm(selected_pos - selected_pos[i], axis=1)
        dists[i] = np.inf
        total_min_dist += dists.min()
    return total_min_dist / n


def run_selection(poses_path, output_dir, n_views=100, defect_aware=False,
                  mesh_path=None, defect_mask_path=None):
    """
    Main selection pipeline.

    Loads keyframe poses, runs FPS (or weighted FPS), writes a filtered
    keyframe set, and prints coverage statistics.
    """
    print("=" * 60)
    print("Module 1: Coverage-Aware Keyframe Selection (FPS)")
    print("=" * 60)

    # Load
    print("\n[1/3] Loading keyframe poses...")
    entries = load_poses(poses_path)
    positions = np.array([[e[1], e[2], e[3]] for e in entries])
    print(f"  Total keyframes: {len(entries)}")
    print(f"  Position range: X[{positions[:, 0].min():.2f}, {positions[:, 0].max():.2f}] "
          f"Y[{positions[:, 1].min():.2f}, {positions[:, 1].max():.2f}] "
          f"Z[{positions[:, 2].min():.2f}, {positions[:, 2].max():.2f}]")

    # Select
    if defect_aware and mesh_path and defect_mask_path:
        print(f"\n[2/3] Defect-aware weighted FPS (target {n_views} views)...")
        cam_intrinsic = np.array([
            [520.908620, 0, 325.141442],
            [0, 521.007327, 249.701764],
            [0, 0, 1],
        ])
        scores = compute_view_scores(
            entries, mesh_path, defect_mask_path, cam_intrinsic
        )
        print(f"  Defect visibility range: {scores.min():.0f} ~ {scores.max():.0f}")
        selected = fps_weighted(positions, scores, n_views, coverage_weight=0.7)
    else:
        print(f"\n[2/3] Geometric FPS sampling (target {n_views} views)...")
        # Use a random first view for better coverage
        start_idx = np.random.randint(0, len(positions))
        selected = fps_sampling(positions, n_views, start_idx=start_idx)

    print(f"  Selected {len(selected)} / {len(entries)} viewpoints")

    # Coverage stats
    coverage = compute_coverage_metric(positions, selected)
    # Original (first N random) coverage for comparison
    original_coverage = compute_coverage_metric(positions, list(range(min(n_views, len(positions)))))
    print(f"  Coverage metric: {coverage:.3f} (vs sequential {original_coverage:.3f})")
    print(f"  Improvement: {(coverage / original_coverage - 1) * 100:.1f}%")

    # Save filtered poses
    print(f"\n[3/3] Writing filtered keyframes...")
    os.makedirs(output_dir, exist_ok=True)
    selected_poses_path = os.path.join(output_dir, "poses_selected.txt")

    with open(selected_poses_path, "w") as f:
        f.write("# timestamp tx ty tz qx qy qz qw\n")
        for idx in selected:
            ts, tx, ty, tz, qx, qy, qz, qw = entries[idx]
            f.write(f"{ts:.6f} {tx:.7f} {ty:.7f} {tz:.7f} "
                    f"{qx:.7f} {qy:.7f} {qz:.7f} {qw:.7f}\n")

    print(f"  Saved: {selected_poses_path}")
    print(f"  Copy corresponding rgb/depth images to use this filtered set.")

    return selected, coverage


def main():
    parser = argparse.ArgumentParser(
        description="Coverage-aware keyframe selection via FPS"
    )
    parser.add_argument("--poses", type=str,
                        default=r"D:\LearnPython\finalproject\data\fr2_desk"
                                r"\rgbd_dataset_freiburg2_desk\keyframes\poses.txt",
                        help="Path to poses.txt (default: fr2_desk keyframes)")
    parser.add_argument("--output_dir", type=str,
                        default=r"D:\LearnPython\finalproject\outputs\selected_keyframes",
                        help="Output directory for filtered poses")
    parser.add_argument("--n_views", type=int, default=100,
                        help="Number of viewpoints to select")
    parser.add_argument("--defect_aware", action="store_true",
                        help="Use defect visibility weighting")
    parser.add_argument("--mesh", type=str,
                        default=r"D:\LearnPython\finalproject\outputs\scene_mesh.ply",
                        help="Path to mesh (for defect-aware mode)")
    parser.add_argument("--defect_mask", type=str,
                        default=r"D:\LearnPython\finalproject\outputs\defect_mask.npy",
                        help="Path to defect mask (for defect-aware mode)")

    args = parser.parse_args()
    run_selection(
        poses_path=args.poses,
        output_dir=args.output_dir,
        n_views=args.n_views,
        defect_aware=args.defect_aware,
        mesh_path=args.mesh,
        defect_mask_path=args.defect_mask,
    )


if __name__ == "__main__":
    main()
