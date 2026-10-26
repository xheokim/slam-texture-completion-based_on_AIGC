"""
Module 1: Structured Keyframe Export from ORB-SLAM3 Output

Reads ORB-SLAM3 KeyFrameTrajectory.txt and matches keyframe timestamps
to the original TUM RGB-D frames, producing a standardized keyframes/
directory ready for downstream modules.

Input:
  - ORB-SLAM3 KeyFrameTrajectory.txt (timestamp tx ty tz qx qy qz qw)
  - TUM rgb.txt, depth.txt (timestamp → filename)
  - TUM rgb/ and depth/ image directories

Output:
  keyframes/
  ├── poses.txt    # frame_id tx ty tz qx qy qz qw (no header)
  ├── rgb/         # 000000.png, 000001.png, ...
  └── depth/       # 000000.png, 000001.png, ...

Usage:
  python slam/export_keyframes.py
"""

import os
import shutil
import numpy as np


def load_trajectory(path):
    """Load ORB-SLAM3 trajectory: [(timestamp, tx, ty, tz, qx, qy, qz, qw), ...]"""
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            entries.append([float(x) for x in parts])
    return entries


def load_tum_timestamps(txt_path):
    """Load TUM rgb.txt/depth.txt: {timestamp: filename}"""
    mapping = {}
    with open(txt_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            ts = float(parts[0])
            fname = parts[1]
            mapping[ts] = fname
    return mapping


def find_closest_timestamp(target, candidates):
    """Find the closest timestamp from a sorted list of candidates."""
    idx = np.argmin(np.abs(np.array(candidates) - target))
    return candidates[idx]


def export_keyframes(
    keyframe_traj_path,
    rgb_txt_path,
    depth_txt_path,
    rgb_dir,
    depth_dir,
    output_dir,
):
    """
    Export standardized keyframes from ORB-SLAM3 output.

    For each keyframe timestamp, finds the temporally closest RGB and depth
    frames, copies them with sequential numbering, and writes poses.txt.
    """
    print("=" * 60)
    print("Module 1: ORB-SLAM3 Keyframe Export")
    print("=" * 60)

    # Load data
    print("\n[1/3] Loading trajectories...")
    keyframes = load_trajectory(keyframe_traj_path)
    rgb_map = load_tum_timestamps(rgb_txt_path)
    depth_map = load_tum_timestamps(depth_txt_path)
    rgb_ts_list = sorted(rgb_map.keys())
    depth_ts_list = sorted(depth_map.keys())
    print(f"  Keyframes: {len(keyframes)}")
    print(f"  RGB frames: {len(rgb_map)}")
    print(f"  Depth frames: {len(depth_map)}")

    # Create output directories
    print("\n[2/3] Matching and exporting frames...")
    os.makedirs(os.path.join(output_dir, "rgb"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "depth"), exist_ok=True)

    poses_lines = []
    exported = 0
    skipped = 0

    for frame_id, entry in enumerate(keyframes):
        ts, tx, ty, tz, qx, qy, qz, qw = entry

        # Find closest RGB frame
        rgb_ts = find_closest_timestamp(ts, rgb_ts_list)
        # Find closest depth frame
        depth_ts = find_closest_timestamp(ts, depth_ts_list)

        rgb_src = os.path.join(rgb_dir, os.path.basename(rgb_map[rgb_ts]))
        depth_src = os.path.join(depth_dir, os.path.basename(depth_map[depth_ts]))

        if not os.path.exists(rgb_src) or not os.path.exists(depth_src):
            skipped += 1
            continue

        # Copy with sequential naming
        dst_name = f"{frame_id:06d}.png"
        shutil.copy2(rgb_src, os.path.join(output_dir, "rgb", dst_name))
        shutil.copy2(depth_src, os.path.join(output_dir, "depth", dst_name))

        poses_lines.append(
            f"{frame_id} {tx:.7f} {ty:.7f} {tz:.7f} "
            f"{qx:.7f} {qy:.7f} {qz:.7f} {qw:.7f}"
        )
        exported += 1

    print(f"  Exported: {exported}, Skipped: {skipped}")

    # Write poses.txt
    print("\n[3/3] Writing poses.txt...")
    poses_path = os.path.join(output_dir, "poses.txt")
    with open(poses_path, "w") as f:
        f.write("# frame_id tx ty tz qx qy qz qw\n")
        f.write("\n".join(poses_lines))
        f.write("\n")

    print(f"  poses.txt saved: {poses_path}")
    print(f"\nExported {exported} keyframes to: {output_dir}/")
    print(f"  rgb/   — {exported} PNG images")
    print(f"  depth/ — {exported} PNG images")
    print(f"  poses.txt — frame_id + 6DoF pose per line")

    return exported


if __name__ == "__main__":
    # Default paths — adjust for your dataset
    DATASET_DIR = r"D:\LearnPython\finalproject\data\fr2_desk\rgbd_dataset_freiburg2_desk"
    export_keyframes(
        keyframe_traj_path=os.path.join(DATASET_DIR, "KeyFrameTrajectory.txt"),
        rgb_txt_path=os.path.join(DATASET_DIR, "rgb.txt"),
        depth_txt_path=os.path.join(DATASET_DIR, "depth.txt"),
        rgb_dir=os.path.join(DATASET_DIR, "rgb"),
        depth_dir=os.path.join(DATASET_DIR, "depth"),
        output_dir=os.path.join(DATASET_DIR, "keyframes"),
    )
