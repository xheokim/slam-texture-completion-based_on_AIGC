#!/bin/bash
# ============================================================================
# Module 1: Run ORB-SLAM3 on TUM RGB-D Dataset
#
# This script runs ORB-SLAM3 on a TUM RGB-D sequence and generates the
# keyframe trajectory needed by downstream modules.
#
# Prerequisites (one-time setup):
#   1. Install ORB-SLAM3 dependencies:
#      sudo apt install libglew-dev libboost-all-dev libeigen3-dev
#      sudo apt install libopencv-dev python3-dev
#
#   2. Install Pangolin (viewer):
#      git clone https://github.com/stevenlovegrove/Pangolin.git
#      cd Pangolin && mkdir build && cd build
#      cmake .. && make -j$(nproc)
#      sudo make install
#
#   3. Build ORB-SLAM3:
#      cd ~/ORB_SLAM3
#      chmod +x build.sh
#      ./build.sh
#
#   4. Download TUM dataset:
#      wget https://vision.in.tum.de/rgbd/dataset/freiburg2/rgbd_dataset_freiburg2_desk.tgz
#      tar -xzf rgbd_dataset_freiburg2_desk.tgz
#
#   5. Generate frame associations:
#      python evaluation/associate.py \
#          rgb.txt depth.txt > associations/fr2_desk.txt
#
# Usage:
#   chmod +x run_slam.sh
#   ./run_slam.sh
# ============================================================================

set -e

ORB_SLAM3_DIR="$HOME/ORB_SLAM3"
DATASET_DIR="/mnt/d/LearnPython/finalproject/data/fr2_desk/rgbd_dataset_freiburg2_desk"
VOCABULARY="$ORB_SLAM3_DIR/Vocabulary/ORBvoc.txt"
CONFIG="$ORB_SLAM3_DIR/Examples/RGB-D/TUM2.yaml"
ASSOCIATIONS="$ORB_SLAM3_DIR/Examples/RGB-D/associations/fr2_desk.txt"

echo "============================================================"
echo "  Module 1: ORB-SLAM3 on TUM fr2_desk"
echo "============================================================"

# Check prerequisites
if [ ! -f "$ORB_SLAM3_DIR/build/rgbd_tum" ]; then
    echo "ERROR: ORB-SLAM3 not built. Run build.sh first."
    echo "  cd ~/ORB_SLAM3 && ./build.sh"
    exit 1
fi

if [ ! -f "$VOCABULARY" ]; then
    echo "ERROR: ORB vocabulary not found. Download ORBvoc.txt.tar.gz and extract."
    echo "  See: https://github.com/UZ-SLAMLab/ORB_SLAM3"
    exit 1
fi

if [ ! -d "$DATASET_DIR" ]; then
    echo "ERROR: TUM fr2_desk dataset not found at $DATASET_DIR"
    echo "  Download from: https://vision.in.tum.de/data/datasets/rgbd-dataset"
    exit 1
fi

echo ""
echo "Dataset: $DATASET_DIR"
echo "Config:  $CONFIG"
echo "Vocab:   $VOCABULARY"
echo ""

# Run ORB-SLAM3
# KeyFrameTrajectory.txt is written to the current working directory
cd "$DATASET_DIR"

echo "Running ORB-SLAM3..."
"$ORB_SLAM3_DIR/build/rgbd_tum" \
    "$VOCABULARY" \
    "$CONFIG" \
    "$ASSOCIATIONS" \
    "$DATASET_DIR"

echo ""
echo "============================================================"
echo "  ORB-SLAM3 completed!"
echo ""
echo "  Output files (in $DATASET_DIR):"
echo "    CameraTrajectory.txt   - all frame poses"
echo "    KeyFrameTrajectory.txt - keyframe poses"
echo ""
echo "  Next step:"
echo "    python slam/export_keyframes.py"
echo "============================================================"
