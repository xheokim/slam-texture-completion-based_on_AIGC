# SLAM Map Texture Completion Based on Generation AI

Master's Thesis — City University of Hong Kong, MSc Artificial Intelligence  
**Author:** XheoKim  
**Date:** July 2026

---

## Overview

An end-to-end pipeline that takes raw RGB-D keyframe data from ORB-SLAM3 and produces a fully textured, visually enhanced 3D triangular mesh using generative AI (ControlNet + Stable Diffusion). The system addresses the problem of incomplete textures on large uniform surfaces (white walls, ceilings, floors) in SLAM-generated 3D maps.

## System Architecture

```
TUM RGB-D Dataset
       │
       ▼
┌─────────────────────────────────────────────────────────────┐
│ Module 1 — SLAM Frontend                          [slam/]   │
│   ORB-SLAM3 localization → 270 keyframe poses               │
│   + Coverage-aware FPS keyframe selection                   │
│   Output: KeyFrameTrajectory.txt, keyframes/{rgb,depth}/     │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ Module 2 — Scene Extraction                       [scene/]  │
│   Per-pixel back-projection → 80M point cloud fusion        │
│   → Poisson surface reconstruction → 81,747-vertex mesh    │
│   Output: scene_mesh.ply, scene_raw.ply                     │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ Module 3 — Defect Detection                  [detection/]   │
│   Dual-modal: Laplacian variance + depth gradient           │
│   → Otsu-adaptive threshold → 72,733 defect faces           │
│   Output: defect_mask.npy, defect_score.npy, scene_defect.ply│
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ Module 4 — Texture Generation                 [generation/] │
│   Mesh rendering → ControlNet depth inpainting              │
│   → Multi-view weighted back-projection (3M+ px accumulations)│
│   Output: scene_textured.ply, texture_accum_final.npz       │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ Module 5 — Texture Fusion                        [fusion/]  │
│   Outlier removal → Bilateral mesh filtering                │
│   → Gradient-domain Poisson optimization → Color harmonization│
│   Output: scene_final.ply                                    │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ Module 6 — Evaluation                        [evaluation/]  │
│   PSNR/SSIM per-region analysis (27 viewpoints)             │
│   → Thesis comparison figures                               │
│   Output: evaluation_report.txt, comparison figures          │
└─────────────────────────────────────────────────────────────┘
```

---

## Repository Structure

```
finalproject/
│
├── slam/                          # Module 1: SLAM Frontend
│   ├── ORB_SLAM3/                 #   ORB-SLAM3 source (RGB-D only, cleaned)
│   ├── Pangolin/                  #   ORB-SLAM3 dependency (GUI library)
│   ├── run_slam.sh                #   WSL script: build & run ORB-SLAM3 on TUM dataset
│   ├── export_keyframes.py        #   Optimization 1: structured keyframe export
│   └── select_keyframes.py        #   Optimization 2: FPS coverage-aware selection
│
├── scene/                         # Module 2: Scene Extraction
│   └── extract_scene.py           #   RGB-D back-projection + Poisson reconstruction
│
├── detection/                     # Module 3: Defect Detection
│   └── detect_defects.py          #   Dual-modal Laplacian+depth defect detector
│
├── generation/                    # Module 4: Texture Generation
│   ├── config.py                  #   Paths, model IDs, inference parameters
│   ├── pipeline.py                #   Main orchestrator: render → inpaint → back-project
│   ├── renderer.py                #   Open3D RaycastingScene wrapper (MeshRenderer)
│   ├── inpainter.py               #   ControlNet + Stable Diffusion inpainting pipeline
│   ├── backproject.py             #   ColorAccumulator: multi-view weighted accumulation
│   └── blender.py                 #   TextureBlender: face color fusion + propagation
│
├── fusion/                        # Module 5: Texture Fusion
│   ├── config.py                  #   Optimization parameters (σ_spatial, σ_color, λ_reg)
│   ├── pipeline.py                #   Main orchestrator: 4-stage optimization
│   ├── optimize.py                #   Outlier removal + bilateral mesh filtering
│   ├── boundary.py                #   Gradient-domain Poisson boundary optimization
│   └── harmonize.py               #   Color harmonization via covariance matching
│
├── evaluation/                    # Module 6: Evaluation
│   ├── config.py                  #   Paths, SSIM parameters, sample rate
│   ├── pipeline.py                #   Main orchestrator: render → evaluate → report
│   ├── render_compare.py          #   Multi-mesh rendering + per-viewpoint evaluation
│   ├── metrics.py                 #   PSNR, SSIM, per-region metric computation
│   └── visualize.py               #   Thesis-quality 4-column comparison figures
│
├── scripts/                       # Utility Scripts
│   ├── generate_thesis.py         #   Master thesis DOCX generator
│   ├── generate_figures.py        #   Figure 1-6 generator (architecture, pipeline, etc.)
│   ├── generate_2nd_midcheck.py   #   2nd Midterm Check report generator
│   ├── omml_builder.py            #   OMML equation builder for native Word formulas
│   ├── run_ablation.py            #   Hyperparameter sensitivity ablation study
│   ├── gen_cn_thesis_v2.py        #   Chinese thesis translation generator
│   └── gen_cn_thesis.py           #   (Legacy) Chinese thesis generator
│
├── data/                          # Dataset
│   └── fr2_desk/
│       └── rgbd_dataset_freiburg2_desk/
│           ├── keyframes/         #   270 keyframe RGB + Depth + poses.txt
│           │   ├── rgb/           #     000000.png ... 000269.png
│           │   ├── depth/         #     000000.png ... 000269.png
│           │   └── poses.txt      #     frame_id tx ty tz qx qy qz qw (270 lines)
│           ├── rgb.txt            #     Original TUM RGB timestamp index
│           ├── depth.txt          #     Original TUM depth timestamp index
│           ├── groundtruth.txt    #     Motion-capture ground truth poses
│           └── accelerometer.txt  #     IMU data (unused)
│
├── outputs/                       # Generated Outputs
│   ├── scene_raw.ply              #   (M2) Raw fused point cloud
│   ├── scene_mesh.ply             #   (M2) Final reconstructed mesh (81,747 vertices)
│   ├── defect_mask.npy            #   (M3) Boolean defect mask (235,366 faces)
│   ├── defect_score.npy           #   (M3) Per-face defect scores (float)
│   ├── scene_defect.ply           #   (M3) Defect visualization mesh (red=defect)
│   ├── texture_accum_final.npz    #   (M4) Accumulator checkpoint (color_sum, weight_sum)
│   ├── scene_textured.ply         #   (M4) Textured mesh after AI inpainting
│   ├── scene_final.ply            #   (M5) Final mesh after 4-stage optimization
│   ├── evaluation_report.txt      #   (M6) PSNR/SSIM per-region report
│   ├── comparison_summary.png     #   (M6) Multi-view 3x4 comparison figure
│   ├── comparison_best.png        #   (M6) Best single-view comparison
│   ├── compare_view_*.png         #   (M6) Individual viewpoint comparisons (8 files)
│   │
│   ├── fig1_system_architecture.png   #   Thesis Figure 1
│   ├── fig2_mesh_combined.png         #   Thesis Figure 2
│   ├── fig3_texture_generation.png    #   Thesis Figure 3
│   ├── fig_pipeline_input.png         #   Thesis Figure 4
│   ├── fig6_fusion_stages.png         #   Thesis Figure 5
│   ├── fig4_defect_detection.png      #   Thesis Figure 7
│   ├── fig_mesh_wireframe.png         #   Mesh wireframe visualization
│   ├── fig_mesh_geometry.png          #   Mesh shaded visualization
│   │
│   ├── ablation/                 #   Ablation study results
│   │   ├── ablation_summary.png  #   4-parameter sensitivity plot
│   │   └── ablation_results.json #   Full numerical results (24 experiments)
│   │
│   └── selected_keyframes/
│       └── poses_selected.txt    #   FPS-selected keyframe subset
│
├── papers/                        # Reference papers (PDF)
├── README.md                      # This file
├── requirements.txt               # Python dependencies
├── .gitignore                     # Git exclusion rules
└── project_overview.md            # Development roadmap (Chinese)
```

---

## Module Details

### Module 1 — SLAM Frontend (`slam/`)

| File | Role |
|---|---|
| `ORB_SLAM3/` | ORB-SLAM3 open-source C++ library (RGB-D mode). Processes TUM fr2_desk sequence, outputs 270 keyframe poses. |
| `Pangolin/` | Lightweight 3D visualization library required by ORB-SLAM3. |
| `run_slam.sh` | One-click WSL script: builds ORB-SLAM3, downloads vocabulary, runs on TUM dataset. |
| `export_keyframes.py` | **Optimization 1**: Matches ORB-SLAM3 keyframe timestamps to original RGB-D frames, exports standardized `keyframes/` directory. |
| `select_keyframes.py` | **Optimization 2**: Farthest Point Sampling (FPS) on camera positions. Geometric FPS achieves 408% viewpoint diversity improvement over sequential selection. Optional defect-aware weighting mode. |

### Module 2 — Scene Extraction (`scene/`)

| File | Role |
|---|---|
| `extract_scene.py` | Complete dense reconstruction pipeline. For each of the 270 keyframes: (1) loads RGB-D image pair, (2) back-projects every pixel to 3D via pinhole model, (3) transforms to world coordinates using ORB-SLAM3 pose, (4) fuses ~80M points. Then: voxel downsample (1cm³) → statistical outlier removal → normal estimation → Poisson surface reconstruction (depth 10) → density filtering → QEM decimation (100K faces) → Laplacian smoothing. |

**Outputs:** `scene_raw.ply`, `scene_mesh.ply`

### Module 3 — Defect Detection (`detection/`)

| File | Role |
|---|---|
| `detect_defects.py` | Dual-modal defect detector combining (1) 2D Laplacian variance (texture richness) and (2) 3D Sobel depth gradient (geometric complexity). Projects all 235,366 mesh faces across 54 sampled keyframes. Unified score: `DefectScore = 0.7×(1-tex_norm) + 0.3×(1-geo_norm)`. Otsu-adaptive threshold at 70th percentile. KNN propagation for unobserved faces. |

**Outputs:** `defect_mask.npy` (72,733 True / 235,366), `defect_score.npy`, `scene_defect.ply`

### Module 4 — Texture Generation (`generation/`)

| File | Role |
|---|---|
| `config.py` | Paths, model IDs (`runwayml/stable-diffusion-v1-5`, `lllyasviel/sd-controlnet-depth`), inference parameters (25 steps, CFG=7.5, ControlNet scale=0.85), viewpoint sampling settings. |
| `pipeline.py` | Main orchestrator. 6-step pipeline: load mesh → load poses → select viewpoints → initialize components → per-view render-inpaint-backproject → fusion + save. |
| `renderer.py` | `MeshRenderer` class. Wraps Open3D RaycastingScene. For each viewpoint, casts 640×480 rays and returns 4 maps: `color` (barycentric-interpolated vertex colors), `depth` (meters), `face_id` (triangle index per pixel), `defect_mask` (bool). Also generates depth condition images for ControlNet. |
| `inpainter.py` | `DepthControlNetInpainter` class. Loads ControlNet + Stable Diffusion inpainting pipeline. Preprocesses to 512×512, runs diffusion with depth conditioning, postprocesses back to 640×480. |
| `backproject.py` | `ColorAccumulator` class. Weighted multi-view accumulation: `w = cos(θ) × 1/(1+d²)`. Stores per-face `color_sum`, `weight_sum`, and `sample_count`. 3,040,026 total pixel accumulations across 30 viewpoints. |
| `blender.py` | `TextureBlender` class. Final face color computation (`color_sum/weight_sum`), propagation to unobserved faces, Laplacian boundary smoothing (5 iterations, 2-ring, 0.6/0.4 blend). Face colors → vertex colors conversion. |

**Outputs:** `scene_textured.ply`, `texture_accum_final.npz`

### Module 5 — Texture Fusion (`fusion/`)

| File | Role |
|---|---|
| `config.py` | Optimization parameters: BILATERAL_ITERS=3, SIGMA_SPATIAL=0.8m, SIGMA_COLOR=0.12, POISSON_LAMBDA=0.5, HARMONIZE_STRENGTH=0.4. |
| `pipeline.py` | Main orchestrator. Loads mesh + defect mask + accumulator checkpoint. Builds face adjacency graph. Runs 4-stage optimization. Converts face colors → vertex colors. |
| `optimize.py` | Stage 1+2: (1) `remove_outliers()` — median-based outlier detection with 3σ threshold (1,737 outliers, 2.4%). (2) `bilateral_filter_mesh()` — graph bilateral filter with spatial kernel (face center distance) and range kernel (color difference), 3 iterations, operates on defect faces + 1-ring neighbors. |
| `boundary.py` | Stage 3: `gradient_domain_blend()` — Poisson mesh editing. Constructs graph Laplacian L from face adjacency. Computes divergence `b = div(grad(generated))`. Solves sparse linear system `L·x = b` with Dirichlet BC at non-defect faces. 52,607 free variables solved via `scipy.sparse.linalg.spsolve`. |
| `harmonize.py` | Stage 4: `color_harmonize()` — covariance matching via Cholesky decomposition. `c' = A·(c−μ_gen) + μ_ref` where `A = L_ref · L_gen⁻¹`. Interpolation strength = 0.4. Reduces L2 color diff by 26.0% (0.0823→0.0609). |

**Output:** `scene_final.ply`

### Module 6 — Evaluation (`evaluation/`)

| File | Role |
|---|---|
| `config.py` | Paths to meshes, sample rate (every 10 frames), SSIM parameters (K1=0.01, K2=0.03, window=11). |
| `pipeline.py` | Main orchestrator. Loads original and completed meshes. Evaluates 27 viewpoints. Aggregates results (mean ± std). Generates formatted ASCII report with per-region breakdown. |
| `render_compare.py` | `load_and_prepare_meshes()` — creates MeshRenderer for each mesh variant. `evaluate_viewpoint()` — renders both meshes, computes PSNR/SSIM for full/defect/non-defect regions. `aggregate_results()` — mean/std across all viewpoints. |
| `metrics.py` | `psnr()` — standard PSNR formula. `ssim()` — sliding-window SSIM (11×11 uniform filter). `compute_metrics()` — unified interface supporting optional mask for per-region analysis. |
| `visualize.py` | `create_comparison_figure()` — 4-column comparison (GT | Defect Mask | Original Mesh | Completed Mesh) with optional 2× zoom row. `create_summary_figure()` — 3×4 multi-view thesis figure. `run_visualization()` — selects top 8 viewpoints by defect pixel count, generates individual + summary + best-view figures. |

**Outputs:** `evaluation_report.txt`, `comparison_summary.png`, `comparison_best.png`, `compare_view_*.png` (×8)

---

## Key Quantitative Results

Evaluated on TUM RGB-D fr2_desk, 27 viewpoints, every 10th keyframe:

| Region | PSNR (dB) | ±σ | SSIM | ±σ |
|---|---|---|---|---|
| Original — Full Image | 12.0 | 2.9 | 0.3509 | 0.1111 |
| Original — Defect Region | 15.5 | 1.1 | 0.7084 | 0.0935 |
| Original — Non-Defect | 15.6 | 1.4 | 0.7601 | 0.0662 |
| **Completed — Full Image** | **12.0** | **2.9** | **0.3539** | **0.1085** |
| **Completed — Defect Region** | **15.3** | **1.0** | **0.6786** | **0.1111** |
| **Completed — Non-Defect** | **15.6** | **1.4** | **0.7551** | **0.0659** |

- 72,733 defect faces detected (89.0% of 235,366 total)
- 26.0% L2 color difference reduction through 4-stage fusion
- 408% viewpoint coverage improvement via FPS keyframe selection
- 3,040,026 pixel accumulations across 30 viewpoints (mean 115 obs/face)

---

## Quick Start

### Prerequisites

- Python 3.11 with Anaconda
- CUDA-capable GPU (8GB+ VRAM recommended)
- WSL2 with Ubuntu 22.04 (for Module 1 — ORB-SLAM3 compilation)
- TUM fr2_desk dataset (download from https://vision.in.tum.de/data/datasets/rgbd-dataset)

### Installation

```bash
git clone <repo-url>
cd finalproject
conda create -n slam-texture python=3.11
conda activate slam-texture
pip install -r requirements.txt
```

### Running the Pipeline

```bash
# Step 1: Run ORB-SLAM3 (in WSL)
wsl -d Ubuntu -e bash slam/run_slam.sh

# Step 2: Extract standardized keyframes
python slam/export_keyframes.py

# Step 3 (optional): Coverage-aware keyframe selection
python slam/select_keyframes.py --n_views 100

# Step 4: Dense scene reconstruction
python scene/extract_scene.py

# Step 5: Defect detection
python detection/detect_defects.py

# Step 6: AI texture generation (requires GPU)
python generation/pipeline.py

# Step 7: Texture fusion optimization
python fusion/pipeline.py

# Step 8: Evaluation + visualization
python evaluation/pipeline.py
python evaluation/visualize.py
```

---

## Citation

```bibtex
@mastersthesis{jin2026slamtexture,
  title   = {SLAM Map Texture Completion Based on Generation AI},
  author  = {xheoKim},
  school  = {City University of Hong Kong},
  year    = {2026}
}
```

## License

This project is for academic research purposes. ORB-SLAM3 is licensed under GPLv3.
