
# SLAM Map Texture Completion — 项目框架总览

> 基于 ORB-SLAM3 + ControlNet LoRA 的 SLAM 地图纹理补全毕设项目

---

## 一、技术选型

| 层次 | 选择 | 理由 |
|------|------|------|
| SLAM | ORB-SLAM3 | 实时 RGB-D，关键帧+位姿输出完整，发表较新(2021) |
| 数据集 | TUM RGB-D `fr2_desk` | 室内办公桌场景，大片白墙低纹理，GT 完整，~2GB |
| 生成模型 | SD v1.5 + ControlNet Canny | 生态成熟，LoRA 资源丰富 |
| 微调 | LoRA (PEFT, rank=16) | 8GB 显存可训练，参数 <1M |
| 3D 处理 | Open3D | Python 原生，点云/网格/可视化一站式 |
| 评估 | PSNR + SSIM + ICP 误差 | 论文标准指标 |

### 硬件

| 组件 | 配置 | 判定 |
|------|------|------|
| GPU | RTX 5060 8GB | 刚好够，需 fp16 + gradient_accumulation |
| RAM | 16GB | 偏紧，跑 TUM 小场景没问题 |

---

## 二、系统架构

```
输入: TUM fr2_desk RGB-D 序列
              │
              ▼
┌──────────────────────────────────────┐
│ 模块① SLAM 前端 (ORB-SLAM3)          │
│ · 逐帧跟踪相机位姿                    │
│ · 筛选关键帧                          │
│ · 生成稀疏点云                        │
│ 输出: 关键帧 + 6DoF位姿 + 稀疏点云    │
└──────────────┬───────────────────────┘
               ▼
┌──────────────────────────────────────┐
│ 模块② 场景提取 (Open3D)              │
│ · 体素下采样 → 滤波 → Poisson重建     │
│ 输出: 三角网格 Mesh                   │
└──────────────┬───────────────────────┘
               ▼
┌──────────────────────────────────────┐
│ 模块③ 缺陷检测 ★核心创新             │
│ · Laplacian方差 (2D纹理稀疏度)        │
│ · 深度梯度模长 (3D几何平坦度)         │
│ · 融合分数 → Otsu阈值 → 2D mask      │
│ · 投影回3D → 3D mask                 │
│ 输出: 低纹理三角面片集合              │
└──────────────┬───────────────────────┘
               ▼
┌──────────────────────────────────────┐
│ 模块④ 纹理生成 (ControlNet + LoRA)   │
│ · LoRA微调: 关键帧训练轻量适配器       │
│ · UV展开: 网格区域做UV映射            │
│ · Inpainting: 扩散模型补全纹理        │
│ 输出: 补全后的纹理贴图                │
└──────────────┬───────────────────────┘
               ▼
┌──────────────────────────────────────┐
│ 模块⑤ 纹理融合 (多视图投影 + 优化)   │
│ · 多视图投影 → Poisson融合去接缝     │
│ 输出: 带完整纹理的3D网格              │
└──────────────┬───────────────────────┘
               ▼
┌──────────────────────────────────────┐
│ 模块⑥ 定量评估                       │
│ · PSNR (>25dB合格)                   │
│ · SSIM (>0.85合格)                   │
│ · ICP 配准误差                        │
│ 输出: 数值报告 + 可视化对比           │
└──────────────────────────────────────┘
```

---

## 三、各模块要理解的核心概念

| 模块 | 输入 → 输出 | 关键概念 |
|------|-------------|----------|
| ① SLAM | RGB-D → 关键帧+位姿+点云 | ORB特征、共视关系、Bundle Adjustment |
| ② 场景 | 稀疏点云 → 三角网格 | 体素下采样、法向量、Poisson重建 |
| ③ 检测 | 帧+网格 → 低纹理mask | 图像梯度、Laplacian算子、Otsu阈值、KNN传播 |
| ④ 生成 | mask+网格 → 纹理贴图 | Diffusion去噪、ControlNet条件控制、LoRA低秩适配 |
| ⑤ 融合 | 纹理+网格+位姿 → 完整模型 | UV映射、透视投影矩阵、Poisson图像融合 |
| ⑥ 评估 | 原始/补全后 → 数值 | PSNR公式、SSIM公式、ICP迭代配准 |

---

## 四、推荐学习路径（2周）

```
Week 1
  Day 1-2: SLAM 基础
    · ORB-SLAM3 论文 Introduction + System Overview
    · 理解"前端跟踪 / 后端优化 / 回环检测"三线程分工

  Day 3-4: 3D 基础
    · Open3D 官方 Tutorial（点云读入、可视化、法向量估计）
    · 理解点云、网格、voxel 是什么

  Day 5: 缺陷检测直觉
    · OpenCV 跑 Laplacian 算子看输出
    · 直观感受"纹理丰富"和"纹理稀疏"的数值差异

Week 2
  Day 6-8: 生成式 AI
    · Stable Diffusion 原理（前向加噪 → 反向去噪）
    · ControlNet 论文摘要 + 示例图（Canny/Depth 条件的作用）
    · LoRA 原理（矩阵分解，低秩替代全参数微调）

  Day 9-10: 纹理融合 + 评估
    · UV 映射概念（3D表面"展开"成2D平面）
    · PSNR/SSIM 定义公式

  Day 11-14: 串联理解
    · 画全模块数据流图，确认每个模块"为什么需要上个模块的输出"
    · 用简化的数据（图片、单个网格）模拟数据流转
```

---

## 五、推荐学习资源

| 领域 | 资源 |
|------|------|
| ORB-SLAM3 | 论文: "ORB-SLAM3: An Accurate Open-Source Library for Visual, Visual-Inertial and Multi-Map SLAM" (T-RO 2021) |
| Open3D | http://www.open3d.org/docs/release/tutorial/ |
| Stable Diffusion | "High-Resolution Image Synthesis with Latent Diffusion Models" (CVPR 2022) |
| ControlNet | "Adding Conditional Control to Text-to-Image Diffusion Models" (ICCV 2023) |
| LoRA | "LoRA: Low-Rank Adaptation of Large Language Models" (ICLR 2022) |
| TUM 数据集 | https://vision.in.tum.de/data/datasets/rgbd-dataset |
| ORB-SLAM3 仓库 | https://github.com/UZ-SLAMLab/ORB_SLAM3 |

---

## 六、目录结构（建议）

```
finalproject/
├── project_overview.md          # 本文件
├── data/
│   └── fr2_desk/               # TUM 数据集（需下载）
├── slam/                        # 模块①: ORB-SLAM3 封装
├── scene/                       # 模块②: 场景提取
├── detection/                   # 模块③: 缺陷检测
├── generation/                  # 模块④: 纹理生成 (LoRA + ControlNet)
├── fusion/                      # 模块⑤: 纹理融合
├── evaluation/                  # 模块⑥: 定量评估
├── scripts/                     # 端到端运行脚本
├── notebooks/                   # Jupyter 实验笔记
└── outputs/                     # 输出结果
```
