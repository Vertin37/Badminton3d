# Badminton3d

面向单目羽毛球比赛视频的视觉分析与三维人体运动恢复项目。项目当前聚焦于建立一条可复现的 Windows 原生流程：从双人二维姿态和球场几何出发，恢复稳定的球员轨迹，并验证 world-grounded human motion recovery 在羽毛球快速移动与起跳动作中的可用性。

## 发布范围与完整三维入口

本次更新包含稳定双人二维姿态与身份跟踪、官方 GVHMR/SMPL-X 三维恢复、
Metric Video Depth Anything 时序深度、整段多线索轨迹融合、支撑脚接触修正和 Open3D 回放。
保留这些流程依赖的传统球场几何标定接口。

近期独立研究的球场学习模型、数据采集与标注工具、训练/评测记录、多视角球场识别和
Good-Badminton 对照实验未包含在本次发布中。原视频、模型权重、标注数据、推理缓存和
生成结果继续由本地保存。

已准备好本机三个独立环境及官方模型后，可运行：

```powershell
python work/run_badminton_3d_pipeline.py --video path/to/input.mp4 --output-dir path/to/new_output
```

三个解释器可通过 `--badminton-python`、`--gvhmr-python`、`--vda-python` 指定；
默认按固定镜头处理，移动镜头使用 `--moving-camera`。命令会检查输入和阶段输出，
仍需视频具备可用的球员观测与球场标定，不保证任意机位都能成功恢复。

详细环境、输入输出和运行步骤见[稳定三维流程](docs/stable_3d_vda_pipeline.md)。
[深度约束说明](docs/depth_constraint_pipeline.md)及[早期深度对照实验](docs/depth_constraint_experiment.md)
保留了方法边界和历史结果；文中的“不上传”记录描述的是当时实验状态。

## 当前进度

### 已完成：稳定二维人体与球场分析

- 基于 RTMLib WholeBody 的双人二维关键点提取；
- 跨帧持续球员 ID、时序滤波和稳定姿态 CSV；
- 基于网柱、球网和场地线的球场检测、地面 Homography 与时间稳定跟踪；
- 输出球员—网柱的地面距离、方位和张角等结构化几何数据；
- 已生成稳定的 `court_detection_video_stable_geometry.mp4`，用于回看球场、人体和坐标标定的一致性。

### 已完成：Windows 原生 GVHMR 三维可行性验证

- 独立 Conda 环境 `gvhmr_win`，不修改原有二维环境；
- 使用官方 GVHMR、SimpleVO 与官方 SMPL-X 模型，在 RTX 4060 Laptop GPU 上完成推理；
- 对完整测试视频恢复两名球员的时序 SMPL-X 网格与 root/global trajectory；
- 将相机坐标中的人体网格接入稳定球场坐标，修正了平面 PnP 的上下法向和二维轨迹纵向约定；
- 可在 Open3D 中自由查看完整三维球场、两名球员和动画，并重点保留起跳—腾空—落地期间的原生 GVHMR 根节点位移。

### 当前结论与限制

这不是多机位动作捕捉。单目视频在遮挡、快速挥拍、远距离人物和绝对尺度上仍存在不确定性；当前三维结果用于验证连续人体姿态和地面轨迹的可行性，而非声称达到逐关节的真实三维测量精度。后续将以稳定二维球场标定为约束，继续评估跳跃轨迹、击球事件和球场坐标下的运动分析。

## 目录

```text
src/
  court_detection/          稳定球场、网柱和地面几何
  pose_biomechanics/        二维人体姿态、持续 ID 与时序处理
  calibration_reconstruction/  相机与场地几何实验
  segmentation_tracking/    羽毛球/目标跟踪实验
  llm_report/               结构化分析报告
work/
  rtmlib_to_gvhmr_adapter.py          二维结果到 GVHMR 输入的适配
  run_gvhmr_native_mesh_slice.py      官方 GVHMR + SMPL-X 网格推理
  build_gvhmr_two_player_court_bundle.py  双人球场坐标融合
  view_gvhmr_two_player_court.py      Open3D 自由查看器
docs/                       流程与代码说明
```

## 环境与运行

二维流程使用既有的 `badminton3d` Conda 环境；GVHMR 使用独立的 `gvhmr_win` 环境。以下是当前完整三维结果的查看方式：

```bat
conda activate gvhmr_win
cd /d D:\Projects\BadmintonPose
python work\view_gvhmr_two_player_court.py
```

模型权重、SMPL-X 文件、原始视频、缓存和生成结果均不提交到仓库。请使用官方 GVHMR 与 SMPL-X 下载渠道获取所需文件，并按本机路径配置后运行。

## 下一步

- 对起跳、腾空与落地片段做定量连续性评估；
- 将球员轨迹与羽毛球事件分析接入统一的球场坐标；
- 评估单目三维结果在不同机位、遮挡和运动强度下的稳定性。

## 文档

- [代码概览](docs/code_overview.md)
- [球场检测与稳定几何](docs/court_detection.md)
