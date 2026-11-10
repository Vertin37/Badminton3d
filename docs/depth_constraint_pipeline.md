# 通用 GVHMR 单目深度轨迹约束程序

> **旧版说明（2026-09-18）**：本页记录的是第一版逐帧 YOLO Depth 实验入口。该默认模型与隔离 vendor 已在官方 Metric Video Depth Anything 通过验证后移除；新任务请使用 `docs\stable_3d_vda_pipeline.md` 和 `work\video_depth_anything\run_vda_multicue_pipeline.py`。旧入口仍可读取已有 `person_depth.csv`，但不再自带 YOLO 权重。

## 1. 用途与输入边界

生产入口：

```text
work\depth_estimation\run_depth_constrained_3d.py
```

该程序用于处理任意已经进入本项目稳定二维/GVHMR 流程的视频。它不是针对 `test2` 手工改轨迹，也不会调用语言模型修改结果；每次运行都会自动完成逐帧深度估计、人物 torso/pelvis 深度提取、时间对齐、鲁棒尺度/偏移拟合、连续运动判别、轨迹优化和完整 SMPL-X bundle 导出。

每个输入视频需要四类同步输入：

1. 原视频或与二维关键点同尺寸、同帧号的视频；
2. 稳定二维 pose CSV，至少包含 `frame_id/player_id/keypoint_id/x/y/confidence`；
3. 既有 GVHMR 球场 bundle，包含每名球员的 `pN_root_court` 和 `pN_vertices_court`；
4. 对应的 `scene_metadata.json`，包含 `court_to_camera_R/t`。

也就是说，“任意输入视频”指任何能够由现有二维 + GVHMR 基线生成上述同步结果的视频。程序不会假设固定文件名、固定 450 帧、固定 `test2` 路径或固定两名球员编号。

## 2. 约束原则

单目 Depth 只负责判断朝向/远离相机的相对距离趋势，并约束短时不合理漂移；GVHMR 继续负责 SMPL-X 姿态、横向运动和垂直跳跃。

自动判别使用连续分数，不使用手写 `AIRBORNE` 状态机：

- `travel_score`：完全由清洗后的 Depth 时间变化计算，持续深度变化视为真实前后移动；
- `jump_score`：GVHMR 垂直运动明显、同时 Depth 前后距离稳定时升高；
- `drift_score`：GVHMR 相机深度与 Depth 趋势出现短时速度或残差异常时升高；
- `foot_stationary_confidence`：只作为连续权重。接触置信度高时增强原 GVHMR/球场先验，接触置信度低时适度增强 Depth 支持。

优化目标可概括为：

```text
E = E_depth + E_GVHMR + E_acceleration + E_jerk
```

其中 Depth 先经过 Hampel 离群抑制、Savitzky–Golay 低频清洗和正尺度鲁棒 affine 对齐。真实持续前后移动会降低加速度/jerk 惩罚；孤立尖峰不会直接写入轨迹。

最重要的几何保护是：程序只改变球场坐标的 XY，所施加的 XY 位移在相机坐标中实现所需的 camera-Z 修正；球场 Z、root 高度和所有顶点高度逐值保持不变。因此正常跳跃不会被 Depth 压平，也绝不执行 `root_z = depth`。

## 3. Windows 环境

在 PowerShell 中从项目根目录运行：

```powershell
Set-Location D:\Projects\BadmintonPose
```

默认使用已经稳定的 `gvhmr_win` Python：

```text
D:\miniconda\envs\gvhmr_win\python.exe
```

Depth 模型和 Ultralytics 代码均为项目内隔离资源：

```text
work\depth_estimation\model_cache\yolo26n-depth.pt
work\depth_estimation\vendor\
```

Ultralytics 可变配置会写入每次运行的 `output-dir\ultralytics_config\`，不会修改全局环境或污染项目根目录。程序支持 `--device cpu`，默认 `--device 0` 使用首块 CUDA GPU。

## 4. 完整自动运行

下面命令会处理输入之间全部重叠帧。输出必须使用新目录：

```powershell
& D:\miniconda\envs\gvhmr_win\python.exe `
  work\depth_estimation\run_depth_constrained_3d.py `
  --video D:\path\to\input.mp4 `
  --pose-csv D:\path\to\pose_data_stable.csv `
  --gvhmr-bundle D:\path\to\two_player_smplx_court_animation.npz `
  --scene-metadata D:\path\to\scene_metadata.json `
  --output-dir D:\path\to\new_depth_constraint_output
```

建议先做 50–100 帧验收：

```powershell
& D:\miniconda\envs\gvhmr_win\python.exe `
  work\depth_estimation\run_depth_constrained_3d.py `
  --video outputs\test2_analysis\court_detection_video_stable_geometry.mp4 `
  --pose-csv outputs\test2_analysis\pose_data_stable.csv `
  --gvhmr-bundle outputs\world_hmr_native_mesh_full\two_player_court_full\two_player_smplx_court_animation.npz `
  --scene-metadata outputs\world_hmr_native_mesh_full\two_player_court_full\scene_metadata.json `
  --output-dir outputs\depth_constraint_new_video `
  --start-frame 361 `
  --end-frame 435 `
  --bundle-start-frame 1
```

若已有 `person_depth.csv`，可跳过 GPU 估深并完全复用观测：

```powershell
& D:\miniconda\envs\gvhmr_win\python.exe `
  work\depth_estimation\run_depth_constrained_3d.py `
  --video D:\path\to\input.mp4 `
  --person-depth-csv D:\path\to\person_depth.csv `
  --gvhmr-bundle D:\path\to\two_player_smplx_court_animation.npz `
  --scene-metadata D:\path\to\scene_metadata.json `
  --output-dir D:\path\to\new_output
```

如果 GVHMR bundle 的索引 0 对应原视频第 101 帧，应传 `--bundle-start-frame 101`。程序没有覆盖开关；每次运行必须使用新的空输出目录。

## 5. 输出

默认输出：

```text
<output-dir>\depth_preview.mp4
<output-dir>\person_depth.csv
<output-dir>\depth_alignment.csv
<output-dir>\depth_constraint_diagnostics.csv
<output-dir>\comparison_metrics.json
<output-dir>\ab_manifest.json
<output-dir>\depth_constrained_smplx_court_animation.npz
<output-dir>\scene_metadata.json
<output-dir>\diagnostics\depth_constraint_overview.png
<output-dir>\ultralytics_config\
```

`ab_manifest.json` 中：

- A 是只读的原 GVHMR bundle；
- B 是新的 `depth_constrained_smplx_court_animation.npz`。

B 保留原 bundle 的全部键和全部帧，只替换选定帧内每名球员 root/mesh 的 XY 平移，并增加 Depth 诊断数组；现有双人球场 viewer 所需的 `p0/p1_vertices_court`、`faces`、球场尺寸等键不丢失。使用 `--no-preview` 可关闭视频预览；只有显式传入 `--save-raw-depth` 才保存体积较大的 `depth_raw.npy`。

查看 B 三维结果：

```powershell
& D:\miniconda\envs\gvhmr_win\python.exe `
  work\view_gvhmr_two_player_court.py `
  --bundle D:\path\to\new_output\depth_constrained_smplx_court_animation.npz
```

## 6. 自动失败保护

- 输出目录不能与视频、pose、GVHMR bundle、metadata 或复用 Depth CSV 所在目录相同；
- 非空输出目录默认拒绝写入；
- 原 GVHMR bundle 永远只读，不会原地覆盖；
- 视频、pose、bundle 帧范围自动取交集，FPS 不匹配会停止；
- pose/Depth CSV 缺少必要列时会给出明确错误；
- 某名球员有效深度不足时，仅该球员自动执行 `skipped_safe_identity`，完整保留其 GVHMR 结果，不影响其他球员；
- 单次修正、修正速度、相对 Depth 趋势偏差和球场边界都有上限；
- 不修改二维稳定环境，不重新运行或替换官方 GVHMR，不使用 Homography 锁脚、逐肢体 PnP 或离散跳跃状态机。

## 7. 已验证结果

在 `test2` 的 361–435 帧上，最终通用实现得到：

| 球员 | Depth RMSE（前→后） | mean jerk（前→后） | 短时漂移尖峰（前→后） | root/mesh Z | 球场内比例 |
|---|---:|---:|---:|---:|---:|
| P0 | 0.2835→0.0955 m | 4303.9→3013.8 m/s³ | 58→29 | 逐值不变 | 100%→100% |
| P1 | 0.2310→0.0556 m | 5356.3→3450.1 m/s³ | 39→16 | 逐值不变 | 100%→100% |

自动估深链路另用 8 帧独立烟雾测试通过，Ultralytics 8.4.137、Torch 2.3.0+cu121、CUDA 可用，GPU 峰值显存约 207 MB。

当前结论仍是：

```text
C. Depth 适合作为低频尺度或 drift 约束，不应直接替代 GVHMR。
```

## 8. 测试与调参

运行核心测试：

```powershell
& D:\miniconda\envs\gvhmr_win\python.exe -m unittest discover `
  -s tests `
  -p test_depth_constraint.py `
  -v
```

可安全覆盖的主要参数为：

```text
--max-correction-m
--max-correction-speed-mps
--max-unexplained-depth-speed-mps
--depth-weight
--gvhmr-weight
```

调参应以 `comparison_metrics.json`、`depth_constraint_overview.png` 和 A/B 三维可视结果共同判断。不要只为降低某一个 jerk 数值而放大 Depth 权重。

## 9. 源码结构

```text
src\pose_biomechanics\depth_constraint.py
    可复用的清洗、对齐、连续分数、稀疏优化和球场坐标修正核心

work\depth_estimation\run_depth_constrained_3d.py
    任意路径/帧数/球员数量的生产命令入口

tests\test_depth_constraint.py
    尖峰、跳跃、持续移动、离群点、脚部接触和双人发现测试
```

本程序只在本地分支工作，不包含任何 GitHub 上传逻辑。
