# 稳定三维人物轨迹：Static GVHMR + VDA + Multi-cue

## 1. 当前生产链路

这套程序面向任何已经通过项目稳定二维流程、并生成同步 GVHMR 输入/球场标定的视频。程序自动计算结果，不调用语言模型手工改帧。

```text
固定镜头：官方 GVHMR static_cam=True
    ↓
双人 SMPL-X 球场包（接触修正限速）
    ↓
官方 Metric Video Depth Anything Small（时序深度）
    ↓
torso/pelvis 人物深度提取
    ↓
整段 multi-cue 稀疏优化
    ↓
连续支撑脚锁定 + SMPL-X 脚底球场平面对齐
    ↓
新的、viewer 兼容的双人 SMPL-X 球场包
```

关键原则：

- Depth 是低频距离/drift 辅助约束，不替代 GVHMR；
- 固定机位必须使用 GVHMR 官方 static-camera 分支，不能注入 SimpleVO 的假相机旋转；
- 2D pelvis 射线、GVHMR 位移、连续脚接触、球场观测、VDA Depth、加速度和 jerk 同时参与整段优化；
- 禁止 `root_z = depth`；球场 Z、跳跃高度和局部 SMPL-X 姿态保持不变；
- 落脚观测的单帧修正有速度上限，噪声脚点不能再把整个人拉出一帧瞬移；
- 脚底顶点由官方 SMPL-X skinning weights 自动确定，不使用旧 SMPL 脚点编号；
- 只有“GVHMR 接触置信度高、脚部局部动作小、相对另一只候选脚接近支撑面”的脚才参与连续锁定；悬空静止脚、腾空与技术动作不会被强制贴地；
- 支撑脚下表面以连续软约束对齐球场 `Z=0`，不逐帧硬改 `root_z`，也不改局部 SMPL-X 姿态；
- 每次必须写入新的空输出目录，原始成功结果只读。

## 2. Windows 环境

```powershell
Set-Location D:\Projects\BadmintonPose
```

稳定三维/融合环境：

```text
D:\miniconda\envs\gvhmr_win\python.exe
```

时序深度独立环境：

```text
D:\miniconda\envs\vda_win\python.exe
```

VDA 使用官方仓库提交 `4f5ae23172ba60fd7bc11ef671cca678842c7072` 和 Apache-2.0 的 Metric Small 权重。它与 `badminton3d`、`gvhmr_win` 完全隔离。

## 3. 固定镜头 GVHMR

对每名球员复用其已有 GVHMR 缓存，写入新目录。示例：

```powershell
& D:\miniconda\envs\gvhmr_win\python.exe `
  work\run_gvhmr_native_mesh_slice.py `
  --video examples\data\input.mp4 `
  --cache-root D:\path\to\player_cache `
  --output-dir D:\path\to\new_static_player_result `
  --start-frame0 0 `
  --end-frame0 450 `
  --static-camera
```

只有真实移动镜头才省略 `--static-camera` 并使用 SimpleVO。

双人球场包：

```powershell
& D:\miniconda\envs\gvhmr_win\python.exe `
  work\build_gvhmr_two_player_court_bundle.py `
  --p0-result D:\path\to\p0\1_hmr4d_results_native.pt `
  --p1-result D:\path\to\p1\1_hmr4d_results_native.pt `
  --output-dir D:\path\to\new_two_player_court `
  --source-start-frame0 0 `
  --anchor-frame0 224
```

打包器会把脚接触修正速度限制为默认 `1.5 m/s`。如需调整，使用 `--max-contact-correction-speed-mps`，不要放宽成无界逐帧跳转。

## 4. 一条命令运行 VDA 与轨迹优化

建议先用 50–100 帧深度窗口验收；轨迹融合仍可覆盖完整 bundle：

```powershell
& D:\miniconda\envs\gvhmr_win\python.exe `
  work\video_depth_anything\run_vda_multicue_pipeline.py `
  --video D:\path\to\input.mp4 `
  --pose-csv D:\path\to\pose_data_stable.csv `
  --gvhmr-bundle D:\path\to\two_player_smplx_court_animation.npz `
  --scene-metadata D:\path\to\scene_metadata.json `
  --output-dir D:\path\to\new_vda_multicue_output `
  --depth-start-frame 361 `
  --depth-end-frame 435 `
  --fusion-start-frame 1 `
  --fusion-end-frame 450 `
  --bundle-start-frame 1
```

输入可以是任意路径、任意帧数和任意 `pN` 球员编号；前提是视频、pose CSV、GVHMR bundle 与标定 metadata 同步。若要对全视频估深，省略 `--depth-end-frame`，并按视频实际情况设置起始帧。

若已存在可信 `person_depth.csv`，可复用并跳过 VDA GPU 推理：

```powershell
& D:\miniconda\envs\gvhmr_win\python.exe `
  work\video_depth_anything\run_vda_multicue_pipeline.py `
  --video D:\path\to\input.mp4 `
  --pose-csv D:\path\to\pose_data_stable.csv `
  --gvhmr-bundle D:\path\to\two_player_smplx_court_animation.npz `
  --scene-metadata D:\path\to\scene_metadata.json `
  --person-depth-csv D:\path\to\person_depth.csv `
  --output-dir D:\path\to\new_output
```

## 5. 输出

```text
<output-dir>\pipeline_manifest.json
<output-dir>\depth\depth_raw.npy
<output-dir>\depth\depth_preview.mp4
<output-dir>\depth\depth_quality.json
<output-dir>\depth\person_depth.csv
<output-dir>\depth\person_depth_quality.json
<output-dir>\fusion\multicue_trajectory_fused_smplx_court_animation.npz
<output-dir>\fusion\comparison_metrics.json
<output-dir>\fusion\trajectory_fusion_diagnostics.csv
<output-dir>\fusion\diagnostics\trajectory_fusion_overview.png
<output-dir>\foot_contact\foot_contact_refined_smplx_court_animation.npz
<output-dir>\foot_contact\foot_contact_diagnostics.csv
<output-dir>\foot_contact\foot_contact_metrics.json
<output-dir>\foot_contact\scene_metadata.json
```

查看最终三维结果：

```powershell
& D:\miniconda\envs\gvhmr_win\python.exe `
  work\view_gvhmr_two_player_court.py `
  --bundle D:\path\to\new_output\foot_contact\foot_contact_refined_smplx_court_animation.npz
```

操作键：空格播放/暂停，`A`/`D` 前后逐帧，`R` 回到第一帧，鼠标拖动旋转，滚轮缩放。

## 6. test2 验证结果

75 帧 VDA Metric Small 测试：

- 分辨率：1280×720；输入尺寸：392；有效深度比例：100%；
- RTX 4060 Laptop GPU 峰值分配显存：约 1715 MB；
- P0/P1 各 75/75 条 torso/pelvis 深度可用；
- 对齐残差尺度：P0 `0.367 m`、P1 `0.292 m`，优于旧 YOLO 的 `0.567 m`、`0.421 m`。

最终 Static GVHMR + 接触限速 + VDA multi-cue 的整段指标：

| 球员 | 最大 root 速度 | p95 root 速度 | p95 jerk | >7.5 m/s 尖峰 | 球场内比例 |
|---|---:|---:|---:|---:|---:|
| P0 | 13.51→7.47 m/s | 5.70→3.97 m/s | 7240→2526 m/s³ | 6→0 | 100% |
| P1 | 8.69→5.10 m/s | 4.67→2.60 m/s | 8100→2133 m/s³ | 3→0 | 100% |

root/mesh Z 全部保持，局部 SMPL-X 姿态未被平滑或重写。右肘的少数大角速度与挥拍时序一致，因此没有把真实挥拍当作“抽动”强行抹平。

在 test2 的最终脚接触阶段：

| 球员 | 静止脚每帧位移中位数 | 静止脚位移 p95 | 支撑脚底高度误差中位数 | 脚底高度误差 p95 |
|---|---:|---:|---:|---:|
| P0 | 3.20→0.83 cm | 8.67→2.94 cm | 11.02→0.16 cm | 37.03→0.91 cm |
| P1 | 2.88→0.78 cm | 8.71→1.98 cm | 12.94→0.15 cm | 23.29→0.77 cm |

少数接触切换帧的最大残余脚位移为 P0 `4.10 cm`、P1 `3.29 cm`；对应的最终 root 最大速度仍低于 `6.0 m/s`，没有重新引入整人瞬移。脚接触阶段只做整帧刚性平移，因此保留 GVHMR 的真实跳跃和足踝姿态；如果后续验收确认是脚掌倾角本身错误，应单独加入足踝旋转优化，不能用平移伪装修复。

当前结论：

```text
C. Depth 适合作为低频尺度或 drift 约束；
主要瞬移应优先由 static-camera GVHMR、接触修正限速和整段几何/时间优化解决。
```

旧 `yolo26n-depth.pt`、旧 Ultralytics vendor 和失败的 VDA 半成品目录已删除。历史成功输出和 CSV 均保留。

## 7. 安全与测试

- 程序拒绝非空输出目录，不覆盖输入；
- 固定镜头与移动镜头的 GVHMR 模式由调用方明确选择；
- Depth 缺失帧自动降低权重，不强拉轨迹；
- 脚接触缺失或脚部正在运动时自动降低锁定权重，不把跳跃强压到地面；
- 输出保留原 bundle 的 viewer 必需键；
- 没有 GitHub push、PR、remote branch 或 release 逻辑。

运行全部测试：

```powershell
& D:\miniconda\envs\gvhmr_win\python.exe -m unittest discover `
  -s tests `
  -p "test_*.py" `
  -v
```
