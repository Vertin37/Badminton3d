# GVHMR + 单目 Depth 约束实验报告

## 1. 实验范围

- 项目：`D:\Projects\BadmintonPose`
- 平台：Windows 原生
- 输入：`outputs\test2_analysis\court_detection_video_stable_geometry.mp4`
- 二维输入：`outputs\test2_analysis\pose_data_stable.csv`
- 帧范围：361–435（75 帧，30 FPS）
- 选择理由：该窗口包含当前 GVHMR 基线中最明显的候选跳跃/腾空变化，适合作为第一版短片段验收窗口。
- GVHMR 基线：官方 GVHMR、`postproc=True`、SimpleVO、官方 SMPL-X；只读使用既有双人 SMPL-X、root/global trajectory 和球场坐标融合结果。
- 原始基线保护：`outputs\world_hmr_native_mesh_full\two_player_court_full\two_player_smplx_court_animation.npz` 未覆盖。

## 2. Depth 模型与输出语义

- 模型：官方文档当前示例的 `yolo26n-depth.pt`
- 包版本：隔离 vendor 目录中的 Ultralytics 8.4.137
- Torch：2.3.0+cu121
- CUDA：可用；NVIDIA GeForce RTX 4060 Laptop GPU
- API：`YOLO(...).predict(...)`，结果通过 `result.depth.data` 读取
- 输入/输出：75 帧、1280×720；原始深度输出为 `(75, 720, 1280)` 的 float32；所有像素均为有限正值。
- 模型包位置：`work\depth_estimation\vendor\`；权重位置：`work\depth_estimation\model_cache\yolo26n-depth.pt`
- 官方参考：[Ultralytics Monocular Depth Estimation](https://docs.ultralytics.com/tasks/depth)

## 3. 阶段产物

```text
outputs\depth_constraint_test2\depth_preview.mp4
outputs\depth_constraint_test2\depth_raw.npy
outputs\depth_constraint_test2\depth_quality.json
outputs\depth_constraint_test2\person_depth.csv
outputs\depth_constraint_test2\person_depth_quality.json
outputs\depth_constraint_test2\depth_alignment.csv
outputs\depth_constraint_test2\depth_alignment_summary.json
outputs\depth_constraint_test2\comparison_metrics.json
outputs\depth_constraint_test2\ab_results\A_gvhmr_only\trajectory_bundle.npz
outputs\depth_constraint_test2\ab_results\B_depth_soft_constraint\trajectory_bundle.npz
outputs\depth_constraint_test2\diagnostics\depth_preview_frame_390.png
outputs\depth_constraint_test2\diagnostics\ab_trajectory_alignment.png
```

两名球员均在 75/75 帧得到 torso/pelvis ROI 深度，ROI 有效像素比例中位数均为 1.0，未出现缺失/遮挡标记。

## 4. GVHMR 对齐诊断

对齐形式为：

```text
root_camera_z_m = scale * depth_median + offset
```

| 球员 | correlation | scale | offset (m) | 对齐残差 RMSE (m) | median abs residual (m) | depth 每帧变化中位数 (m) | root 每帧变化中位数 (m) |
|---|---:|---:|---:|---:|---:|---:|---:|
| P0 | 0.497 | 0.363 | 11.706 | 0.306 | 0.232 | 0.232 | 0.040 |
| P1 | 0.235 | 0.125 | 17.034 | 0.234 | 0.104 | 0.278 | 0.027 |

解释：Depth 与 GVHMR root camera depth 存在一定低频关系，但相关性不强，且两名球员的 scale 明显不同；Depth 的逐帧抖动远大于 GVHMR root 的逐帧变化。因此不适合直接作为高频位置替代量。

## 5. A/B 方案

- A：原始 GVHMR 短片段。
- B：对已做 scale/offset 对齐的 Depth 使用 15 帧移动平均，只沿标定相机 Z 方向做低频 drift correction；修正权重为 foot stationary/contact confidence 与 ROI quality 的连续函数。
- 未使用离散 AIRBORNE 状态机。
- 未执行 `root_z = depth`。
- B 的修正限制为相机深度方向 ±0.75 m，并保留 GVHMR 作为主结果和缺失/低质量帧回退。

## 6. A/B 指标

### 6.1 P0

| 指标 | A：GVHMR | B：Depth soft constraint | 变化 |
|---|---:|---:|---:|
| temporal jerk mean (m/s³) | 4303.91 | 3374.08 | -21.6% |
| temporal jerk median (m/s³) | 2269.57 | 1727.17 | -23.9% |
| depth consistency RMSE (m) | 0.2834 | 0.1820 | -35.7% |
| depth consistency median abs error (m) | 0.2412 | 0.1553 | -35.6% |
| ground contact median abs min-vertex Z (m) | 0.1557 | 0.1477 | 小幅改善 |
| contact frames below -0.05 m | 0.0% | 0.0% | 不变 |
| court XY inside ratio | 100.0% | 100.0% | 不变 |

### 6.2 P1

| 指标 | A：GVHMR | B：Depth soft constraint | 变化 |
|---|---:|---:|---:|
| temporal jerk mean (m/s³) | 5356.28 | 4828.80 | -9.8% |
| temporal jerk median (m/s³) | 2569.14 | 2248.17 | -12.5% |
| depth consistency RMSE (m) | 0.2298 | 0.1675 | -27.1% |
| depth consistency median abs error (m) | 0.0811 | 0.0584 | -28.0% |
| ground contact median abs min-vertex Z (m) | 0.0223 | 0.0263 | 小幅变差 |
| contact frames below -0.05 m | 12.5% | 12.5% | 不变 |
| court XY inside ratio | 100.0% | 100.0% | 不变 |

## 7. 主结论

```text
C. 只适合低频尺度或 drift 约束
```

证据：

1. Depth map 在本片段中逐帧生成稳定，人物 torso/pelvis ROI 可提取。
2. 低频对齐后，P0/P1 的 depth consistency RMSE 分别下降约 35.7% 和 27.1%，temporal jerk mean 分别下降约 21.6% 和 9.8%。
3. A/B 均保持 100% court XY inside ratio；地面一致性整体没有明显破坏，但 P1 的 contact median abs min-vertex Z 有小幅变差。
4. P0/P1 correlation 只有 0.497/0.235，Depth 逐帧变化明显大于 GVHMR root 变化，说明当前模型/ROI/坐标关系不足以支持高频或强约束融合。

因此当前结果支持把 Depth 用作低频尺度或 drift 诊断/修正，不支持立即扩展到完整 450 帧，也不支持把 Depth 直接写入 root 深度。若要进入下一轮，应先修正人物 ROI、时间/坐标/尺度关系，并在更多短片段上复核。

## 8. 限制与下一步

- 本次只验证一个 75 帧窗口，没有完整视频结论。
- 没有独立的真实 metric depth ground truth，因此“深度质量”主要由有效性、视觉结构和 GVHMR 对齐诊断判断。
- ROI 为基于肩/髋关键点的矩形 torso/pelvis 区域，尚未引入人物分割 mask。
- 当前 A/B 指标用于实验筛选，不是最终发表级评测。
- 等待用户验收；在用户明确同意前不跑全视频、不接入 SAM2/VGGT 场景扩展、不重构工程。

## 9. 安全记录

- 本实验只使用本地分支 `feature/depth-constraint-local`。
- 本实验没有执行 GitHub 远程写操作：无 `git push`、无 PR、无远程分支、无 release。
- 原始 GVHMR 成功 NPZ 未覆盖。

## 10. 后续通用实现（取代旧 A/B 脚本作为推荐入口）

短片段实验之后已增加通用生产入口：

```text
work\depth_estimation\run_depth_constrained_3d.py
```

它把本报告的 C 类结论实现为自动低频 drift 约束，并修复了旧 `run_soft_constraint_ab.py` 中两名球员写入同名 `trajectory_bundle.npz`、后者覆盖前者的问题。旧脚本现在改为分别输出 `p0_trajectory_bundle.npz` 和 `p1_trajectory_bundle.npz`；新项目应优先使用通用入口。

最终通用算法只修改球场 XY，不修改 root 或 mesh 的任何 Z 值。75 帧复验中，P0/P1 的短时漂移尖峰分别由 58 降至 29、39 降至 16，Depth RMSE 分别由 0.2835 m 降至 0.0955 m、0.2310 m 降至 0.0556 m；mean jerk 分别下降约 30.0% 和 35.6%，球场内比例保持 100%，地面/跳跃高度逐值不变。

完整输入契约、运行命令、自动失败保护和输出说明见 `docs\depth_constraint_pipeline.md`。
