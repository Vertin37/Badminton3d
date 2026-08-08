<<<<<<< HEAD
# Badminton3d

基于单目视频的羽毛球赛事视觉分析项目，覆盖球场几何标定、SAM 分割后处理、羽毛球轨迹/落点判断、人体姿态提取、生物力学特征整理和 Qwen 复盘报告生成。

本仓库主要整理研究阶段代码。当前脚本仍保留了部分本地路径和实验文件名，使用前需要按自己的视频、CSV、TRC 路径做少量配置。

## 项目结构

```text
BadmintonPose/
├── src/
│   ├── calibration_reconstruction/  # 相机标定、PnP、球场线/网几何重建
│   ├── segmentation_tracking/       # SAM 后处理、目标提取、羽毛球追踪与落点判断
│   ├── pose_biomechanics/           # RTMLib/Sports2D 姿态提取、球拍估计、生物力学特征
│   └── llm_report/                  # 调用 Qwen 生成技术复盘报告
├── docs/
│   └── code_overview.md             # 各脚本用途与推荐 pipeline
├── examples/
│   └── data/                        # 小型示例数据占位，不提交大视频
├── outputs/                         # 本地输出目录，占位保留，结果文件不提交
├── requirements.txt                 # Python 依赖参考
└── 8-基于单目视觉的羽毛球赛事三维场景重建与球员动作分析系统.pptx
```

## 推荐流程

### 1. 球场标定与几何重建

对应目录：`src/calibration_reconstruction/`

- `相机标定.py`：基于棋盘格图片估计相机内参和畸变参数。
- `pnp.py`：根据球场关键点和相机参数估计相机位姿。
- `10.12final.py`：球场线检测、球网/球场辅助可视化和 3D 展示实验脚本。

### 2. SAM 分割与羽毛球轨迹分析

对应目录：`src/segmentation_tracking/`

- `use_sam3.py`：从原视频和 SAM 输出视频生成二值 mask 视频。
- `use_sam3_彩色.py`：生成黑底彩色目标提取视频。
- `sam3数据结构拆分.py`：从分割视频中抽取目标坐标并保存 CSV。
- `羽毛球识别.py`：基于颜色/帧差的羽毛球快速检测实验。
- `优化对羽毛球落地判定.py`、`1.23完整版.py`、`1.23完整版final.py`：轨迹追踪、状态判断、击球/落点事件分析。

### 3. 姿态、生物力学与球拍推断

对应目录：`src/pose_biomechanics/`

- `start_analysis.py`：调用 RTMLib Wholebody 从视频提取人体关键点。
- `sports2D.py`：读取 Sports2D TRC 数据并模拟球拍位置。
- `sports2D可视化.py`：把球拍估计结果叠加回视频。
- `数据提取.py`：从双人 TRC 中提取步态、关节角度等结构化特征。

### 4. 大模型复盘报告

对应目录：`src/llm_report/`

- `api调用qwen.py`：读取结构化 CSV，调用阿里云百炼兼容 OpenAI SDK 的 Qwen 模型生成复盘文本。

运行前请把 API Key 放到环境变量或本地配置中，不要直接提交到 Git。

## 环境准备

建议使用 Python 3.9+。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

部分能力依赖外部工具或模型：

- SAM3 / SAM 输出视频：本仓库不包含模型权重和大视频。
- Sports2D：可用外部 Sports2D 工具先生成 `.trc` 文件。
- RTMLib：用于实时人体关键点检测。
- Qwen / DashScope：用于文本复盘报告生成。

## 数据与输出约定

仓库不提交大体积视频、模型权重、缓存和中间输出。建议本地使用：

```text
examples/data/   # 放少量可公开示例输入
outputs/         # 放分析结果、导出视频和 CSV
```

脚本里仍有一些研究阶段的硬编码路径，例如 `E:\badminton\...`、`final.mp4`、`*.trc`。复现时请先根据自己的目录改成对应输入输出路径。

## 文档

更细的脚本说明和最短复现链路见 [docs/code_overview.md](docs/code_overview.md)。

## 双人持续 ID

`run_pose_test.py` 现在提供跨帧双人身份保持。跟踪器使用肩、髋、膝、踝等身体关键点的加权中心、人体尺度、速度预测和归一化姿态形状进行匹配，不使用简单的左右位置作为身份。

先用已有姿态 CSV 做离线验证，不会加载 RTMLib 或使用 GPU：

```powershell
python run_pose_test.py --from-csv outputs/pose_test/pose_data.csv `
    --video outputs/pose_test/analyzed_video.mp4
```

默认生成：

- `outputs/pose_test/pose_data_tracked.csv`：固定 `player_id`（同时保留兼容用的 `person_id` 列）。
- `outputs/pose_test/tracking_debug.csv`：每帧匹配、漏检、候选数量和源检测编号。
- `outputs/pose_test/tracking_debug.mp4`：若当前 Python 环境有 OpenCV，则生成带 `P0/P1` 标签的视频。

实际运行姿态推理时使用视频模式；`--device auto` 可能选择 CUDA，确认 GPU 环境和结果后再运行：

```powershell
python run_pose_test.py --device cpu --no-display
```

## 时序滤波

双人固定 ID 之后，可用 One Euro Filter 对每个 `player_id + keypoint_id` 独立平滑。该路径只读取 CSV，不导入 RTMLib、ONNX Runtime，也不会使用 GPU：

```powershell
python run_pose_test.py `
    --filter-from-csv outputs/pose_test/pose_data_tracked.csv `
    --output-dir outputs/pose_test `
    --filter-fps 25
```

默认生成：

- `outputs/pose_test/pose_data_stable.csv`：`x/y` 为滤波后坐标，`raw_x/raw_y` 保留原始坐标，同时包含 `filter_status`、异常拒绝和滤波覆盖标记。
- `outputs/pose_test/temporal_filter_stats.json`：处理帧数、低置信度点、异常点、插值/保持数量以及每个 player 的覆盖率。

躯干/下肢、普通关键点和手腕/手部使用不同的 One Euro 响应参数；短缺失只做有限插值或保持，长缺失保留为空。视频推理流程在生成固定 ID CSV 后也会自动写出这两个新文件。

将滤波后骨架叠加回原视频进行目视检查：

```powershell
python render_stable_pose_video.py `
    --video examples/data/test.mp4 `
    --csv outputs/pose_test/pose_data_stable.csv `
    --output outputs/pose_test/analyzed_video_stable.mp4
```

生成的视频中，彩色骨架是滤波后坐标，浅灰色骨架是原始坐标；加上 `--filtered-only` 可隐藏原始对照。
