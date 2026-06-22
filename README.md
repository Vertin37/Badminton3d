# BadmintonPose

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
