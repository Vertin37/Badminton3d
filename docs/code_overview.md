# 代码整理说明

本项目代码来自多个研究阶段脚本。整理后的原则是：保留原始脚本名称，按功能归档，先把研究链路讲清楚，再逐步做参数化和工程化。

## 主线 A：球场、分割、轨迹与落点

```text
原始视频 / 单帧图片
  -> calibration_reconstruction/ 进行相机标定、球场线识别、PnP 位姿估计
  -> segmentation_tracking/use_sam3*.py 从 SAM 输出中提取目标区域
  -> segmentation_tracking/sam3数据结构拆分.py 导出球/人坐标 CSV
  -> segmentation_tracking/优化对羽毛球落地判定.py 或 1.23完整版final.py 做轨迹和事件分析
```

推荐先跑这一条链路，因为它最接近“从视频到可解释事件”的最短路径。

## 主线 B：人体姿态、生物力学与复盘报告

```text
原始视频
  -> pose_biomechanics/start_analysis.py 提取人体关键点
  -> Sports2D 外部流程生成 TRC
  -> pose_biomechanics/sports2D.py / sports2D可视化.py 推断并可视化球拍
  -> pose_biomechanics/数据提取.py 生成结构化技术指标
  -> llm_report/api调用qwen.py 生成文字复盘报告
```

这条链路依赖 RTMLib、Sports2D 和大模型 API，复现前需要先准备环境和输入文件。

## 目录与脚本

### `src/calibration_reconstruction/`

| 脚本 | 作用 | 典型输入 | 典型输出 |
|---|---|---|---|
| `相机标定.py` | 棋盘格相机标定 | `calibration_images/*.jpg` | 相机内参、畸变参数 |
| `pnp.py` | 球场角点 PnP 位姿估计 | 球场 2D/3D 对应点 | 相机位姿、3D 可视化 |
| `10.12final.py` | 球场线、球网、3D 场景实验 | 单帧图片 | 辅助可视化图片 |

### `src/segmentation_tracking/`

| 脚本 | 作用 | 典型输入 | 典型输出 |
|---|---|---|---|
| `use_sam3.py` | 生成二值 mask 视频 | 原视频 + SAM 标注视频 | 黑白 mask 视频 |
| `use_sam3_彩色.py` | 保留目标原色的黑底视频 | 原视频 + SAM 标注视频 | 彩色目标提取视频 |
| `sam3数据结构拆分.py` | 从分割视频提取目标坐标 | 分割结果视频 | 坐标/状态 CSV |
| `羽毛球识别.py` | 快速羽毛球检测实验 | 原视频 | 检测框视频 |
| `优化对羽毛球落地判定.py` | 落地/击球规则判断实验 | 目标视频 | 分析视频 |
| `1.23完整版.py` | 集成式轨迹追踪实验 | 目标提取视频 | 轨迹视频、CSV |
| `1.23完整版final.py` | 增强版集成式分析 | 目标提取视频 | 分析视频、CSV |

### `src/pose_biomechanics/`

| 脚本 | 作用 | 典型输入 | 典型输出 |
|---|---|---|---|
| `start_analysis.py` | RTMLib Wholebody 姿态提取 | 原视频 | 关键点 CSV、叠加视频 |
| `sports2D.py` | 基于 TRC 模拟球拍位置 | Sports2D `.trc` | 球拍坐标 CSV |
| `sports2D可视化.py` | 球拍坐标视频叠加 | 视频 + 球拍 CSV | 叠加视频 |
| `数据提取.py` | 提取双人技术/生物力学指标 | 双人 TRC | 结构化分析 CSV |

### `src/llm_report/`

| 脚本 | 作用 | 典型输入 | 典型输出 |
|---|---|---|---|
| `api调用qwen.py` | 调用 Qwen 生成复盘报告 | 结构化分析 CSV | 文本报告 |

## 复现前必须检查

1. 修改脚本中的本地绝对路径，例如 `E:\badminton\...`。
2. 确认输入视频、SAM 输出视频、TRC 文件的帧率和帧数是否一致。
3. 如果在服务器或无界面环境运行，注释掉 `cv2.imshow` / `cv2.waitKey`。
4. 不要把 API Key 写入代码；建议用环境变量读取。
5. 大视频、中间 CSV、模型权重不要提交到 Git，统一放在 `examples/data/` 或 `outputs/` 的本地副本中。

## 后续工程化建议

- 把脚本中的输入输出路径改成 `argparse` 参数。
- 将通用的视频读写、轨迹滤波、CSV 导出逻辑抽到公共模块。
- 增加一个 `configs/` 目录保存示例配置。
- 准备 5-10 秒的小样例视频，便于做 smoke test。
- 给关键 pipeline 增加最小运行命令和预期输出截图。
