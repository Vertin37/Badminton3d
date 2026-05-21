
# BadmintonPose - 核心源码目录说明

本目录包含了 **BadmintonPose** 系统的核心算法实现模块，涵盖球场三维几何标定、递进式目标分割、时序滤波以及物理运动建模等核心研究过程。

> ⚠️ **外部框架配置提示：** 本仓库侧重于业务逻辑与核心管线（Pipeline）的实现。前沿骨干网络 **SAM3** 与 **VGGT** 涉及复杂的底层依赖与预训练权重加载，需要结合其官方开源仓库进行配置，具体请参见主目录 [README.md](https://github.com/Pomegranate-fried-wine/BadmintonPose/blob/main/README.md) 的技术路线引用。
> 
> 

---

## 📂 目录结构

```text
code/
├── calibration/
│   ├── camera_calibrator.m       # 基于 MATLAB 的相机内参标定辅助脚本
│   └── hough_court_detect.py     # 基于霍夫变换的羽毛球场线自动检测与空间映射
├── perception/
│   ├── dynamic_roi_filter.py     # 动态感兴趣区域（ROI）裁剪算法（用于加速 SAM3 推理）
│   └── track_players_ball.py     # 球员与羽毛球的多目标时序追踪管线
├── filtering/
│   ├── kalman_ball_filter.py     # 用于羽毛球三维轨迹平滑与预测的卡尔曼滤波器
│   └── aerodynamics_model.py     # 空气动力学物理约束模型（提取击球/落地关键帧）
└── ui/
    └── run_yolo_ui.py            # 预配置的 YOLO 检测与可视化交互 UI 系统

```

---

## 🛠 核心模块与研究源码概述

### 1. 三维场景重建与几何标定 (`/calibration`)

在进行视频流的 3D 空间映射前，需要消除单目相机的镜头畸变并确立基准坐标系。

* **`camera_calibrator.m`**：利用 MATLAB Camera Calibrator 工作流编写的标定脚本。导入标准棋盘格照片即可全自动计算相机内参矩阵（Camera Matrix）。


* **`hough_court_detect.py`**：利用霍夫变换（Hough Transform）对输入视频的首帧进行场地线提取。结合标准场地尺寸，计算出 2D 像素平面到 3D 世界坐标系的单应性矩阵（Homography）。



### 2. 多模态感知与分割 (`/perception`)

本模块负责对复杂的运动场景进行细粒度像素级分割。

* **`track_players_ball.py`**：承接 YOLO 的检测框输入，将球员和羽毛球的时序轨迹进行初步关联。


* **`dynamic_roi_filter.py`**：由于 **SAM3 (Segment Anything Model 3)** 模型体量大、全图直接推理耗时较长，我们设计了该动态 ROI 过滤脚本。它会根据上一帧的运动矢量预测下一帧目标的大致区域，仅将裁剪后的局部区域（ROI）作为 Prompt 送入 SAM3，从而在不损失精细度的前提下大幅提高系统推理速度。


* 关于 SAM3 与 VGGT 的环境编译与预训练模型部署，请务必严格参考 [SAM3 官方 GitHub](https://www.google.com/search?q=https://github.com/facebookresearch/sam3) 与 [VGGT 官方 GitHub](https://www.google.com/search?q=https://github.com/facebookresearch/vggt) 进行本地克隆与配置。



### 3. YOLO 预配置 UI 系统 (`/ui`)

鉴于在羽毛球垂直场景下，从零开始自建数据集、标注成百上千张高动态模糊图片并进行 YOLO 训练的工程量极其繁琐且耗时：

* **`run_yolo_ui.py`**：为了降低使用门槛，本项目集成并调用了一个**配置好的端到端 UI 系统**。该界面已封装好针对羽毛球及球员优化过的预训练权重。您无需修改底层代码，即可通过图形化界面直接导入赛事视频、交互式调节置信度阈值（Confidence Threshold）、一键导出检测框数据，大大提升了研究效率。



### 4. 时序滤波与空气动力学分析 (`/filtering`)

* **`kalman_ball_filter.py`**：针对羽毛球体积小、飞行速度快且在高速挥拍时极易产生运动模糊或被身体遮挡的特点，实现了一个专门优化的卡尔曼滤波器。通过建立状态转移矩阵，对丢失的帧进行轨迹预测与物理平滑补全。


* **`aerodynamics_model.py`**：由于羽毛球独特的裙摆结构使其具有极高的空气阻力系数。该脚本建立了符合非线性空气动力学的物理方程，通过捕捉轨迹中的速度突变点和加速度转折点，精准提取出击球瞬间（Stroke Apex）**与**球落地（Landing）的关键时序帧。



---

## 🚀 快速上手说明

1. **环境准备**：
请确保您已按照主目录 [README.md](https://www.google.com/search?q=../README.md) 的说明配置好基础的 Python 或 MATLAB 环境。


2. **生成相机参数**：
运行 `calibration/camera_calibrator.m`，并将导出的内参矩阵保存至指定配置文件中。


3. **启动 UI 检测系统**：
无需繁琐地训练数据集，直接运行预配置的 UI 界面进行目标检测提取：


```bash
python ui/run_yolo_ui.py

```


4. **轨迹物理平滑**：
将 UI 系统导出的原始检测坐标传入滤波管线，即可生成全自动 3D 还原轨迹：


```bash
python filtering/kalman_ball_filter.py --input ../data/raw_bbox.csv

```



```

```
