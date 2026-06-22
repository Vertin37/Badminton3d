import numpy as np
import cv2
import matplotlib.pyplot as plt

import numpy as np
import cv2
import matplotlib.pyplot as plt

def visualize_camera_pose_centered(imagePoints, cameraMatrix, distCoeffs=None,
                                   court_length=13.4, court_width=6.1, net_height=1.524):
    """
    使用PnP算法估计相机位姿并在3D中可视化
    输入的四个图像点顺序：左上、左下、右下、右上
    世界坐标以球场中心为原点。
    """
    
    # ===== 1️⃣ 构建球场四角的世界坐标（以中心为原点） =====
    half_L = court_length / 2
    half_W = court_width / 2

    # 世界坐标顺序对应：左上、左下、右下、右上
    objectPoints = np.array([
        [-half_W, half_L, 0],  # 右上
        [-half_W, - half_L, 0],  # 左上
        [ half_W, - half_L, 0],  # 左下
        [ half_W, half_L, 0],  # 右下
    ], dtype=np.float32)

    # ===== 2️⃣ 运行PnP求解相机姿态 =====
    success, rvec, tvec = cv2.solvePnP(objectPoints, imagePoints, cameraMatrix, distCoeffs)
    if not success:
        raise RuntimeError("solvePnP求解失败，请检查输入数据！")

    R, _ = cv2.Rodrigues(rvec)
    camera_position = -R.T @ tvec
    camera_position = camera_position.flatten()

   

    print("\n==== 相机参数 ====")
    print("旋转向量 rvec:\n", R)  # 可以保留rvec原始值
    print("旋转矩阵 R:\n", R)
    print("平移向量 tvec:\n", tvec)
    print("相机在世界坐标系中的位置:\n", camera_position)

    # 相机朝向（Z轴方向）
    camera_dir = R.T @ np.array([[0], [0], [1]])
    print("相机Z轴方向向量:\n", camera_dir.flatten())


    # ===== 3️⃣ 构造3D场景元素 =====
    # 球场四边形（Z=0）
    court = np.array([
        [-half_W, -half_L, 0],
        [-half_W,  half_L, 0],
        [ half_W,  half_L, 0],
        [ half_W, -half_L, 0],
        [-half_W, -half_L, 0]
    ])

    # 球网（位于Y=0平面）
    net = np.array([
        [-half_W, 0, 0],
        [ half_W, 0, 0],
        [ half_W, 0, net_height],
        [-half_W, 0, net_height],
        [-half_W, 0, 0]
    ])

    # ===== 4️⃣ 可视化 =====
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 球场边框
    ax.plot(court[:, 0], court[:, 1], court[:, 2], color='green', lw=2, label="Court")

    # 球网
    ax.plot(net[:, 0], net[:, 1], net[:, 2], color='blue', lw=2, label="Net")

    # 相机位置
    ax.scatter(*camera_position, color='red', s=80, label='Camera')

    # 绘制相机朝向箭头（Z轴方向）
    camera_dir = R.T @ np.array([[0], [0], [1]])
    camera_dir = camera_dir.flatten()
    ax.quiver(camera_position[0], camera_position[1], camera_position[2],
              camera_dir[0], camera_dir[1], camera_dir[2],
              length=3, color='red', arrow_length_ratio=0.2, label='View direction')

    # ===== 5️⃣ 美化坐标系 =====
    ax.set_title("3D Camera Pose Visualization (Court Centered)", fontsize=14)
    ax.set_xlabel("X (Left-Right, m)")
    ax.set_ylabel("Y (Front-Back, m)")
    ax.set_zlabel("Z (Height, m)")
    ax.legend()

    # 设置坐标比例
    ax.set_box_aspect([court_width, court_length, 3])
    ax.view_init(elev=25, azim=-60)

    # 绘制坐标原点
    ax.scatter(0, 0, 0, color='black', s=40, label='Court Center (Origin)')
    ax.text(0, 0, 0.1, "Origin", color='black')

    plt.show()

    return camera_position, R, tvec



# 2️⃣ 像素坐标（单位：像素）
imagePoints = np.array([
    [1714.18518519,761],#右上
    [472.88405797,761],#左上
    [199.55072464,876],#左下
    [1984.64814815,876]#右下
   
    
], dtype=np.float32)

# 3️⃣ 相机内参
fx, fy = 2656, 2006
cx, cy = 1920, 1080
cameraMatrix = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]], dtype=np.float64)

visualize_camera_pose_centered(imagePoints, cameraMatrix)
