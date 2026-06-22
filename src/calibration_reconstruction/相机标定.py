import cv2
import numpy as np
import glob
import os

# 1. 设置棋盘格参数
chessboard_size = (9, 6)  # 内角点数：9列 x 6行
square_size = 25.0        # 单格边长（单位：mm）

# 2. 准备世界坐标（Z=0）
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
objp *= square_size

# 3. 存储所有图像的世界点和图像点
objpoints = []  # 3D 点
imgpoints = []  # 2D 点

# 4. 读取标定图像（支持 JPG/PNG）
images = glob.glob('calibration_images/*.jpg')  # 替换为你的图像路径

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 查找棋盘格角点
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

    if ret:
        objpoints.append(objp)
        # 亚像素精确化
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)

        # 可视化（可选）
        cv2.drawChessboardCorners(img, chessboard_size, corners2, ret)
        cv2.imshow('Calibration', img)
        cv2.waitKey(500)

cv2.destroyAllWindows()

# 5. 执行标定
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None
)

# 6. 输出结果
print("Camera matrix (内参):\n", mtx)
print("Distortion coefficients (畸变系数):\n", dist.ravel())

# 7. 计算重投影误差（越小越好，理想 < 0.5 像素）
mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
    mean_error += error
print("Mean reprojection error:", mean_error / len(objpoints))