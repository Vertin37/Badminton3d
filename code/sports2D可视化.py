import cv2
import pandas as pd
import numpy as np
import os

# --- 1. 配置路径 ---
# 你的原始视频
video_path = r'E:\badminton\peoplepose\final.MP4'
# 你刚生成的球拍模拟数据 CSV
racket_csv_path = r'E:\badminton\peoplepose\Output\racket_simulated.csv'
# 结果保存路径
output_video_path = r'E:\badminton\peoplepose\Output\final_with_racket.mp4'

# --- 2. 加载数据与视频 ---
if not os.path.exists(racket_csv_path) or not os.path.exists(video_path):
    print("错误：找不到视频或球拍 CSV 文件，请检查路径。")
    exit()

# 读取球拍 CSV (跳过 Frame 列，直接读取 Wrist/RacketHead 的 X/Y)
# 我们假设 CSV 列名是: Frame,Wrist_X,Wrist_Y,RacketHead_X,RacketHead_Y
racket_df = pd.read_csv(racket_csv_path)

# 打开原始视频获取信息
cap = cv2.VideoCapture(video_path)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# 定义视频写入器
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

print(f"--- 开始实时绘图 (视频总长: {total_frames} 帧) ---")

# --- 3. 逐帧绘图循环 ---
frame_count = 0
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break
    
    # 获取当前帧对应的球拍坐标数据
    # 注意：CSV 索引是从 0 开始的
    if frame_count < len(racket_df):
        try:
            row = racket_df.iloc[frame_count]
            
            # 提取坐标（强制转为 int，OpenCV 绘图需要）
            w_x, w_y = int(row['Wrist_X']), int(row['Wrist_Y'])
            r_x, r_y = int(row['RacketHead_X']), int(row['RacketHead_Y'])
            
            # --- 核心可视化逻辑 ---
            # 1. 在手腕位置画一个蓝色的圆点 (半径 6, 蓝色)
            cv2.circle(frame, (w_x, w_y), 6, (255, 0, 0), -1)
            
            # 2. 画出球拍主体：从手腕连线到模拟拍头 (厚度 3, 红色)
            # 这根线就是我们外推出来的“模拟球拍”
            cv2.line(frame, (w_x, w_y), (r_x, r_y), (0, 0, 255), 3)
            
            # 3. 在模拟拍头位置画一个小的空心圆 (半径 10, 白色)
            # 模拟羽毛球拍面中心位置
            cv2.circle(frame, (r_x, r_y), 10, (255, 255, 255), 2)
            
            # 4. 可选：在左上角显示帧号和“模拟”标识
            text = f"Frame: {frame_count} | Racket: Simulated"
            cv2.putText(frame, text, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        except (ValueError, KeyError):
            # 处理 NaN 值或缺失数据帧
            pass

    # --- 4. 显示与保存 ---
    # 实时显示（可按 'q' 键手动退出）
    cv2.imshow('Racket Kinematics Simulation', frame)
    out.write(frame)
    
    frame_count += 1
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- 5. 清理资源 ---
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"--- 绘图完成！ ---")
print(f"带球拍轨迹的视频已保存至: {output_video_path}")