import cv2
import numpy as np
import pandas as pd # 用于记录和导出数据
from collections import deque

class BadmintonDataLogger:
    def __init__(self, ball_color, p1_color, p2_color):
        # 颜色配置 (BGR格式)
        self.colors = {
            'ball': ball_color,
            'player1': p1_color,
            'player2': p2_color
        }
        
        # 数据存储清单
        self.data_records = []
        
        # 轨迹平滑（用于计算速度）
        self.history = {
            'ball': deque(maxlen=5),
            'p1': deque(maxlen=5),
            'p2': deque(maxlen=5)
        }

    def get_centroid(self, frame, target_color):
        """通过颜色提取Mask并计算质心"""
        # 容差范围
        lower = np.array([max(0, c - 10) for c in target_color])
        upper = np.array([min(255, c + 10) for c in target_color])
        
        mask = cv2.inRange(frame, lower, upper)
        moments = cv2.moments(mask)
        
        if moments["m00"] > 0:
            cx = int(moments["m10"] / moments["m00"])
            cy = int(moments["m01"] / moments["m00"])
            return (cx, cy)
        return None

    def log_frame(self, frame_idx, frame):
        # 1. 提取各个对象的坐标
        b_pos = self.get_centroid(frame, self.colors['ball'])
        p1_pos = self.get_centroid(frame, self.colors['player1'])
        p2_pos = self.get_centroid(frame, self.colors['player2'])

        # 2. 计算简单物理量 (如球速)
        ball_speed = 0
        if b_pos and len(self.history['ball']) > 0:
            prev_b = self.history['ball'][-1]
            ball_speed = np.linalg.norm(np.array(b_pos) - np.array(prev_b))

        # 3. 记录数据行
        record = {
            'frame': frame_idx,
            'ball_x': b_pos[0] if b_pos else None,
            'ball_y': b_pos[1] if b_pos else None,
            'p1_x': p1_pos[0] if p1_pos else None,
            'p1_y': p1_pos[1] if p1_pos else None,
            'p2_x': p2_pos[0] if p2_pos else None,
            'p2_y': p2_pos[1] if p2_pos else None,
            'ball_speed': round(ball_speed, 2)
        }
        self.data_records.append(record)

        # 更新历史
        if b_pos: self.history['ball'].append(b_pos)
        
        return b_pos, p1_pos, p2_pos

    def save_to_csv(self, filename):
        df = pd.DataFrame(self.data_records)
        df.to_csv(filename, index=False)
        print(f"数据已保存至 {filename}")

# --- 主程序 ---
def process_sam_video(video_path, output_csv):
    cap = cv2.VideoCapture(video_path)
    
    # 【重要】根据你的SAM输出视频修改这里的颜色 (BGR)
    # 例如：球是白色(255,255,255)，球员1是红色(0,0,255)，球员2是蓝色(255,0,0)
    logger = BadmintonDataLogger(
        ball_color=(255, 255, 255), 
        p1_color=(0, 0, 255), 
        p2_color=(255, 0, 0)
    )

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # 提取并记录数据
        b, p1, p2 = logger.log_frame(frame_idx, frame)

        # 在画面上做可视化（可选）
        if b: cv2.circle(frame, b, 5, (0, 255, 0), -1) # 绿色点标记提取到的球
        if p1: cv2.rectangle(frame, (p1[0]-10, p1[1]-10), (p1[0]+10, p1[1]+10), (0,0,255), 2)
        
        cv2.imshow('SAM Data Extraction', frame)
        frame_idx += 1
        
        if cv2.waitKey(1) & 0xFF == 27: break

    logger.save_to_csv(output_csv)
    cap.release()
    cv2.destroyAllWindows()

# 运行
process_sam_video('uesesam_sam_人+球1.mp4', 'badminton_telemetry.csv')