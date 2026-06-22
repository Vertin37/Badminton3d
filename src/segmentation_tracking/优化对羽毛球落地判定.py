
import cv2
import numpy as np
import math
from collections import deque

class BadmintonAnalyzer:
    def __init__(self):
        # --- 核心参数调整区 ---
        self.min_area = 12      # 噪点过滤：太小的白色块忽略
        self.stop_frames = 5     # 连续多少帧静止才算落地
        self.stop_dist_th = 3.0   # 静止判定阈值：如果两帧移动距离小于2像素，视作静止
        self.hit_acc_th = 40    # 击球判定阈值：加速度（速度变化量）超过此值
        
        # 轨迹缓存
        self.history = deque(maxlen=30) # 记录最近30帧坐标
        self.state = "Flying"           # 当前状态: Flying, Landed

    def get_ball_position(self, frame):
        if len(frame.shape) > 2:
           gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
           gray = frame

        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return None

        last_pos = None
        if len(self.history) > 0:
            last_pos = np.array(self.history[-1])

        best_candidate = None
        best_area = 0

        for c in contours:
            area = cv2.contourArea(c)
            if area < self.min_area:
                continue

            M = cv2.moments(c)
            if M["m00"] == 0:
               continue

            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            curr_pos = np.array([cx, cy])

          # ===== 核心新增：瞬移约束 =====
            if last_pos is not None:
              dist = np.linalg.norm(curr_pos - last_pos)
              if dist > 50:   # <<< 瞬移阈值（可调）
                   continue

        # 在满足“不瞬移”的前提下，再选面积最大的
            if area > best_area:
                best_area = area
            best_candidate = (cx, cy)

        return best_candidate

    def analyze_motion(self, current_pos):
        """
        纯轨迹物理分析逻辑
        """
        if current_pos is None:
            return None

        self.history.append(current_pos)
        
        # 数据不足，无法计算
        if len(self.history) < 5:
            return None

        # --- 1. 计算基本物理量 ---
        # 提取最近几帧
        p_now = np.array(self.history[-1])
        p_prev = np.array(self.history[-2])
        p_prev2 = np.array(self.history[-3])

        # 瞬时速度向量
        v1 = p_now - p_prev     # 当前速度
        v2 = p_prev - p_prev2   # 上一帧速度
        
        # 速度模长（标量速度）
        speed_now = np.linalg.norm(v1)
        speed_prev = np.linalg.norm(v2)

        # --- 2. 落地判定逻辑 (Land Logic) ---
        # 逻辑：连续 N 帧，球的位置变化极小
        if len(self.history) >= self.stop_frames:
            # 取最近 N 帧的位移标准差，或者简单的最大距离
            recent_points = np.array(list(self.history)[-self.stop_frames:])
            # 计算这些点的离散程度
            movement_range = np.max(recent_points, axis=0) - np.min(recent_points, axis=0)
            max_drift = np.linalg.norm(movement_range)

            if max_drift < self.stop_dist_th * self.stop_frames:
                # 还可以加一个辅助判断：球通常在画面下方落地（y坐标较大）
                # if p_now[1] > frame_height * 0.5:
                if self.state != "Landed":
                    self.state = "Landed"
                    return "EVENT_LAND" # 触发落地事件
                return "STATUS_LANDED"

        # --- 3. 击球判定逻辑 (Hit Logic) ---
        # 逻辑：速度方向突变 + 速度爆发
        # 计算两个速度向量的夹角余弦
        if speed_now > 0 and speed_prev > 0:
            cos_angle = np.dot(v1, v2) / (speed_now * speed_prev)
            # 限制在 [-1, 1] 避免浮点误差
            cos_angle = np.clip(cos_angle, -1.0, 1.0) 
            angle = np.degrees(np.arccos(cos_angle))
            
            # 判定条件：
            # A. 轨迹折角大（大于45度，通常击球会反向或变向）
            # B. 且 当前速度不低（不是轻轻碰了一下）
            # C. 或者 速度模长突然激增（杀球）
            
            is_turn = angle > 70 
            is_acceleration = speed_now > speed_prev * 10 # 速度瞬间变为1.5倍
            
            if (is_turn and speed_now > 5.0) or (is_acceleration and speed_now > 10.0):
                if self.state != "Landed": # 只有球在飞的时候才能被击打
                    return "EVENT_HIT"
        
        self.state = "Flying"
        return None
def init_video_writer(cap, save_path, fps=30):
    """
    初始化视频写入器
    """
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 通用 mp4 编码
    writer = cv2.VideoWriter(save_path, fourcc, fps, (width, height))
    return writer

# --- 主程序：如何使用 ---
def run_detection(video_path, save_path=None):
    cap = cv2.VideoCapture(video_path)
    analyzer = BadmintonAnalyzer()

    # ===== 新增：初始化视频保存 =====
    writer = None
    if save_path is not None:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30
        writer = init_video_writer(cap, save_path, fps)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 1. 获取坐标
        pos = analyzer.get_ball_position(frame)
        
        # 2. 绘制轨迹
        if pos:
            cv2.circle(frame, pos, 5, (0, 0, 255), -1)
        
        # 3. 分析运动状态
        event = analyzer.analyze_motion(pos)

        # 4. 显示事件
        if event == "EVENT_HIT":
            cv2.putText(frame, "HIT !!!", (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        elif event == "EVENT_LAND":
            cv2.putText(frame, "LANDED", (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

        # ===== 新增：写入视频 =====
        if writer is not None:
            writer.write(frame)

        cv2.imshow('Track Analysis', frame)
        if cv2.waitKey(30) & 0xFF == 27:
            break

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()

# 将你的视频路径填在这里
run_detection(
    video_path='uesesam_video1.mp4',
    save_path='analysis_uesesam_video1.mp4'
)
