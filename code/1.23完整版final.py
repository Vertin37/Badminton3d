import cv2
import numpy as np
import pandas as pd
from collections import deque
import os

class BadmintonFinalIntegrator:
    def __init__(self, fps, width, height):
        self.fps = fps
        self.width, self.height = width, height
        self.court_points = []
        self.role_colors = {"p_far": None, "p_near": None, "ball": None}
        self.data_records = []
        
        # --- 运动分析逻辑参数 (保持原版参数不变) ---
        self.stop_frames = 5     
        self.stop_dist_th = 3.0   
        self.history = deque(maxlen=30) 
        self.state = "Flying"           

    def click_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            param['points'].append((x, y))
            color_text = param.get('text', 'Point')
            cv2.circle(param['img'], (x, y), 5, (0, 255, 255), -1)
            cv2.putText(param['img'], f"{color_text}{len(param['points'])}:({x},{y})", (x+10, y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.imshow(param['win'], param['img'])

    def setup(self, original_path, extracted_path):
        # 1. 场地标定
        cap_orig = cv2.VideoCapture(original_path)
        ret, frame_orig = cap_orig.read()
        cap_orig.release()
        if ret:
            print("\n步骤1：点击球场标定点，完成后按任意键。")
            data = {'img': frame_orig.copy(), 'points': [], 'win': 'Step1: Court Calibration'}
            cv2.imshow(data['win'], data['img'])
            cv2.setMouseCallback(data['win'], self.click_callback, data)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            self.court_points = data['points']

        # 2. 颜色吸取
        frame_number = 35  # ←←←【在这里修改帧号，例如改成 120 表示第120帧（从0开始）】

        cap_ext = cv2.VideoCapture(extracted_path)
        cap_ext.set(cv2.CAP_PROP_POS_FRAMES, frame_number)  # 跳转到指定帧
        ret, frame_ext = cap_ext.read()
        cap_ext.release()
        if ret:
            print(f"\n步骤2：在第 {frame_number} 帧上依次点击：1.远端人 2.近端人 3.球")
            data = {'img': frame_ext.copy(), 'points': [], 'win': 'Step2: Role Color Picker', 'text': 'Role'}
            cv2.imshow(data['win'], data['img'])
            cv2.setMouseCallback(data['win'], self.click_callback, data)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            hsv_frame = cv2.cvtColor(frame_ext, cv2.COLOR_BGR2HSV)
            roles = ["p_far", "p_near", "ball"]
            for i, pt in enumerate(data['points'][:3]):
                self.role_colors[roles[i]] = hsv_frame[pt[1], pt[0]].tolist()
        else:
            print(f"错误：无法读取第 {frame_number} 帧，请检查帧号是否超出范围。")

    def get_pos_from_color(self, frame, target_hsv, is_ball=False):
        """核心：通过1.23版调试好的HSV逻辑锁定坐标"""
        if target_hsv is None: return None
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h_tol = 10 if is_ball else 15
        lower = np.array([max(0, target_hsv[0]-h_tol), 50, 50])
        upper = np.array([min(179, target_hsv[0]+h_tol), 255, 255])
        mask = cv2.inRange(hsv_frame, lower, upper)
        kernel = np.ones((3,3) if is_ball else (15,15), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts: return None
        best = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(best) < (2 if is_ball else 60): return None
        x, y, w, h = cv2.boundingRect(best)
        return (int(x + w/2), int(y + h/2)) if is_ball else (int(x + w/2), int(y + h))

    def analyze_motion(self, b_px):
        """直接使用追踪到的十字中心b_px进行逻辑判定"""
        if b_px is None: return None
        self.history.append(b_px)
        if len(self.history) < 5: return None

        p_now, p_prev, p_prev2 = np.array(self.history[-1]), np.array(self.history[-2]), np.array(self.history[-3])
        v1, v2 = p_now - p_prev, p_prev - p_prev2
        speed_now, speed_prev = np.linalg.norm(v1), np.linalg.norm(v2)

        # 1. 落地判定
        if len(self.history) >= self.stop_frames:
            recent_points = np.array(list(self.history)[-self.stop_frames:])
            movement_range = np.max(recent_points, axis=0) - np.min(recent_points, axis=0)
            max_drift = np.linalg.norm(movement_range)
            if max_drift < self.stop_dist_th * self.stop_frames:
                if self.state != "Landed":
                    self.state = "Landed"
                    return "EVENT_LAND"
                return "STATUS_LANDED"

        # 2. 击球判定
        if speed_now > 0 and speed_prev > 0:
            cos_angle = np.clip(np.dot(v1, v2) / (speed_now * speed_prev), -1.0, 1.0)
            angle = np.degrees(np.arccos(cos_angle))
            is_turn = angle > 100 
            is_acceleration = speed_now > speed_prev * 3.5
            if (is_turn and speed_now > 15.0) or (is_acceleration and speed_now > 20.0):
                if self.state != "Landed": 
                    self.state = "Flying" # 击球后状态切回飞翔
                    return "EVENT_HIT"
        
        self.state = "Flying"
        return None

    def draw_visuals(self, frame, b_px, f_px, n_px, event):
        # 轨迹线绘制
        #points = list(self.history)
        #for i in range(1, len(points)):
         #   thickness = int(np.sqrt(len(points) / float(i + 1)) * 2.5)
         #   cv2.line(frame, points[i-1], points[i], (255, 255, 0), thickness)
        
        # 角色标记
        if f_px: cv2.circle(frame, f_px, 8, (0, 0, 255), -1)
        if n_px: cv2.circle(frame, n_px, 8, (255, 0, 0), -1)
        if b_px:
            cv2.circle(frame, b_px, 15, (0, 255, 0), 2)  # 绿色圆环
            cv2.drawMarker(frame, b_px, (255, 255, 255), cv2.MARKER_CROSS, 25, 2) # 十字
        
        # 事件大字提醒
        if event == "EVENT_HIT":
            cv2.putText(frame, "HIT !!!", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        elif event == "EVENT_LAND":
            cv2.putText(frame, "LANDED", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

    def run(self, extracted_path, output_video, output_csv):
        cap = cv2.VideoCapture(extracted_path)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
        out_vw = cv2.VideoWriter(output_video, fourcc, self.fps, (self.width, self.height))
        
        print("正在进行深度分析并导出...")
        frame_idx = 0
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break

                # 获取坐标
                f_px = self.get_pos_from_color(frame, self.role_colors["p_far"])
                n_px = self.get_pos_from_color(frame, self.role_colors["p_near"])
                b_px = self.get_pos_from_color(frame, self.role_colors["ball"], is_ball=True)

                # 运动分析 (传入b_px)
                event = self.analyze_motion(b_px)

                # 数据记录
                self.data_records.append({
                    "frame": frame_idx,
                    "ball_x": b_px[0] if b_px else None, "ball_y": b_px[1] if b_px else None,
                    "p_far_x": f_px[0] if f_px else None, "p_far_y": f_px[1] if f_px else None,
                    "p_near_x": n_px[0] if n_px else None, "p_near_y": n_px[1] if n_px else None,
                    "event": event if event in ["EVENT_HIT", "EVENT_LAND"] else ""
                })

                # 绘制视觉效果
                self.draw_visuals(frame, b_px, f_px, n_px, event)

                out_vw.write(frame)
                cv2.imshow("Badminton AI Pro Analytics", frame)
                if cv2.waitKey(1) & 0xFF == 27: break
                frame_idx += 1
        finally:
            cap.release()
            out_vw.release()
            cv2.destroyAllWindows()

        # 存盘
        pd.DataFrame(self.data_records).to_csv(output_csv, index=False)
        with open("court_pixels.txt", "w") as f:
            f.write(f"Court Points: {self.court_points}")
        print(f"处理完成。视频：{output_video}, 数据：{output_csv}")

if __name__ == "__main__":
    VIDEO_IN = "球拍+球优化_color_extracted_result.mp4"
    cap = cv2.VideoCapture(VIDEO_IN)
    fps, w, h = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    tracker = BadmintonFinalIntegrator(fps, w, h)
    tracker.setup("球拍+球优化.mp4", VIDEO_IN)
    tracker.run(VIDEO_IN, "球拍+球优化0_final_analysis_output.mp4", "球拍+球优化0_final_analytics.csv")