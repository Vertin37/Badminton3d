import cv2
import numpy as np
import pandas as pd
import os

class BadmintonProTracker:
    def __init__(self, fps, width, height):
        self.fps = fps
        self.width, self.height = width, height
        self.court_points = []  # 存储球场标定点像素坐标
        self.role_colors = {"p_far": None, "p_near": None, "ball": None}
        self.data_records = []

    def click_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            param['points'].append((x, y))
            color_text = param.get('text', 'Point')
            # 视觉反馈
            cv2.circle(param['img'], (x, y), 5, (0, 255, 255), -1)
            cv2.putText(param['img'], f"{color_text}{len(param['points'])}:({x},{y})", (x+10, y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.imshow(param['win'], param['img'])

    def setup(self, original_path, extracted_path):
        # 1. 标定球场像素坐标
        cap_orig = cv2.VideoCapture(original_path)
        ret, frame_orig = cap_orig.read()
        cap_orig.release()
        if ret:
            print("\n步骤1：请在球场上点击标定点（通常6个），完成后按‘空格’或‘ESC’退出窗口。")
            data = {'img': frame_orig.copy(), 'points': [], 'win': 'Step1: Court Calibration'}
            cv2.namedWindow(data['win'])
            cv2.setMouseCallback(data['win'], self.click_callback, data)
            cv2.imshow(data['win'], data['img'])
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            self.court_points = data['points']
            print(f"已记录球场标定点: {self.court_points}")

        # 2. 引导角色颜色
        cap_ext = cv2.VideoCapture(extracted_path)
        ret, frame_ext = cap_ext.read()
        cap_ext.release()
        if ret:
            print("\n步骤2：在提取视频中依次点击：1.远端人色块 2.近端人色块 3.球色块")
            data = {'img': frame_ext.copy(), 'points': [], 'win': 'Step2: Role Color Picker', 'text': 'Role'}
            cv2.namedWindow(data['win'])
            cv2.setMouseCallback(data['win'], self.click_callback, data)
            cv2.imshow(data['win'], data['img'])
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
            hsv_frame = cv2.cvtColor(frame_ext, cv2.COLOR_BGR2HSV)
            roles = ["p_far", "p_near", "ball"]
            for i, pt in enumerate(data['points'][:3]):
                self.role_colors[roles[i]] = hsv_frame[pt[1], pt[0]].tolist()

    def get_pos(self, frame, target_hsv, is_ball=False):
        if target_hsv is None: return None
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # 锁定色调，宽容亮度和饱和度
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
        # 球取质心，人取脚底
        return (int(x + w/2), int(y + h/2)) if is_ball else (int(x + w/2), int(y + h))

    def run(self, extracted_path, output_video, output_csv):
        cap = cv2.VideoCapture(extracted_path)
        
        # 使用 mp4v 配合 .mp4 后缀，或者 avc1 (需系统安装对应解码器)
        # 这里优先使用兼容性最强的 mp4v
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
        out_vw = cv2.VideoWriter(output_video, fourcc, self.fps, (self.width, self.height))
        
        print("正在分析并导出 MP4 视频...")
        frame_idx = 0
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break

                f_px = self.get_pos(frame, self.role_colors["p_far"])
                n_px = self.get_pos(frame, self.role_colors["p_near"])
                b_px = self.get_pos(frame, self.role_colors["ball"], is_ball=True)

                self.data_records.append({
                    "frame": frame_idx,
                    "ball_x": b_px[0] if b_px else None, "ball_y": b_px[1] if b_px else None,
                    "p_far_x": f_px[0] if f_px else None, "p_far_y": f_px[1] if f_px else None,
                    "p_near_x": n_px[0] if n_px else None, "p_near_y": n_px[1] if n_px else None
                })

                # 视觉增强
                if f_px: cv2.circle(frame, f_px, 8, (0, 0, 255), -1)
                if n_px: cv2.circle(frame, n_px, 8, (255, 0, 0), -1)
                if b_px:
                    cv2.circle(frame, b_px, 15, (0, 255, 0), 2)  # 绿色圆环
                    cv2.drawMarker(frame, b_px, (255, 255, 255), cv2.MARKER_CROSS, 25, 2) # 醒目十字

                out_vw.write(frame)
                cv2.imshow("Tracking", frame)
                if cv2.waitKey(1) & 0xFF == 27: break
                frame_idx += 1
        finally:
            cap.release()
            out_vw.release()
            cv2.destroyAllWindows()

        # --- 保存数据到 CSV ---
        df = pd.DataFrame(self.data_records)
        df.to_csv(output_csv, index=False)
        
        # 将球场像素坐标单独存为一个 txt 或在 print 输出，方便你查看
        with open("court_pixels.txt", "w") as f:
            f.write("Court Calibration Pixel Coordinates (X, Y):\n")
            for i, pt in enumerate(self.court_points):
                f.write(f"Point {i+1}: {pt}\n")
        
        print(f"\n任务结束！")
        print(f"1. 追踪视频: {output_video}")
        print(f"2. 追踪数据: {output_csv}")
        print(f"3. 场地坐标: court_pixels.txt")

if __name__ == "__main__":
    VIDEO_FILE = "color_extracted_result.mp4"
    cap = cv2.VideoCapture(VIDEO_FILE)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    tracker = BadmintonProTracker(fps, w, h)
    # 输入原视频标定，提取视频分析
    tracker.setup("12月28日 (1)(1).mp4", VIDEO_FILE)
    tracker.run(VIDEO_FILE, "tracking_output.mp4", "tracking_data.csv")