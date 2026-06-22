import cv2
from rtmlib import Wholebody
import os
import csv # 导入 CSV 库

# 配置路径
video_path = r'E:\badminton\peoplepose\final.MP4'
output_dir = r'E:\badminton\peoplepose\Output'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# CSV 文件路径
csv_file_path = os.path.join(output_dir, 'pose_data.csv')

# 💡 尝试CUDA，不行用CPU
try:
    pose_model = Wholebody(device='cuda')
    print("--- 正在尝试使用 CUDA 加速 ---")
except Exception:
    print("--- 使用 CPU 进行分析（速度较慢） ---")
    pose_model = Wholebody(device='cpu')

# 打开视频
cap = cv2.VideoCapture(video_path)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# 定义视频写入器
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(os.path.join(output_dir, 'analyzed_video.mp4'), fourcc, fps, (width, height))

# 💡 初始化 CSV 写入器
csv_file = open(csv_file_path, mode='w', newline='', encoding='utf-8')
csv_writer = csv.writer(csv_file)
# 写入 CSV 表头
csv_writer.writerow(['frame', 'keypoint_id', 'x', 'y', 'confidence'])

print("--- 开始实时分析并记录数据 (按 'q' 退出) ---")

frame_count = 0
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break
    
    frame_count += 1
    
    # 检测结果
    results = pose_model(frame)
    if isinstance(results, tuple):
        keypoints_list, scores_list = results
    else:
        keypoints_list = results
        scores_list = [None] * len(keypoints_list)

    # 绘制骨架并记录数据
    for person_id, keypoints in enumerate(keypoints_list):
        for kp_id, point in enumerate(keypoints):
            if len(point) == 3:
                x, y, conf = point
            else:
                x, y = point
                conf = 1.0

            # 绘制绿色点
            if conf > 0.3:
                cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 0), -1)
                
                # 💡 记录坐标数据到 CSV
                csv_writer.writerow([frame_count, kp_id, x, y, conf])

    # 显示结果并写入文件
    cv2.imshow('Sports Analysis', frame)
    out.write(frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 💡 关闭文件和资源
cap.release()
out.release()
csv_file.close()
cv2.destroyAllWindows()
print(f"--- 分析完成！ ---")
print(f"视频已保存至: {os.path.join(output_dir, 'analyzed_video.mp4')}")
print(f"坐标数据已保存至: {csv_file_path}")