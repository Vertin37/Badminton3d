import cv2
import numpy as np

video_path = 'final.mp4'
cap = cv2.VideoCapture(video_path)

ret, prev = cap.read()
if not ret:
    raise FileNotFoundError("无法读取视频！")
prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

# 获取视频参数
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 创建视频写入对象
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 或 'XVID'
out = cv2.VideoWriter('final_detected_red_filtered.mp4', fourcc, fps, (width, height))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray, prev_gray)
    _, thresh = cv2.threshold(diff,75, 255, cv2.THRESH_BINARY)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 先保存当前帧所有候选框
    candidate_boxes = []
    for cnt in contours:
        if 2 < cv2.contourArea(cnt) < 200:  # 羽毛球较小
            x, y, w, h = cv2.boundingRect(cnt)
            candidate_boxes.append((x, y, w, h))

    # 筛选：如果某个框与其他框非常近，就忽略（认为是衣服干扰）
    final_boxes = []
    for i, (x1, y1, w1, h1) in enumerate(candidate_boxes):
        too_close = False
        cx1, cy1 = x1 + w1 // 2, y1 + h1 // 2
        for j, (x2, y2, w2, h2) in enumerate(candidate_boxes):
            if i == j:
                continue
            cx2, cy2 = x2 + w2 // 2, y2 + h2 // 2
            distance = ((cx1 - cx2)**2 + (cy1 - cy2)**2)**0.5
            if distance < 10:  # 距离阈值，可微调
                too_close = True
                break
        if not too_close:
            final_boxes.append((x1, y1, w1, h1))

    # 绘制最终框
    for x, y, w, h in final_boxes:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)  # 红色粗框

    # 显示检测结果
    cv2.imshow('detect', frame)
    out.write(frame)

    prev_gray = gray
    if cv2.waitKey(30) == 27:  # ESC退出
        break

cap.release()
out.release()
cv2.destroyAllWindows()
