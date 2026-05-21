import cv2
import numpy as np
import os

def generate_color_extracted_video(original_video_path, processed_sam_video_path, output_path, threshold_val=20):
    """
    计算原视频和 SAM 处理后视频的差异，生成黑底但保留分割区域颜色的视频。
    """
    print(f"开始处理: {original_video_path} 和 {processed_sam_video_path}")
    
    cap_orig = cv2.VideoCapture(original_video_path)
    cap_sam = cv2.VideoCapture(processed_sam_video_path)

    if not cap_orig.isOpened() or not cap_sam.isOpened():
        print("错误：无法打开一个或两个视频文件。请检查路径。")
        return

    # 获取视频属性
    frame_width = int(cap_orig.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap_orig.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap_orig.get(cv2.CAP_PROP_FPS)

    # 【修改1】isColor 必须为 True，因为我们要保留颜色
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height), isColor=True)

    frame_id = 0
    while True:
        ret_orig, frame_orig = cap_orig.read()
        ret_sam, frame_sam = cap_sam.read()

        if not ret_orig or not ret_sam:
            break

        # 尺寸兼容性检查
        if frame_sam.shape != frame_orig.shape:
            frame_sam = cv2.resize(frame_sam, (frame_orig.shape[1], frame_orig.shape[0]))

        # 1. 计算两帧之间的绝对差异
        diff = cv2.absdiff(frame_orig, frame_sam)

        # 2. 转换为灰度并二值化生成 Mask
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        _, binary_mask = cv2.threshold(gray_diff, threshold_val, 255, cv2.THRESH_BINARY)
        kernel = np.ones((3,3), np.uint8)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel) # 闭运算填补小洞
        # 【修改2】使用位运算提取颜色
        # 这一步会将 binary_mask 为 0 的地方变为黑色，为 255 的地方保留 frame_sam 的原始颜色
        color_extracted_frame = cv2.bitwise_and(frame_sam, frame_sam, mask=binary_mask)

        # 3. 写入输出视频
        out.write(color_extracted_frame)
        
        frame_id += 1
        if frame_id % 100 == 0:
            print(f"已处理 {frame_id} 帧...")

    cap_orig.release()
    cap_sam.release()
    out.release()
    cv2.destroyAllWindows()
    
    print(f"\n处理完成，彩色提取视频已保存：{output_path}")

if __name__ == "__main__":
    ORIGINAL_VIDEO_PATH = "球拍+球优化.mp4"
    SAM_PROCESSED_VIDEO_PATH = "sam_球拍+球优化.mp4"
    OUTPUT_COLOR_PATH = "球拍+球优化_color_extracted_result.mp4"
    THRESHOLD_VALUE = 30 # 稍微调低可以保留更多边缘细节

    if os.path.exists(ORIGINAL_VIDEO_PATH) and os.path.exists(SAM_PROCESSED_VIDEO_PATH):
        generate_color_extracted_video(
            ORIGINAL_VIDEO_PATH, 
            SAM_PROCESSED_VIDEO_PATH, 
            OUTPUT_COLOR_PATH,
            THRESHOLD_VALUE
        )