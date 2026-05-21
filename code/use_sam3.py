import cv2
import numpy as np
import os
def init_video_writer(cap, save_path, fps=30):
    """
    初始化视频写入器
    """
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 通用 mp4 编码
    writer = cv2.VideoWriter(save_path, fourcc, fps, (width, height))
    return writer

def generate_binary_mask_video(original_video_path, processed_sam_video_path, output_mask_path, threshold_val=20):
    """
    通过计算原视频和 SAM 处理后视频的像素差异，生成纯黑背景上的纯白 Mask 视频。

    Args:
        original_video_path (str): 原始视频文件路径。
        processed_sam_video_path (str): SAM 处理后的视频文件路径（包含分割标记）。
        output_mask_path (str): 生成的二值 Mask 视频文件保存路径。
        threshold_val (int): 用于将差异图二值化的阈值 (0-255)。值越小，对差异越敏感。
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

    # 定义 VideoWriter 编码器和输出文件
    # 使用 'mp4v' 编码器，并确保输出是灰度 (单通道) 或三通道 (但只写入黑白)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_mask_path, fourcc, fps, (frame_width, frame_height), isColor=False) # isColor=False 确保输出是单通道灰度

    frame_id = 0
    while True:
        ret_orig, frame_orig = cap_orig.read()
        ret_sam, frame_sam = cap_sam.read()

        if not ret_orig or not ret_sam:
            break

        # 1. 计算两帧之间的绝对差异
        # 差异图会突出 SAM 处理视频中与原视频不一致的像素 (即分割区域或高亮)
        #print("frame_orig shape:", frame_orig.shape if ret_orig else None)
        #print("frame_sam shape:", frame_sam.shape if ret_sam else None)
# 1. 计算两帧之间的绝对差异
# SAM 视频可能与原视频尺寸不同，这里强制 resize
        if frame_sam.shape != frame_orig.shape:
            frame_sam = cv2.resize(frame_sam, (frame_orig.shape[1], frame_orig.shape[0]))

        diff = cv2.absdiff(frame_orig, frame_sam)

        # 2. 将差异图转换为灰度，以便进行统一的二值化处理
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

        # 3. 二值化：将差异大于阈值的区域设为纯白 (255)，其余设为纯黑 (0)
        # 这一步生成了所需的纯白 on 纯黑的 Mask
        _, binary_mask = cv2.threshold(gray_diff, threshold_val, 255, cv2.THRESH_BINARY)

        # 4. 写入输出视频
        out.write(binary_mask)
        
        frame_id += 1
        if frame_id % 100 == 0:
            print(f"已处理 {frame_id} 帧...")

    cap_orig.release()
    cap_sam.release()
    out.release()
    cv2.destroyAllWindows()
    
    print(f"\n处理完成，共生成 {frame_id} 帧 Mask 视频：{output_mask_path}")


if __name__ == "__main__":
    # --- 请替换为你自己的真实文件路径 ---
    ORIGINAL_VIDEO_PATH = "12月28日 (1)(1).mp4"
    SAM_PROCESSED_VIDEO_PATH = "sam_人+球1.mp4"
    
    # 这是你在 localize_player.py 中需要引用的 Mask 视频路径
    OUTPUT_MASK_PATH = "uesesam_sam_人+球1.mp4"
    
    # 阈值：如果 SAM 标记区域与背景差异很小，可以降低此值 (默认 20)
    # 如果 SAM 标记区域颜色很亮，可以提高此值
    THRESHOLD_VALUE = 40

    # 检查输入文件是否存在 (请确保替换了占位符路径)
    if os.path.exists(ORIGINAL_VIDEO_PATH) and os.path.exists(SAM_PROCESSED_VIDEO_PATH):
        generate_binary_mask_video(
            ORIGINAL_VIDEO_PATH, 
            SAM_PROCESSED_VIDEO_PATH, 
            OUTPUT_MASK_PATH,
            THRESHOLD_VALUE
        )
    else:
        print("\n请修改脚本中的 ORIGINAL_VIDEO_PATH 和 SAM_PROCESSED_VIDEO_PATH 为真实文件路径后再运行！")