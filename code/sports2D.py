import pandas as pd
import numpy as np
import os

def simulate_racket_position(trc_file_path, output_path, extrapolation_factor=2.5):
    # 1. 自动寻找包含关键点名称的行
    marker_line_index = -1
    clean_names = []
    
    with open(trc_file_path, 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            parts = [p.strip() for p in line.split('\t') if p.strip()]
            if 'RElbow' in parts or 'RWrist' in parts:
                marker_line_index = i
                clean_names = parts
                break
    
    if marker_line_index == -1:
        print("错误：在文件中找不到 'RElbow' 或 'RWrist'。请确认文件内容是否包含这些点。")
        return

    try:
        # 在清洗后的名称列表中找到位置
        elbow_pos = clean_names.index('RElbow')
        wrist_pos = clean_names.index('RWrist')
        
        # TRC标准：前两列是Frame/Time，后面每个点占3列
        # 索引计算：2 + (位置在列表中的索引 * 3)
        # 注意：这里假设列表是从第一个关键点开始的
        idx_ex = 2 + (elbow_pos * 3)
        idx_wx = 2 + (wrist_pos * 3)
        
        print(f"成功在第 {marker_line_index + 1} 行找到关键点！")
        print(f"列索引：RElbow({idx_ex}), RWrist({idx_wx})")
    except Exception as e:
        print(f"计算列索引出错: {e}")
        return

    # 2. 读取数据区（跳过所有表头行，通常数据从第6行开始，即索引5）
    df = pd.read_csv(trc_file_path, sep='\t', skiprows=5, header=None)

    # 3. 强制转换并提取坐标
    ex = pd.to_numeric(df[idx_ex], errors='coerce')
    ey = pd.to_numeric(df[idx_ex + 1], errors='coerce')
    wx = pd.to_numeric(df[idx_wx], errors='coerce')
    wy = pd.to_numeric(df[idx_wx + 1], errors='coerce')

    # 4. 向量计算与外推
    vx = wx - ex
    vy = wy - ey
    mag = np.sqrt(vx**2 + vy**2).replace(0, 0.001)

    racket_x = wx + (vx / mag) * (mag * extrapolation_factor)
    racket_y = wy + (vy / mag) * (mag * extrapolation_factor)

    # 5. 保存结果
    pd.DataFrame({
        'Frame': range(len(df)),
        'Wrist_X': wx, 'Wrist_Y': wy,
        'RacketHead_X': racket_x, 'RacketHead_Y': racket_y
    }).to_csv(output_path, index=False)
    
    print(f"--- 模拟成功！数据已存至: {output_path} ---")

# --- 运行 ---
input_trc = r'E:\badminton\peoplepose\Sports2D_Output\final_Sports2D\final_Sports2D_px_person00.trc'
output_csv = r'E:\badminton\peoplepose\Output\racket_simulated.csv'
simulate_racket_position(input_trc, output_csv)