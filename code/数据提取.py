import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
import os

class BadmintonDualOmniAnalyzer:
    def __init__(self, trc_p0, trc_p1, fps=25):
        self.fps = fps
        # 加载并平滑数据
        self.p0 = self._process_trc(trc_p0)
        self.p1 = self._process_trc(trc_p1)
        self.min_len = min(len(self.p0), len(self.p1))
        
        # 计算参考身高（用于归一化，解决近大远小问题）
        self.h0 = self._get_ref_h(self.p0)
        self.h1 = self._get_ref_h(self.p1)

    def _process_trc(self, path):
        df = pd.read_csv(path, sep='\t', skiprows=4)
        df.columns = [c.strip() for c in df.columns]
        # 对所有坐标进行平滑处理
        for col in df.columns:
            if any(axis in col for axis in ['X', 'Y']):
                # 填充空值并平滑
                data = pd.to_numeric(df[col], errors='coerce').ffill().bfill()
                df[col] = savgol_filter(data, 11, 3)
        return df

    def _get_ref_h(self, df):
        # 鼻尖(17)到脚踝(4)的平均距离
        return abs(df['Y17'].mean() - df['Y4'].mean())

    def _get_angle(self, df, m1, m2, m3, f):
        """计算指定帧的关节角度"""
        a = np.array([df[f'X{m1}'][f], df[f'Y{m1}'][f]])
        b = np.array([df[f'X{m2}'][f], df[f'Y{m2}'][f]])
        c = np.array([df[f'X{m3}'][f], df[f'Y{m3}'][f]])
        ba, bc = a - b, c - b
        norm = np.linalg.norm(ba) * np.linalg.norm(bc)
        return np.degrees(np.arccos(np.clip(np.dot(ba, bc)/norm, -1, 1))) if norm > 0 else 0

    def run_analysis(self):
        full_data = []
        
        for f in range(1, self.min_len):
            # 1. 瞬时速度 (归一化：速度/身高)
            v0 = np.sqrt((self.p0['X20'][f]-self.p0['X20'][f-1])**2 + (self.p0['Y20'][f]-self.p0['Y20'][f-1])**2) / self.h0
            v1 = np.sqrt((self.p1['X20'][f]-self.p1['X20'][f-1])**2 + (self.p1['Y20'][f]-self.p1['Y20'][f-1])**2) / self.h1
            
            # 2. 重心状态 (Hip Y轴相对于各自平均身高的偏移)
            # 越小代表重心越高（起跳），越大代表越低（深蹲）
            p0_hip_rel = (self.p0['Y1'][f] - self.p0['Y1'].mean()) / self.h0
            p1_hip_rel = (self.p1['Y1'][f] - self.p1['Y1'].mean()) / self.h1
            
            # 3. 双人空间交互：水平间距
            inter_dist = abs(self.p0['X1'][f] - self.p1['X1'][f]) / ((self.h0 + self.h1)/2)

            row = {
                'Frame': f,
                'Time(s)': round(f/self.fps, 2),
                # 两人挥拍速度
                'P0_V': round(v0, 4), 'P1_V': round(v1, 4),
                # 两人手臂伸展度
                'P0_Arm': round(self._get_angle(self.p0, 18, 19, 20, f), 1),
                'P1_Arm': round(self._get_angle(self.p1, 18, 19, 20, f), 1),
                # 两人重心起伏 (负值代表重心拔高，正值代表下沉)
                'P0_HipDelta': round(p0_hip_rel, 3),
                'P1_HipDelta': round(p1_hip_rel, 3),
                # 两人膝盖角度 (判断准备姿态)
                'P0_Knee': round(self._get_angle(self.p0, 2, 3, 4, f), 1),
                'P1_Knee': round(self._get_angle(self.p1, 2, 3, 4, f), 1),
                # 战术间距
                'Inter_Dist': round(inter_dist, 2)
            }
            full_data.append(row)
            
        return pd.DataFrame(full_data)

# --- 运行提取 ---
file0 = r'E:\badminton\peoplepose\Sports2D_Output\final_Sports2D\final_Sports2D_px_person00.trc'
file1 = r'E:\badminton\peoplepose\Sports2D_Output\final_Sports2D\final_Sports2D_px_person01.trc'

analyzer = BadmintonDualOmniAnalyzer(file0, file1)
raw_stats = analyzer.run_analysis()

# 输出关键数据摘要（例如前100帧）
print(f"提取完成！总帧数: {len(raw_stats)}")
print(raw_stats.head(100))

# 建议将此数据保存为 CSV，然后上传给 Qwen
raw_stats.to_csv('badminton_full_analysis.csv', index=False)