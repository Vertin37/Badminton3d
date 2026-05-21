from openai import OpenAI
import os
import pandas as pd

# --- 配置区域 ---
CSV_FILE_PATH = r'e:\badminton\peoplepose\badminton_full_analysis.csv'  # 你的本地CSV路径
API_KEY = ""
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# --- 1. 读取本地分析好的 CSV 数据 ---
try:
    df = pd.read_csv(CSV_FILE_PATH)
    # 将 DataFrame 转换为文本格式
    csv_content = df.to_csv(index=False)
except Exception as e:
    print(f"读取文件失败: {e}")
    exit()

# --- 2. 组装完整的专家级提示词 (完全保留你的要求) ---
full_prompt = f"""# Role: 国家级羽毛球运动生物力学首席分析师 & 战术教练

## Context (背景)
我正在开发一套基于计算机视觉的羽毛球双人对战复盘系统。附件（或下方提供的数据）是系统提取的两位运动员（P0 和 P1）在连续比赛片段中的全时序生物力学与空间交互数据。

## Data Dictionary (数据字典说明)
请严格基于以下物理量定义进行分析：
- **Frame / Time(s)**: 视频帧数与绝对时间。
- **P0_V / P1_V (挥拍速率)**: 右手腕的瞬时像素位移速率（已按身高归一化）。数值的**局部峰值（Peak）代表击球爆发瞬间**。
- **P0_Arm / P1_Arm (持拍手伸展度)**: 肩-肘-腕关节夹角（单位：度）。击球瞬间接近 180° 为理想的充分伸展发力，小于 150° 通常意味着“缩手”或被动击球。
- **P0_HipDelta / P1_HipDelta (重心起伏)**: 髋部相对整场平均高度的偏移（归一化百分比）。**负值变小代表重心拔高（起跳/主动击球）**；**正值变大代表重心下沉（防守准备/接杀下蹲）**。
- **P0_Knee / P1_Knee (膝盖弯曲度)**: 髋-膝-踝夹角。防守时该角度变小（通常 < 160°）意味着双腿弯曲，处于良好的蓄力准备状态。
- **Inter_Dist (战术间距)**: 两人重心的水平距离（归一化）。间距的剧烈波动代表一人被另一人通过球路大范围调动。

## Objective (分析目标)
请仔细阅读提供的数据流，对 P0 和 P1 两名运动员的攻防表现进行极其客观、细致的量化分析，并给出针对性的专业建议。

## Constraints & Workflow (工作流要求)
请严格按照以下步骤进行推演，并在回答中引用具体的帧数（Frame）和数据值作为证据：

1. **寻找击球事件 (Event Detection)**:
 扫描 `P0_V` 和 `P1_V` 的极值点，以此定位真正的“攻防交锋”瞬间。
2. **进攻质量诊断 (Attacker Analysis)**:
 当一方挥速（V）达到峰值时，检查其对应时刻的 `Arm` 角度。评估其动力链传导是否顺畅、是否存在习惯性发力缺陷。
3. **防守预判与体态诊断 (Defender Analysis)**:
 当一方进攻（V达峰）时，**同步检查另一方**在同一帧及前 5 帧的 `HipDelta` 和 `Knee` 角度。评估防守方是“提前降低重心”还是“站立看球导致反应迟缓”。
4. **战术博弈复盘 (Tactical Spacing)**:
 结合 `Inter_Dist` 的时序变化，分析双方的跑位控制权。谁在主导节奏？

## Output Format (输出格式)
请以专业报告的形式输出：
### 一、 核心攻防事件数据还原
（挑选数据中 2-3 个最典型的高速对冲回合，用数据还原当时的场景）
### 二、 P0 运动员专属诊断报告
- **进攻端表现**：（引用运动员数据与标准数据评估）
- **防守端表现**：（引用运动员数据与标准数据评估）
- **核心短板与训练建议**：（给出 5 条具体建议）
### 三、 P1 运动员专属诊断报告
- **进攻端表现**：（引用运动员数据与标准数据评估）
- **防守端表现**：（引用运动员数据与标准数据评估）
- **核心短板与训练建议**：（给出 5 条具体建议）

---
附完整数据表：
{csv_content}
"""

# --- 3. 调用 OpenAI SDK (阿里云百炼) ---
client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
)

messages = [{"role": "user", "content": full_prompt}]

completion = client.chat.completions.create(
    model="qwen-max",  # 建议使用 max 处理大量 CSV 数据
    messages=messages,
    extra_body={"enable_thinking": True},
    stream=True
)

# --- 4. 处理流式输出 ---
is_answering = False
print("\n" + "=" * 20 + " 模型深度思考过程 " + "=" * 20)

for chunk in completion:
    delta = chunk.choices[0].delta
    
    # 输出思考过程
    if hasattr(delta, "reasoning_content") and delta.reasoning_content:
        print(delta.reasoning_content, end="", flush=True)
        
    # 输出正式分析报告
    if hasattr(delta, "content") and delta.content:
        if not is_answering:
            print("\n" + "=" * 20 + " 完整复盘报告输出 " + "=" * 20)
            is_answering = True
        print(delta.content, end="", flush=True)