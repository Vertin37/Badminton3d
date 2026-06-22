import cv2
import numpy as np
import math
from sklearn.cluster import KMeans
img_path = 'finalpic.jpg'  # 替换为你的图像路径
img = cv2.imread(img_path)


# ================= Step 1：读取图像 =================

if img is None:
    raise FileNotFoundError(f"无法加载图像：{img_path}")
img_display = img.copy()

# ================= Step 2：检测绿线边界（Hough） =================
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5,5), 0)
edges = cv2.Canny(blur, 50, 150)

lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100,
                        minLineLength=120, maxLineGap=10)

# 可视化绿线
if lines is not None:
    for (x1, y1, x2, y2) in lines[:,0]:
        cv2.line(img_display, (x1, y1), (x2, y2), (0,255,0), 2)
print(f"检测到绿线数量: {len(lines) if lines is not None else 0}")
vis = img.copy()
cv2.imshow("green",img_display)
cv2.waitKey(0)
cv2.destroyAllWindows()
# ================= Step 3：手动点击四个角点 =================# ================== Step3: 手动选择四点 + 匹配四条边（标注颜色） ==================
clicked_points = []

def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(clicked_points) < 4:
        clicked_points.append((x, y))
        cv2.circle(vis, (x, y), 5, (0, 0, 255), -1)
        cv2.putText(vis, f"P{len(clicked_points)}", (x+10, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
        cv2.imshow("Select 4 Points (A,B,C,D)", vis)

# 创建显示副本
vis_copy = vis.copy()
cv2.imshow("Select 4 Points (A,B,C,D)", vis_copy)
cv2.setMouseCallback("Select 4 Points (A,B,C,D)", click_event)
cv2.waitKey(0)
cv2.destroyAllWindows()

if len(clicked_points) != 4:
    raise ValueError("请点击四个矩形顶点！")

# 四个点
pts_clicked = [np.array(p, dtype=float) for p in clicked_points]

# 匹配每条边到绿线
matched_lines = []

def line_distance(p1, p2, line):
    """计算点连线与候选线段的距离平均值"""
    x1, y1, x2, y2 = line
    def point_line_dist(px, py):
        num = abs((y2-y1)*px - (x2-x1)*py + x2*y1 - y2*x1)
        den = math.hypot(y2-y1, x2-x1)
        return num/den
    return (point_line_dist(*p1) + point_line_dist(*p2)) / 2

# 四条边顺序 AB, BC, CD, DA
edge_order = [(0,1), (1,2), (2,3), (3,0)]
pts_rect = []

# 颜色列表（4条边不同颜色）
colors = [(0,0,255), (0,255,0), (255,0,0), (0,255,255)]

vis_edges = vis.copy()

for idx, (idx_start, idx_end) in enumerate(edge_order):
    p_start = pts_clicked[idx_start]
    p_end = pts_clicked[idx_end]
    best_line = None
    min_dist = float('inf')
    for line in lines[:,0]:
        dist = line_distance(p_start, p_end, line)
        if dist < min_dist:
            min_dist = dist
            best_line = line
    matched_lines.append(best_line)
    # 保留起点作为矩形顶点
    pts_rect.append(p_start)
    # 绘制匹配到的线段
    x1, y1, x2, y2 = best_line
    cv2.line(vis_edges, (x1, y1), (x2, y2), colors[idx], 3)

pts_rect = np.array(pts_rect, dtype=float)


print("\n匹配到的边线信息 (x1,y1,x2,y2)：")
for line in matched_lines:
    print(line)
# ================== Step3.1: 延长匹配线段并求交点 ==================
def line_params_from_segment(line):
    """返回线段的直线参数 a*x + b*y + c = 0"""
    x1, y1, x2, y2 = line
    if abs(x2 - x1) < 1e-6:  # 垂直线
        return (1, 0, -x1)
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return (m, -1, b)

def intersection(L1, L2):
    """求两条直线交点"""
    A1, B1, C1 = L1
    A2, B2, C2 = L2
    det = A1 * B2 - A2 * B1
    if abs(det) < 1e-6:
        return None
    x = (B1 * C2 - B2 * C1) / det
    y = (C1 * A2 - C2 * A1) / det
    return np.array([x, y])

# 匹配线段顺序 AB, BC, CD, DA
extended_lines = [line_params_from_segment(l) for l in matched_lines]

# 求交点更新 pts_rect
pts_rect_new = []
for i in range(4):
    L1 = extended_lines[i]
    L2 = extended_lines[(i+1)%4]
    pt = intersection(L1, L2)
    if pt is None:
        raise ValueError(f"边 {i}-{(i+1)%4} 无交点")
    pts_rect_new.append(pt)

pts_rect = np.array(pts_rect_new, dtype=float)
print("\n✅ 更新后的矩形顶点 pts_rect：")
for i, pt in enumerate(pts_rect):
    print(f"P{i} = {pt}")

# ================== 显示匹配结果 ==================
cv2.imshow("Matched Rectangle Edges", vis_edges)
cv2.waitKey(0)
cv2.destroyAllWindows()


# ================= Step 4：单击选择网线 =================
selected_line = []
def click_net_line(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(selected_line)==0:
        # 找到最近绿线
        min_dist = float('inf')
        nearest = None
        for (x1,y1,x2,y2) in lines[:,0]:
            # 距离点到线段的距离
            px = x2 - x1
            py = y2 - y1
            norm = px*px + py*py
            u = ((x - x1)*px + (y - y1)*py)/norm if norm!=0 else 0
            u = max(0,min(1,u))
            closest_x = x1 + u*px
            closest_y = y1 + u*py
            dist = math.hypot(x - closest_x, y - closest_y)
            if dist < min_dist:
                min_dist = dist
                nearest = (x1,y1,x2,y2)
        selected_line.append(nearest)
        x1,y1,x2,y2 = nearest
        cv2.line(img_display,(x1,y1),(x2,y2),(255,0,255),3)
        cv2.imshow("Select net line", img_display)

cv2.imshow("Select net line", img_display)
cv2.setMouseCallback("Select net line", click_net_line)
cv2.waitKey(0)
cv2.destroyAllWindows()

if len(selected_line)!=1:
    raise ValueError("请点击选择一条网线")

net_line = selected_line[0]

# ================= Step 5：选择 EF 两点竖直基准线 =================
ef_points = []
def click_ef_points(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(ef_points)<2:
        ef_points.append((x,y))
        cv2.circle(img_display,(x,y),5,(0,255,255),-1)
        cv2.putText(img_display,f"E/F{len(ef_points)}",(x+10,y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,0),2)
        cv2.imshow("Select EF line", img_display)

cv2.imshow("Select EF line", img_display)
cv2.setMouseCallback("Select EF line", click_ef_points)
cv2.waitKey(0)
cv2.destroyAllWindows()

if len(ef_points)!=2:
    raise ValueError("请点击选择 EF 两点")

E,F = ef_points

# ================= Step 6：绘制最终结果 =================
final_display = img.copy()

# 绘制绿线 Hough
for (x1,y1,x2,y2) in lines[:,0]:
    cv2.line(final_display,(x1,y1),(x2,y2),(0,255,0),2)

# 绘制红色矩形
cv2.polylines(final_display,[pts_rect.astype(int)],isClosed=True,color=(0,0,255),thickness=3)

# 绘制选中网线
x1,y1,x2,y2 = net_line
cv2.line(final_display,(x1,y1),(x2,y2),(255,0,255),3)

# 绘制 EF 竖直基准线
cv2.line(final_display,tuple(E),tuple(F),(0,255,255),2)

# 绘制角点
for i,p in enumerate(pts_rect):
    cv2.circle(final_display,tuple(p.astype(int)),6,(255,0,0),-1)
    cv2.putText(final_display,f"P{i+1}",(int(p[0])+10,int(p[1])-10),
                cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,0),2)

# 绘制 EF 两点
for i,p in enumerate([E,F]):
    cv2.circle(final_display,tuple(p),5,(0,255,255),-1)
    cv2.putText(final_display,f"E/F{i+1}",(p[0]+10,p[1]-10),
                cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,0),2)

cv2.imshow("Final Court with Net & EF", final_display)
cv2.imwrite("court_with_net_ef.jpg",final_display)
print("✅ 已保存结果: court_with_net_ef.jpg")

cv2.waitKey(0)
cv2.destroyAllWindows()
# ================== Step X: 打印关键变量信息 ==================
print("\n===== 当前版本关键变量信息 =====")
print(f"四角点 pts_rect (A,B,C,D):\n{pts_rect}")
print(f"选中的中网线 net_line: {net_line}")
print(f"竖直基准线 E,F: {E}, {F}")
print(f"检测到的绿线数量: {len(lines) if lines is not None else 0}")

# 如果需要，也可以打印每条绿线的坐标
#if lines is not None:
#    print("\n绿线 Hough 检测结果 (每行: x1,y1,x2,y2):")
#    for idx, (x1,y1,x2,y2) in enumerate(lines[:,0]):
#        print(f"Line {idx+1}: ({x1},{y1}) -> ({x2},{y2})")
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
clicked_M = []

def click_M(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_M.append((x, y))
        cv2.circle(param, (x, y), 6, (0,0,255), -1)
        cv2.putText(param, "M", (x+10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        cv2.imshow("Select M (Ball)", param)

# 显示图像，手动选择 M
img_temp = img.copy()
cv2.imshow("Select M (Ball)", img_temp)
cv2.setMouseCallback("Select M (Ball)", click_M, img_temp)
cv2.waitKey(0)
cv2.destroyAllWindows()

if len(clicked_M) == 0:
    raise ValueError("请在图像上点击羽毛球位置 M!")

M_pixel = clicked_M[0]
Z_ball=1.8
# 调用投影函数
def project_ball_to_ground(img, M_pixel, E, F, Z_ball=1.8, Z_net=1.55):
    M = np.array(M_pixel, dtype=float)
    E = np.array(E, dtype=float)
    F = np.array(F, dtype=float)
    
    v_EF = F - E
    length_EF = np.linalg.norm(v_EF)
    v_unit = v_EF / length_EF
    
    ratio = Z_ball / Z_net
    N = M - ratio * length_EF * v_unit
    
    img_vis = img.copy()
    cv2.circle(img_vis, tuple(np.round(M).astype(int)), 6, (0,0,255), -1)
    cv2.putText(img_vis, "M", (int(M[0])+10,int(M[1])-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
    cv2.circle(img_vis, tuple(np.round(N).astype(int)), 6, (0,255,0), -1)
    cv2.putText(img_vis, "N", (int(N[0])+10,int(N[1])-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    
    return N, img_vis

N_pixel, img_with_MN = project_ball_to_ground(img, M_pixel,  E, F)

# 显示结果
cv2.imshow("M and N Projection", img_with_MN)
cv2.waitKey(0)
cv2.destroyAllWindows()

print(f"M 点像素坐标: {M_pixel}")
print(f"N 点像素坐标（地面投影）: {N_pixel}")
#####################################################根据N还原

def map_pixel_to_real(N_pixel, pts_rect, court_length=13.4, court_width=6.1):
    """
    将图像中投影点 N_pixel 映射到真实场地坐标 N_real
    Args:
        N_pixel: (x, y) 像素坐标
        pts_rect: np.array 四角像素坐标 [[Ax,Ay],[Bx,By],[Cx,Cy],[Dx,Dy]]
        court_length: 场地长度（米）
        court_width: 场地宽度（米）
    Returns:
        N_real: (X, Y) 实际二维坐标
    """

    # 真实场地矩形坐标（顺序对应 pts_rect）
    pts_real = np.array([
        [0, 0],                     # A
        [court_width, 0],           # B
        [court_width, court_length],# C
        [0, court_length]           # D
    ], dtype=np.float32)

    pts_pixel = np.array(pts_rect, dtype=np.float32)

    # 计算透视变换矩阵：图像坐标 -> 实际场地坐标
    H = cv2.getPerspectiveTransform(pts_pixel, pts_real)

    # 将 N_pixel 转为齐次坐标 [x, y, 1]
    N_h = np.array([N_pixel[0], N_pixel[1], 1.0], dtype=np.float32).reshape(3,1)

    # 应用透视变换
    N_real_h = H @ N_h  # 3x3 @ 3x1 -> 3x1

    # 归一化齐次坐标
    N_real = (N_real_h[:2] / N_real_h[2]).flatten()

    print(f"N_pixel = {N_pixel}")
    print(f"N_real = {N_real}")

    return N_real
N_real = map_pixel_to_real(N_pixel, pts_rect)

print("N_real:", N_real)
#####绘图#####################################################


##############################################################


import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib import rcParams

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 黑体
plt.rcParams['axes.unicode_minus'] = False    #
def plot_3d_shuttlecourt_with_EF(N_real, Z_ball, court_length=13.4, court_width=6.1, net_height=1.55):
    """
    绘制三维羽毛球场景，包括：
    - 球场矩形
    - 球网（真实矩形）
    - 球位置 M 和地面投影 N
    - EF 竖直参考线
    
    Args:
        N_real: 球投影在地面坐标 [X, Y]
        Z_ball: 球高度 (米)
        court_length: 球场长度 (米)
        court_width: 球场宽度 (米)
        net_height: 球网高度 (米)
    """
    fig = plt.figure(figsize=(court_width, court_length / court_width * court_width))
    ax = fig.add_subplot(111, projection='3d')
    
    # ===== 球场矩形 =====
    pts_real = np.array([
        [0, 0],
        [court_width, 0],
        [court_width, court_length],
        [0, court_length],
        [0, 0]
    ])
    ax.plot(pts_real[:,0], pts_real[:,1], np.zeros_like(pts_real[:,0]), color='blue', linewidth=2, label='Court Boundary')
    
    # ===== 中网真实矩形 =====
    net_y = court_length / 2
    net_thickness = 0.05  # 网厚度 5cm
    net_x = np.linspace(0, court_width, 2)
    net_z = np.linspace(0, net_height, 2)
    net_X, net_Z = np.meshgrid(net_x, net_z)
    net_Y, _ = np.meshgrid(np.array([net_y - net_thickness/2, net_y + net_thickness/2]), net_z)
    ax.plot_surface(net_X, net_Y, net_Z, color='orange', alpha=0.6)
    
    # ===== EF 竖直参考线 =====
    EF_x = 0
    EF_y = 0  # 位置随意，可调整
    EF_bottom = 0
    EF_top = net_height
    #ax.plot([EF_x, EF_x], [EF_y, EF_y], [EF_bottom, EF_top], color='green', linewidth=3, label='Vertical EF Line')
    
    # ===== 球 M =====
    ax.scatter(N_real[0], N_real[1], Z_ball, color='red', s=100, label='球M')
    ax.text(N_real[0], N_real[1], Z_ball + 0.1, 'M', color='red', fontsize=12)
    
    # ===== 球地面投影 N =====
    ax.scatter(N_real[0], N_real[1], 0, color='green', s=80, label='球向地面投影N')
    ax.text(N_real[0], N_real[1], 0.1, 'N', color='green', fontsize=12)
    
    # ===== 坐标轴与视角 =====
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_xlim(-1, court_width + 1)
    ax.set_ylim(-1, court_length + 1)
    ax.set_zlim(0, max(Z_ball, net_height + 0.5))
    ax.set_title('球场匹配3D视图')
    ax.view_init(elev=30, azim=-60)
    ax.legend()
    ax.grid(True)
    
    plt.show()
    
    print(f"球M的真实坐标: [X={N_real[0]:.2f}, Y={N_real[1]:.2f}, Z={Z_ball:.2f}]")
    #print(f"Projection N on ground: [X={N_real[0]:.2f}, Y={N_real[1]:.2f}, Z=0.00]")
    #print(f"Net rectangle surface from Y={net_y - net_thickness/2:.2f} to Y={net_y + net_thickness/2:.2f}, height={net_height} m")
    #print(f"EF line at X={EF_x:.2f}, Y={EF_y:.2f}, from Z=0 to Z={EF_top:.2f}")
plot_3d_shuttlecourt_with_EF(N_real, Z_ball)

