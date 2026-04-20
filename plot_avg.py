import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LogNorm

# 提取并计算得到的 Brand A 平均值矩阵 (9x9)
avg_data_blue = np.array([
    [0.05, 0.35, 0.65, 1.05, 0.70, 0.60, 1.05, 1.05, 1.20],
    [0.10, 0.55, 0.70, 0.55, 0.75, 0.60, 1.20, 0.85, 1.30],
    [0.00, 0.40, 0.70, 0.85, 0.90, 0.65, 0.80, 1.35, 1.10],
    [0.05, 0.65, 0.45, 0.85, 0.65, 0.65, 0.70, 0.85, 0.85],
    [0.00, 0.50, 0.60, 0.55, 0.70, 0.95, 0.70, 1.15, 0.70],
    [0.00, 0.65, 0.65, 0.85, 0.90, 0.75, 1.00, 1.05, 1.05],
    [0.05, 0.50, 0.65, 0.75, 0.90, 0.85, 0.80, 1.00, 0.85],
    [0.00, 0.40, 0.55, 0.60, 0.80, 0.75, 1.05, 1.15, 1.10],
    [0.00, 0.50, 0.65, 0.75, 0.85, 0.85, 1.15, 1.00, 0.80]
])

# 坐标轴标签
x_labels = [0, 100, 200, 300, 400, 500, 600, 700, 800]
y_labels = [800, 700, 600, 500, 400, 300, 200, 100, 0]

# 设置绘图画布大小和分辨率
plt.figure(figsize=(10, 8), dpi=150)

# 使用 seaborn 绘制热力图，改为 Blues 配色以匹配 Brand A
sns.heatmap(avg_data_blue, 
            annot=True, 
            fmt=".3g", 
            cmap="Blues", 
            xticklabels=x_labels, 
            yticklabels=y_labels, 
            linewidths=0.5, 
            linecolor='white',
            norm=LogNorm(vmin=0.1, vmax=max(0.1, avg_data_blue.max())),
            cbar_kws={'label': 'Value (Log Scale)'})

plt.title("Average Heatmap: Brand A\n(Subtracted Baseline & Zero-Clipped)", fontsize=14)
plt.xlabel("X_top_k", fontsize=12)
plt.ylabel("Y_top_k", fontsize=12)

# 直接保存在当下路径
plt.savefig("average_heatmap_brand_a_blue.png", dpi=300, bbox_inches='tight')
print("蓝色热力图已成功保存为 average_heatmap_brand_a_blue.png")

# 如果需要预览可以取消注释下面这行
# plt.show()