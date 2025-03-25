import matplotlib
import numpy as np
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# 数据定义
models = ['MHAN-Light(ours)', 'ELAN', 'SwinIR', 'RFDN', 'IMDN', 'ShuffleMixer', 'PAN', 'CARN']
params = [697, 601, 897, 550, 715, 411, 272, 1592]  # Parameters (K)
psnr = [32.40, 32.43, 32.44, 32.24, 32.21, 32.21, 32.13, 32.13]  # PSNR (dB)
flops = [52, 43, 49, 24, 40, 28, 28, 90]  # Flops (G)
sizes = [flop * 30 for flop in flops]  # 气泡大小（与 Flops 相关）

# 定义气泡颜色，根据 Flops 分级
colors = []
for flop in flops:
    if flop <= 30:
        colors.append('lightgreen')
    elif flop <= 60:
        colors.append('skyblue')
    elif flop <= 90:
        colors.append('blue')
    else:
        colors.append('purple')

# 绘制气泡图
plt.figure(figsize=(10, 6))
plt.scatter(params, psnr, s=sizes, c=colors, alpha=0.6, edgecolors='black')
# 设置纵坐标范围和间隔
plt.yticks(np.arange(32.1, 32.6, 0.1))  # 纵坐标从 32.0 到 32.5，每隔 0.1 一格

# 添加标签
for i, model in enumerate(models):
    plt.text(params[i], psnr[i], model, fontsize=9, ha='center', va='center')

# 图例
import matplotlib.patches as mpatches
legend_handles = [
    mpatches.Patch(color='lightgreen', label='<=30'),
    mpatches.Patch(color='skyblue', label='30-60G'),
    mpatches.Patch(color='blue', label='60-90G'),
    mpatches.Patch(color='purple', label='>90G'),
]
plt.legend(handles=legend_handles, title='Flops', loc='upper right')

# 设置轴标签和标题
plt.xlabel('Parameters (K)')
plt.ylabel('PSNR (dB)')
plt.title('Params vs Flops vs PSNR')

# 设置网格
plt.grid(True, linestyle='--', alpha=0.6)

# 显示图像
plt.tight_layout()
plt.show()
