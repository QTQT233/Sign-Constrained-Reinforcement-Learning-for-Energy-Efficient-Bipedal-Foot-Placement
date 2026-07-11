import h5py
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# 设置参数
N1, N2 = 50, 50
total_elements = N1 * N2 * N1 * N2

# 文件路径配置
file_paths = {
    'active': 'C:/Users/Admin/Desktop/paper-code - F/working_save(-1,0,1)-50,50',
    'passive10': 'C:/Users/Admin/Desktop/paper-code - F/working_save(-1,0)-50,50',
    'passive01': 'C:/Users/Admin/Desktop/paper-code - F/working_save(0,1)-50,50',
    'continuous': 'D:/L&S/Mas/Project/Paper1/Energy_Comparison/60,30(0.33m)/working_save_active_continuous-50-50'
}

# 读取数据
def load_data(file_path, dataset_name='working_save'):
    """从HDF5文件加载数据"""
    with h5py.File(file_path, 'r') as h5f:
        return h5f[dataset_name][:]

print("Loading data...")
Pie_Active = load_data(file_paths['active'])
Pie_Passive10 = load_data(file_paths['passive10'])
Pie_Passive01 = load_data(file_paths['passive01'])
Pie_Continuous = load_data(file_paths['continuous'], 'working_save_active_continuous')

# 初始化计数器
counters = {
    'active': {-2: 0, -1: 0, 0: 0, 1: 0},
    'passive': {-2: 0, -1: 0, 0: 0, 1: 0},
    'continuous': {-2: 0, -1: 0, 0: 0, 1: 0}
}

print("Processing data...")
# 处理Passive数据 - 使用向量化操作替代循环
Pie_Passive = np.zeros_like(Pie_Passive10)

# 使用条件掩码进行向量化操作
mask_passive_2 = (Pie_Passive10 == -2) | (Pie_Passive01 == -2)
mask_passive_neg1 = (Pie_Passive10 == -1) & (Pie_Passive01 != 0)
mask_passive_1 = (Pie_Passive10 != 0) & (Pie_Passive01 == 1)
mask_passive_0 = (Pie_Passive10 == 0) | (Pie_Passive01 == 0)

Pie_Passive[mask_passive_2] = -2
Pie_Passive[mask_passive_neg1] = -1
Pie_Passive[mask_passive_1] = 1
Pie_Passive[mask_passive_0] = 0

# 统计Continuous数据 - 将-6映射到-2
continuous_mask_2 = Pie_Continuous == -6
continuous_mask_neg1 = (Pie_Continuous != -6) & (Pie_Continuous < 0)
continuous_mask_1 = (Pie_Continuous != -6) & (Pie_Continuous > 0)
continuous_mask_0 = (Pie_Continuous != -6) & (Pie_Continuous == 0)

counters['continuous'][-2] = np.sum(continuous_mask_2)
counters['continuous'][-1] = np.sum(continuous_mask_neg1)
counters['continuous'][1] = np.sum(continuous_mask_1)
counters['continuous'][0] = np.sum(continuous_mask_0)

# 统计Active和Passive数据
for value in [-2, -1, 0, 1]:
    counters['active'][value] = np.sum(Pie_Active == value)
    counters['passive'][value] = np.sum(Pie_Passive == value)

# 打印失败率
active_failure_rate = 1 - (counters['active'][-1] + counters['active'][1] + counters['active'][0]) / total_elements
passive_failure_rate = 1 - (counters['passive'][-1] + counters['passive'][1] + counters['passive'][0]) / total_elements
continuous_failure_rate = counters['continuous'][-2] / total_elements

print(f"Active failure rate: {active_failure_rate:.4f}")
print(f"Passive failure rate: {passive_failure_rate:.4f}")
print(f"Continuous failure rate: {continuous_failure_rate:.4f}")

# 准备绘图数据
labels = ['Fail', '-4 Nm', '0', '4 Nm']
active_sizes = [counters['active'][-2], counters['active'][-1], counters['active'][0], counters['active'][1]]
passive_sizes = [counters['passive'][-2], counters['passive'][-1], counters['passive'][0], counters['passive'][1]]
continuous_sizes = [counters['continuous'][-2], counters['continuous'][-1], counters['continuous'][0], counters['continuous'][1]]

# 转换为百分比
active_percent = np.array(active_sizes) / total_elements * 100
passive_percent = np.array(passive_sizes) / total_elements * 100
continuous_percent = np.array(continuous_sizes) / total_elements * 100

# 创建图表
plt.figure(figsize=[8, 5])
bar_width = 0.25
x_pos = np.arange(len(labels))

# 绘制柱状图
plt.bar(x_pos - bar_width, active_percent, bar_width, color='b', alpha=0.7, label='Active')
plt.bar(x_pos, passive_percent, bar_width, color='r', alpha=0.7, label='Passive')
plt.bar(x_pos + bar_width, continuous_percent, bar_width, color='g', alpha=0.7, label='Continuous')

plt.xlabel('Action Category')
plt.ylabel('Probability [%]')
plt.xticks(x_pos, labels)
plt.legend()
plt.title('Action Probability Distribution by Strategy')
plt.grid(True, axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

# 保存图表
# plt.savefig("C:/Users/Admin/Desktop/IEEE/Figure7.svg")
# plt.savefig("C:/Users/Admin/Desktop/IEEE/Figure7.pdf")
plt.show()

# 可选：显示一段时间后关闭
plt.pause(20)