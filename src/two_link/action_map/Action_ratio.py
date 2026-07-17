import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import h5py
import torch
import imageio
import numpy as np
from tqdm import tqdm
from numba import njit
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# N1 = 10
# N2 = 30
N1 = 30
N2 = 30
N = N1
speed_nondim = 3.5
tht1s = np.linspace(60, 120, N1) * np.pi / 180
# dtht1s = np.linspace(-speed_nondim, speed_nondim, N2)
dtht1s = np.linspace(-2, 2, N2)
tht2s = np.linspace(-60, 60, N1) * np.pi / 180
# dtht2s = np.linspace(-speed_nondim, speed_nondim, N2)
dtht2s = np.linspace(-2, 2, N2)
with h5py.File('C:/Users/Admin/Desktop/paper-code - F/working_save(-1,0,1)-30,30_from_2-2', 'r') as h5f:
    data4 = h5f['working_save'][:]
Pie_Active = data4
# with h5py.File('C:/Users/Admin/Desktop/paper-code - F/check_save(-1,0,1)-30,30', 'r') as h5f:
#     data5 = h5f['check_save'][:]
# Pie_Active_check = data5
with h5py.File('C:/Users/Admin/Desktop/paper-code - F/working_save(-1,0)-30,30', 'r') as h5f:
    data6 = h5f['working_save'][:]
Pie_Passive10 = data6
with h5py.File('C:/Users/Admin/Desktop/paper-code - F/working_save(0,1)-30,30', 'r') as h5f:
    data7 = h5f['working_save'][:]
Pie_Passive01 = data7


# N1 = 50
# N2 = 50
N1 = 30
N2 = 30
cont_active_2 = 0
cont_active0 = 0
cont_active_1 = 0
cont_active1 = 0
cont_active_action = 0
cont_active_noaction = 0
cont_passive_2 = 0
cont_passive0 = 0
cont_passive_1 = 0
cont_passive1 = 0
con = 0
con_total = 0
Pie_Passive = np.zeros((N1, N2, N1, N2))
for i in tqdm(range(N1)):
    for j in range(N2):
        for k in range(N1):
            for ll in range(N2):
                if Pie_Passive10[i, j, k, ll] == -1 and Pie_Passive01[i, j, k, ll] != 0:
                    Pie_Passive[i, j, k, ll] = -1
                elif Pie_Passive10[i, j, k, ll] != 0 and Pie_Passive01[i, j, k, ll] == 1:
                    Pie_Passive[i, j, k, ll] = 1
                elif Pie_Passive10[i, j, k, ll] == 0 or Pie_Passive01[i, j, k, ll] == 0:
                    Pie_Passive[i, j, k, ll] = 0
                elif Pie_Passive10[i, j, k, ll] == -2 or Pie_Passive01[i, j, k, ll] == -2:
                    Pie_Passive[i, j, k, ll] = -2
                if Pie_Active[i, j, k, ll] == -2:
                    cont_active_2 += 1
                    con += 1
                elif Pie_Active[i, j, k, ll] == -1:
                    cont_active_1 += 1
                    con += 1
                elif Pie_Active[i, j, k, ll] == 1:
                    cont_active1 += 1
                    con += 1
                elif Pie_Active[i, j, k, ll] == 0:
                    cont_active0 += 1
                    con += 1
                if Pie_Passive[i, j, k, ll] == -2:
                    cont_passive_2 += 1
                elif Pie_Passive[i, j, k, ll] == -1:
                    cont_passive_1 += 1
                elif Pie_Passive[i, j, k, ll] == 1:
                    cont_passive1 += 1
                elif Pie_Passive[i, j, k, ll] == 0:
                    cont_passive0 += 1
                if Pie_Passive[i, j, k, ll] != -2 and Pie_Active[i, j, k, ll] != -2:
                    con_total += 1
                # if Pie_Active[i, j, k, ll] != -2 and Pie_Active_check[i, j, k, ll, 1] != 0:
                #     cont_active_action += 1
                # elif Pie_Active[i, j, k, ll] != -2 and Pie_Active_check[i, j, k, ll, 1] == 0:
                #     cont_active_noaction += 1

value_frequencies_active = {
    -2: cont_active_2,
    -1: cont_active_1,
    0: cont_active0,
    1: cont_active1
}
value_frequencies_action = {
    0: cont_active_action,
    1: cont_active_noaction
}
value_frequencies_passive = {
    -2: cont_passive_2,
    -1: cont_passive_1,
    0: cont_passive0,
    1: cont_passive1
}


labels_active = list(value_frequencies_active.keys())
sizes_active = list(value_frequencies_active.values())
labels_passive = list(value_frequencies_passive.keys())
sizes_passive = list(value_frequencies_passive.values())
labels_action = list(value_frequencies_action.keys())
sizes_action = list(value_frequencies_action.values())

labels_dict = {-2: 'Fail area', -1: '-5NM', 0: '0', 1: '5NM'}
labels_dict1 = {0: 'Multiple action', 1: 'Two action'}

# plt.figure(figsize=[4, 2], dpi=600)
# barWidth = 0.25
# br1 = np.arange(len(sizes_active))
# br2 = [x + barWidth for x in br1]
# plt.bar(br1, np.array(sizes_active) / np.sum(sizes_active)*100, color='b', width=barWidth, label='Active')
# plt.bar(br2, np.array(sizes_passive) / np.sum(sizes_passive)*100, color='r', width=barWidth, label='Passive')
# plt.xlabel('Action')
# plt.ylabel('Probability [%]')
# plt.xticks([r + barWidth for r in range(len(sizes_active))], ['Fail', '-5 Nm', '0', '5 Nm'])
# plt.legend()
# plt.tight_layout()
# plt.savefig("C:/Users/Admin/Desktop/IEEE/Figure7.svg")
# plt.savefig("C:/Users/Admin/Desktop/IEEE/Figure7.pdf")
# plt.savefig('temp.pdf')
print(1-((cont_active_1+cont_active1+cont_active0)/con))
print(1-((cont_passive_1+cont_passive1+cont_passive0)/con))
print(cont_active_1+cont_active1+cont_active0)
print(cont_passive_1+cont_passive1+cont_passive0)
print(con_total)