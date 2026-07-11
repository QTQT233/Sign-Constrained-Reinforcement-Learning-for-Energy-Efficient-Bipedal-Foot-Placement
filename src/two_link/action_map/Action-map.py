import h5py
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from numba import njit
import torch
import imageio
from matplotlib.patches import Rectangle

# N1 = 10
# N2 = 30
N1 = 10
N2 = 30
N = N1
speed_nondim = 3.5
tht1s = np.linspace(60, 120, N1) * np.pi / 180
# dtht1s = np.linspace(-speed_nondim, speed_nondim, N2)
dtht1s = np.linspace(-2, 2, N2)
tht2s = np.linspace(-60, 60, N1) * np.pi / 180
# dtht2s = np.linspace(-speed_nondim, speed_nondim, N2)
dtht2s = np.linspace(-2, 2, N2)

with h5py.File('C:/Users/Admin/Desktop/paper-code - F/working_save(-1,0,1)-10,30_from2-2', 'r') as h5f:
    data0 = h5f['working_save'][:]
working_save_active = data0
with h5py.File('C:/Users/Admin/Desktop/paper-code - F/working_save(-1,0)-10,30_from2-2', 'r') as h5f:
    data1 = h5f['working_save'][:]
working_save10 = data1
with h5py.File('C:/Users/Admin/Desktop/paper-code - F/working_save(0,1)-10,30_from2-2', 'r') as h5f:
    data2 = h5f['working_save'][:]
working_save01 = data2
# with h5py.File('C:/Users/Admin/Desktop/paper-code - F/check_save(-1,0,1)-10,30', 'r') as h5f:
#     data3 = h5f['check_save'][:]
# check_save = data3

for i in tqdm(range(N1)):
    for j in range(N2):
        for k in range(N1):
            for ll in range(N2):
                if working_save10[i, j, k, ll] == -1 and working_save01[i, j, k, ll] != 0:
                    working_save_passive[i, j, k, ll] = -1
                elif working_save10[i, j, k, ll] != 0 and working_save01[i, j, k, ll] == 1:
                    working_save_passive[i, j, k, ll] = 1
                elif working_save10[i, j, k, ll] == 0 or working_save01[i, j, k, ll] == 0:
                    working_save_passive[i, j, k, ll] = 0
                elif working_save10[i, j, k, ll] == -2 or working_save01[i, j, k, ll] == -2:
                    working_save_passive[i, j, k, ll] = -2

plt.figure(figsize=(N, N))
plt.axis('off')
# plt.title("Active", fontsize=30)
# plt.xlabel(r'$\theta_2$', fontsize=30)
# plt.ylabel(r'$\theta_1$', fontsize=30)
# plt.xlim(-np.pi/3, np.pi/3)  # 设置x轴范围
# plt.ylim(np.pi/3, 2*np.pi/3)
# plt.xticks([-np.pi/3, 0, np.pi/3], [r'$-{\pi}/3$', r'0', r'$\pi/3$'], fontsize=20)
# plt.yticks([np.pi/3, np.pi/2, 2*np.pi/3], [r'$\pi/3$', r'$\pi/2$', r'2*pi/3'], fontsize=20)
plt.xticks([])
plt.yticks([])
for ii in range(len(tht1s)):
    for jj in range(len(tht1s)):
        plt.subplot(N, N, ((N - 1) - ii) * N + jj + 1)
        plt.imshow(working_save_active[ii, :, jj, :], cmap='jet', origin='lower', vmin=-2, vmax=1,
                   extent=(float(dtht1s[0]), float(dtht1s[-1]), float(dtht2s[0]), float(dtht2s[-1])))
        plt.subplots_adjust(wspace=0.02, hspace=0.1)
        plt.xticks([])
        plt.yticks([])
        # if ii == (N - 1):
        #     plt.title(', '.join(map(str, [ii, jj])))
# plt.tight_layout()
# plt.savefig("C:/Users/Admin/Desktop/IEEE/Figure16.svg")
plt.show()
plt.pause(1e0)
# plt.savefig("Active.svg")

plt.figure(figsize=(N, N))
plt.axis('off')
# plt.title("Passive", fontsize=30)
# plt.xlabel(r'$\theta_2$', fontsize=30)
# plt.ylabel(r'$\theta_1$', fontsize=30)
# plt.xlim(-np.pi/3, np.pi/3)  # 设置x轴范围
# plt.ylim(np.pi/3, 2*np.pi/3)
# plt.xticks([-np.pi/3, 0, np.pi/3], [r'$-{\pi}/3$', r'0', r'$\pi/3$'], fontsize=20)
# plt.yticks([np.pi/3, np.pi/2, 2*np.pi/3], [r'$\pi/3$', r'$\pi/2$', r'2*pi/3'], fontsize=20)
plt.xticks([])
plt.yticks([])
for ii in range(len(tht1s)):
    for jj in range(len(tht1s)):
        plt.subplot(N, N, ((N - 1) - ii) * N + jj + 1)
        plt.imshow(working_save_passive[ii, :, jj, :], cmap='jet', origin='lower', vmin=-2, vmax=1,
                   extent=(float(dtht1s[0]), float(dtht1s[-1]), float(dtht2s[0]), float(dtht2s[-1])))
        plt.xticks([])
        plt.yticks([])
        plt.subplots_adjust(wspace=0.02, hspace=0.1)
        # if ii == (N - 1):
        #     plt.title(', '.join(map(str, [ii, jj])))
# plt.tight_layout()
# plt.savefig("C:/Users/Admin/Desktop/IEEE/Figure15.svg")
plt.show()
plt.pause(1e0)
# plt.savefig("Passive.svg")

# plt.figure(figsize=(N, N))
# # plt.title("check", fontsize=30)
# plt.xlabel(r'$\theta_2$', fontsize=30)
# plt.ylabel(r'$\theta_1$', fontsize=30)
# plt.xlim(-np.pi/3, np.pi/3)  # 设置x轴范围
# plt.ylim(np.pi/3, 2*np.pi/3)
# plt.xticks([-np.pi/3, 0, np.pi/3], [r'$-{\pi}/3$', r'0', r'$\pi/3$'], fontsize=20)
# plt.yticks([np.pi/3, np.pi/2, 2*np.pi/3], [r'$\pi/3$', r'$\pi/2$', r'2*pi/3'], fontsize=20)
# for ii in range(len(tht1s)):
#     for jj in range(len(tht1s)):
#         plt.subplot(N, N, ((N - 1) - ii) * N + jj + 1)
#         plt.imshow(check_save[ii, :, jj, :, 1], cmap='jet', origin='lower', vmin=-1, vmax=1,
#                    extent=(float(dtht1s[0]), float(dtht1s[-1]), float(dtht2s[0]), float(dtht2s[-1])))
#         plt.xticks([])
#         plt.yticks([])
#         # if ii == (N - 1):
#         #     plt.title(', '.join(map(str, [ii, jj])))
# plt.tight_layout()
# plt.show()
# plt.pause(1e0)
# plt.savefig("check.svg")

