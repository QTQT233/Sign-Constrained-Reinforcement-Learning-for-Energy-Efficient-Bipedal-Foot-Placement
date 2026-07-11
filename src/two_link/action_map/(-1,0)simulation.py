import imageio
import torch
import numpy as np
from tqdm import tqdm
import random
import os
import matplotlib.pyplot as plt
from numba import njit
import h5py


@njit
def update_y(y_, temp_, dt_, c1_, c2_):
    y_[1] += temp_[0][0] * dt_ - c1_ * y_[0]
    y_[0] += y_[1] * dt_
    y_[3] += temp_[1][0] * dt_ - c2_ * y_[2]
    y_[2] += y_[3] * dt_
    return y_


@njit
def calc_new_a_b(y_, m1_, m2_, l1_, l2_, l1_1, l2_1, g_):
    a_ = np.array([[(m1_ * l1_1 ** 2 + m2_ * l1_ ** 2), m2_ * l1_ * l2_1 * np.sin(y_[2] - y_[0])],
                   [m2_ * l1 * l2_1 * np.sin(y_[2] - y_[0]), (m2_ * l2_1 ** 2 + m2_ * l2_1 ** 2)]])
    b = np.array([[-m2_ * l1 * l2_1 * y_[3] * np.cos(y_[2] - y_[0]) * (y_[3] - y_[1])
                   - m2_ * l1_1 * l2_1 * y_[1] * y_[3] * np.cos(y_[2] - y_[0])
                   - m1_ * g_ * l1_1 * np.cos(y_[0]) - m2_ * g_ * l1_ * np.cos(y_[0])],
                  [-m2_ * l1_1 * l2_1 * y_[1] * np.cos(y_[2] - y_[0]) * (y_[3] - y_[1])
                   + m2_ * l1 * l2_1 * y_[1] * y_[3] * np.cos(y_[2] - y_[0])
                   - m2_ * g_ * l2_1 * np.sin(y_[2])]])
    identity_a = np.eye(a_.shape[0])
    inverse_a = np.linalg.solve(a_, identity_a)
    return inverse_a, b


device = torch.device("cpu")
nnn = 512
policy = torch.nn.Sequential(torch.nn.Linear(4, nnn * 2), torch.nn.Tanh(),
                             torch.nn.Linear(nnn * 2, nnn), torch.nn.Tanh(),
                             torch.nn.Linear(nnn, 2), torch.nn.Softmax(dim=1))
policy.load_state_dict(torch.load('C:/Users/Admin/Desktop/paper-code - F/Policy_Net_Pytorch(-1,0)_1046.pth'))
policy.to(device)
# 系统参数
settle = np.deg2rad(5)  # the acceptable error for the target location
m1 = 1.4122
m2 = 0.0839
l1 = 0.22
l2 = 0.31
l1_r = 0.33
l2_r = 0.33
c1 = 0  # 0.1
c2 = 0  # 0.95
gamma = 1
gamma1 = 1
g = 9.8
dt = 0.01
torque = 5
simu_time = 3.2
steps = int(simu_time / dt)
# target location
target1 = 60
target2 = 30
target3 = 120
target4 = -30
theta1_target1 = 60 * np.pi / 180
theta1_target2 = 120 * np.pi / 180
theta2_target1 = 30 * np.pi / 180
theta2_target2 = -30 * np.pi / 180

# Terminate conditions
theta1_range = 90
theta2_range = 90
rad_theta1_range = np.deg2rad(theta1_range)
rad_theta2_range = np.deg2rad(theta2_range)
speed_range = 3.5
actions = [0, -1]

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
working_save = np.zeros((N1, N2, N1, N2))
for i_ in tqdm(range(N1)):
    for j in range(N2):
        for k in range(N1):
            for ll in range(N2):
                working_save[i_, j, k, ll] = -2
                y = np.array([tht1s[i_], dtht1s[j], tht2s[k], dtht2s[ll]])
                flag = 0
                con = 0
                a_save = 0
                for step in range(steps):
                    if (abs(y[0] - target1) < settle and abs(y[2] - target2) < settle) or \
                            (abs(y[0] - target3) < settle and abs(y[2] - target4) < settle):
                        working_save[i_, j, k, ll] = a_save
                        flag = 1
                        break
                    inverse_A, B = calc_new_a_b(y, m1, m2, l1_r, l2_r, l1, l2, g)
                    state_in_net__ = np.array([(y[0] - np.pi / 2) / rad_theta1_range,
                                               y[1] / speed_range, y[2] / rad_theta2_range, y[3] / speed_range])
                    prob = policy(torch.FloatTensor(state_in_net__).reshape(1, 4))[0].cpu().detach().numpy()
                    a = actions[np.argmax(prob)]
                    if con == 0:
                        a_save = a
                        con += 1
                    arr = np.array([[-a * torque], [a * torque]])
                    temp = np.dot(inverse_A, B + arr)
                    y = update_y(y, temp, dt, c1, c2)
                    if y[0] < 0 or y[0] > np.pi or y[2] < -np.pi / 2 or y[2] > np.pi / 2:
                        break
                    if (abs(y[0] - theta1_target1) < settle and abs(y[2] - theta2_target1) < settle) or \
                            (abs(y[0] - theta1_target2) < settle and abs(y[2] - theta2_target2) < settle):
                        working_save[i_, j, k, ll] = a_save
                        flag = 1
                        break
with h5py.File('C:/Users/Admin/Desktop/paper-code - F/working_save(-1,0)-10,30_from2-2', 'w') as h5f:
    h5f.create_dataset('working_save', data=working_save)

# plt.figure(figsize=(N, N))
# # plt.title("Passive", fontsize=30)
# plt.xticks([])
# plt.yticks([])
# for ii in range(len(tht1s)):
#     for jj in range(len(tht1s)):
#         plt.subplot(N, N, ((N - 1) - ii) * N + jj + 1)
#         plt.imshow(working_save[ii, :, jj, :], cmap='jet', origin='lower', vmin=-2, vmax=1,
#                    extent=(float(dtht1s[0]), float(dtht1s[-1]), float(dtht2s[0]), float(dtht2s[-1])))
#         plt.xticks([])
#         plt.yticks([])
#         # if ii == (N - 1):
#         #     plt.title(', '.join(map(str, [ii, jj])))
# plt.tight_layout()
# plt.show()
# # plt.title()
# plt.pause(1e0)
