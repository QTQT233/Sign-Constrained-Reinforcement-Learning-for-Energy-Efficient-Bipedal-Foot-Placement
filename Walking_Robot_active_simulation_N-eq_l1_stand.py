import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
import random
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from numba import njit
import h5py


@njit
def update_y(y_, temp_, dt_, c1_, c2_):
    y_[1] += temp_[0][0] * dt_ - c1_ * y_[1]
    y_[0] += y_[1] * dt_
    y_[3] += temp_[1][0] * dt_ - c2_ * y_[3]
    y_[2] += y_[3] * dt_
    return y_


@njit
def calc_new_a_b(y_, m1_, m2_, j1, j2, l1_r, l2_r, l1_1, l2_1, g_):
    a_ = np.array([[(j1 + m2_ * l1_r ** 2), m2_ * l1_r * l2_1 * np.sin(y_[2] - y_[0])],
                   [m2_ * l1_r * l2_1 * np.sin(y_[2] - y_[0]), (j2 + m2_ * l2_1 ** 2)]])
    b = np.array([[- m2_ * y_[1] * l1_r * y_[3] * l2_1 * np.cos(y_[2] - y_[0])
                   - m1_ * g_ * l1_1 * np.cos(y_[0]) - m2_ * g_ * l1_r * np.cos(y_[0])
                   - m2_ * l1_r * y_[3] * l2_1 * np.cos(y_[2] - y_[0]) * (y_[3] - y_[1])],
                  [m2_ * y_[1] * l1_r * y_[3] * l2_1 * np.cos(y_[2] - y_[0])
                   - m2_ * g_ * l2_1 * np.sin(y_[2])
                   - m2_ * l1_r * y_[1] * l2_1 * np.cos(y_[2] - y_[0]) * (y_[3] - y_[1])]])
    identity_a = np.eye(a_.shape[0])
    inverse_a = np.linalg.solve(a_, identity_a)
    return inverse_a, b


settle = np.deg2rad(5)  # the acceptable error for the target location
m1 = 1.5981
m2 = 0.6503
l1 = 0.521
l2 = 0.481
l1_prime = 0.4114
l2_prime = 0.218
J1 = 0.5
J2 = 0.105
c1 = 0.00047630386389081107
c2 = 0.0001
# m1 = 0.6503
# m2 = 1.5981
# l1 = 0.521
# l2 = 0.481
# l1_prime = 0.30300000000000005
# l2_prime = 0.05916
# J1 = 0.34
# J2 = 0.1
# c1 = 0.001
# c2 = 0.0001
gamma = 1
gamma1 = 1
g = 9.8
dt = 0.01
torque = 4
simu_time = 4.2
steps = int(simu_time / dt)

# Terminate conditions
target1 = 60
target2 = 30
target3 = 120
target4 = -30
theta1_range = 90
theta2_range = 90
rad_theta1_range = np.deg2rad(theta1_range)
rad_theta2_range = np.deg2rad(theta2_range)
speed_range = 2

action = [-torque, 0, torque]
actions = [-1, 0, 1]

# ========================================================================================================================
device = torch.device("cpu")
nnn = 256
policy = torch.nn.Sequential(torch.nn.Linear(4, nnn * 2), torch.nn.Tanh(),
                             torch.nn.Linear(nnn * 2, nnn), torch.nn.Tanh(),
                             torch.nn.Linear(nnn, 3), torch.nn.Softmax(dim=1))
policy.to(device)
optimizer_policy = torch.optim.Adam(policy.parameters(), lr=1e-3)
loss_fn = torch.nn.MSELoss()

policy.load_state_dict(
    torch.load('D:/L&S/Mas/Project/Paper2/l1_stand(-1,0,1)_1417.pth'))
policy.to(device)


N1 = 30
N2 = 60
N = N1
connt=0
speed_nondim = 2
tht1s = np.linspace(60, 120, N1) * np.pi / 180
dtht1s = np.linspace(-speed_nondim, speed_nondim, N2)
tht2s = np.linspace(-60, 60, N1) * np.pi / 180
dtht2s = np.linspace(-speed_nondim, speed_nondim, N2)
working_save_active = np.zeros((N1, N2, N1, N2))
check_save = np.zeros((N1, N2, N1, N2, 5))
for i_ in tqdm(range(N1)):
    for j in range(N2):
        for k in range(N1):
            for ll in range(N2):
                # if working_save_active[i_, j, k, ll] == 1 or working_save_active[i_, j, k, ll] == 0 or working_save_active[i_, j, k, ll] == -1:
                #     print("")
                #     print(i_)
                working_save_active[i_, j, k, ll] = -2
                check_save[i_, j, k, ll, 3] = -2
                y = np.array([tht1s[i_], dtht1s[j], tht2s[k], dtht2s[ll]])
                flag = 0
                con = 0
                a_save = 0
                for step in range(steps):
                    reward1 = abs(y[0] - np.deg2rad(target1)) < settle and abs(
                        y[2] - np.deg2rad(target2)) < settle
                    reward2 = abs(y[0] - np.deg2rad(target3)) < settle and abs(
                        y[2] - np.deg2rad(target4)) < settle
                    if reward1 or reward2:
                        working_save_active[i_, j, k, ll] = a_save
                        break
                    inverse_A, B = calc_new_a_b(y, m1, m2, J1, J2, l1, l2, l1_prime, l2_prime, g)
                    state_in_net_ = np.array([(y[0] - np.pi / 2) / rad_theta1_range,
                                              y[1] / speed_range, y[2] / rad_theta2_range, y[3] / speed_range])
                    prob = policy(torch.FloatTensor(state_in_net_).reshape(1, 4))[0].cpu().detach().numpy()
                    a = actions[np.argmax(prob)]
                    # if a != 0 and check_save[i_, j, k, ll, 0] == 0:
                    #     check_save[i_, j, k, ll, 0] = a
                    # if a != 0 and check_save[i_, j, k, ll, 0] != a:
                    #     check_save[i_, j, k, ll, 1] = 1
                    #     check_save[i_, j, k, ll, 2] += 1
                    if con == 0:
                        a_save = a
                        con += 1
                    arr = np.array([[-a * torque], [a * torque]])
                    temp = np.dot(inverse_A, B + arr)
                    y = update_y(y, temp, dt, c1, c2)
                    if y[0] < 0 or y[0] > np.pi or y[2] < -np.pi / 2 or y[2] > np.pi / 2:
                        break
                    reward1 = abs(y[0] - np.deg2rad(target1)) < settle and abs(
                        y[2] - np.deg2rad(target2)) < settle
                    reward2 = abs(y[0] - np.deg2rad(target3)) < settle and abs(
                        y[2] - np.deg2rad(target4)) < settle
                    if reward1 or reward2:
                        working_save_active[i_, j, k, ll] = a_save
                        break


with h5py.File('D:/L&S/Mas/Project/Paper2/working_l1_stand_save_active(-1,0,1)-30,60', 'w') as h5f:
    h5f.create_dataset('working_save_active', data=working_save_active)
# with h5py.File('/check_save(-1,0,1)-10,30', 'w') as h5f:
#     h5f.create_dataset('check_save', data=check_save)
# print(check)

# plt.figure(figsize=(8, 8))
# plt.xlabel(r'$\theta_2$', fontsize=20)
# plt.ylabel(r'$\theta_1$', fontsize=20)
# plt.xlim(-np.pi / 3, np.pi / 3)  # 设置x轴范围
# plt.ylim(np.pi / 3, np.pi * 2 / 3)
# plt.xticks([-np.pi / 3, 0, np.pi / 3], [r'$-{\pi}/3$', r'0', r'$-{\pi}/3$'], fontsize=15)
# plt.yticks([np.pi * 2 / 3, np.pi / 2, np.pi / 3], [r'$\pi*2/3$', r'$-{\pi}/2$', r'$-{\pi}/3$'], fontsize=15)
# plt.tight_layout()
# plt.title("Passive", fontsize=30)
# for ii_ in range(len(tht1s)):
#     for jj_ in range(len(tht1s)):
#         plt.subplot(N, N, ((N - 1) - ii_) * N + jj_ + 1)
#         plt.imshow(working_save_active[ii_, :, jj_, :], cmap='jet', origin='lower', vmin=-2, vmax=1,
#                    extent=(float(dtht1s[0]), float(dtht1s[-1]), float(dtht2s[0]), float(dtht2s[-1])))
#         plt.xticks([])
#         plt.yticks([])
# plt.tight_layout()
# plt.show()
# plt.pause(1e0)

