import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import imageio
import torch
import numpy as np
from tqdm import tqdm
import random
import os
import matplotlib.pyplot as plt
from numba import njit
import h5py
from pathlib import Path
import sys

_ENERGY_COMPARISON_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ENERGY_COMPARISON_ROOT))
from cmt_metrics import positive_actuator_work_increment


# Preserve the archived location as the default while allowing portable reruns.
ENERGY_COMPARISON_DATA_ROOT = os.environ.get(
    "ENERGY_COMPARISON_DATA_ROOT",
    "D:/L&S/Mas/Project/Paper1/Energy_Comparison",
).rstrip("/\\")


@njit
def update_y(y_, temp_, dt_, c1_, c2_):
    y_[1] += temp_[0][0] * dt_ - c1_ * y_[0]
    y_[0] += y_[1] * dt_
    y_[3] += temp_[1][0] * dt_ - c2_ * y_[2]
    y_[2] += y_[3] * dt_
    return y_


@njit
def calc_new_a_b(y_, m1_, m2_, j1_, j2_, l1_, l2_, l1_1, l2_1, g_):
    a_ = np.array([[(j1_ + m2_ * l1_ ** 2), m2_ * l1_ * l2_1 * np.sin(y_[2] - y_[0])],
                   [m2_ * l1_ * l2_1 * np.sin(y_[2] - y_[0]), (j2_ + m2_ * l2_1 ** 2)]])
    b = np.array([[- m2_ * y_[1] * l1_ * y_[3] * l2_1 * np.cos(y_[2] - y_[0])
                   - m1_ * g_ * l1_1 * np.cos(y_[0]) - m2_ * g_ * l1_ * np.cos(y_[0])
                   - m2_ * l1_ * y_[3] * l2_1 * np.cos(y_[2] - y_[0]) * (y_[3] - y_[1])],
                  [m2_ * y_[1] * l1_ * y_[3] * l2_1 * np.cos(y_[2] - y_[0])
                   - m2_ * g_ * l2_1 * np.sin(y_[2])
                   - m2_ * l1_ * y_[1] * l2_1 * np.cos(y_[2] - y_[0]) * (y_[3] - y_[1])]])
    identity_a = np.eye(a_.shape[0])
    inverse_a = np.linalg.solve(a_, identity_a)
    return inverse_a, b


device = torch.device("cpu")
nnn = 512
policy_passive_discrete = torch.nn.Sequential(torch.nn.Linear(4, nnn * 2), torch.nn.Tanh(),
                                              torch.nn.Linear(nnn * 2, nnn), torch.nn.Tanh(),
                                              torch.nn.Linear(nnn, 2), torch.nn.Softmax(dim=1))
policy_passive_discrete_1 = torch.nn.Sequential(torch.nn.Linear(4, nnn * 2), torch.nn.Tanh(),
                                                torch.nn.Linear(nnn * 2, nnn), torch.nn.Tanh(),
                                                torch.nn.Linear(nnn, 2), torch.nn.Softmax(dim=1))
policy_active_discrete = torch.nn.Sequential(torch.nn.Linear(4, nnn * 2), torch.nn.Tanh(),
                                             torch.nn.Linear(nnn * 2, nnn), torch.nn.Tanh(),
                                             torch.nn.Linear(nnn, 3), torch.nn.Softmax(dim=1))
policy_active_continuous = torch.nn.Sequential(
    torch.nn.Linear(4, nnn * 2), torch.nn.Tanh(),
    torch.nn.Linear(nnn * 2, nnn), torch.nn.Tanh(),
    torch.nn.Linear(nnn, 2)  # 输出均值和标准差
)

policy_active_continuous.load_state_dict(
    torch.load('D:/L&S/Mas/Project/Paper1/Check_continuous/'
               'Policy_Net_Pytorch(-1,0,1)_1560_continuous_old_form.pth'))
policy_active_continuous.to(device)
policy_passive_discrete.load_state_dict(torch.load
                                        ('C:/Users/Admin/Desktop/paper-code - F/Policy_Net_Pytorch(1,0)_900.pth'))
policy_passive_discrete.to(device)
policy_passive_discrete_1.load_state_dict(torch.load
                                          ('C:/Users/Admin/Desktop/paper-code - F/Policy_Net_Pytorch(-1,0)_1046.pth'))
policy_passive_discrete_1.to(device)
policy_active_discrete.load_state_dict(torch.load
                                       ('D:/L&S/Mas/Project/Paper1/Check_continuous'
                                        '/Policy_Net_Pytorch(-1,0,1)_1635.pth'))
# policy_active_discrete.load_state_dict(torch.load
#                                        ('C:/Users/Admin/Desktop/paper-code - F'
#                                         '/Policy_Net_Pytorch(-1,0,1).pth'))
policy_active_discrete.to(device)

# 系统参数
settle = np.deg2rad(5)  # the acceptable error for the target location
m1 = 1.4122
m2 = 0.0839
l1 = 0.22
l2 = 0.31
l1_r = 0.33
l2_r = 0.31
c1 = 0  # 0.1
c2 = 0  # 0.95
j1 = m1 * (l1 ** 2)
j2 = m2 * (l2 ** 2)
gamma = 1
gamma1 = 1
g = 9.8
dt = 0.01
torque = 4
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
success = []
actions1 = [0, -1]
actions = [-1, 0, 1]

N1 = 10
N2 = 30
N = N1
speed_nondim = 3.5
tht1s = np.linspace(60, 120, N1) * np.pi / 180
dtht1s = np.linspace(-speed_nondim, speed_nondim, N2)
# dtht1s = np.linspace(-2, 2, N2)
tht2s = np.linspace(-60, 60, N1) * np.pi / 180
dtht2s = np.linspace(-speed_nondim, speed_nondim, N2)
# dtht2s = np.linspace(-2, 2, N2)
working_save01 = np.zeros((N1, N2, N1, N2))
Energy_save01 = np.zeros((N1, N2, N1, N2))
Cmt_save01 = np.zeros((N1, N2, N1, N2))
for i_ in tqdm(range(N1)):
    for j in range(N2):
        for k in range(N1):
            for ll in range(N2):
                working_save01[i_, j, k, ll] = -2
                y = np.array([tht1s[i_], dtht1s[j], tht2s[k], dtht2s[ll]])
                initial_theta1 = y[0]
                initial_theta2 = y[2]
                flag = 0
                con = 0
                a_save = 0
                for step in range(steps):
                    if (abs(y[0] - theta1_target1) < settle and abs(y[2] - theta2_target1) < settle) or \
                            (abs(y[0] - theta1_target2) < settle and abs(y[2] - theta2_target2) < settle):
                        working_save01[i_, j, k, ll] = a_save
                        final_theta1 = y[0]
                        final_theta2 = y[2]
                        flag = 1
                        initial_center_x = (m1 * l1 * np.cos(initial_theta1) +
                                            m2 * (l1_r * np.cos(initial_theta1) + l2 * np.sin(initial_theta2))) / (
                                                       m1 + m2)
                        final_center_x = (m1 * l1 * np.cos(final_theta1) +
                                          m2 * (l1_r * np.cos(final_theta1) + l2 * np.sin(final_theta2))) / (m1 + m2)
                        D = abs(final_center_x - initial_center_x)
                        W = (m1 + m2) * g
                        E = Energy_save01[i_, j, k, ll]
                        Cmt_save01[i_, j, k, ll] = E / (W * D) if (W * D) > 0 else 0
                        break
                    inverse_A, B = calc_new_a_b(y, m1, m2, j1, j2, l1_r, l2_r, l1, l2, g)
                    state_in_net__ = np.array([(y[0] - np.pi / 2) / rad_theta1_range,
                                               y[1] / speed_range, y[2] / rad_theta2_range, y[3] / speed_range])
                    prob = policy_passive_discrete(torch.FloatTensor(state_in_net__).reshape(1, 4))[
                        0].cpu().detach().numpy()
                    a = np.argmax(prob)
                    if a * (y[3]-y[1]) > 0:
                        Energy_save01[i_, j, k, ll] += abs(a * torque) * abs(y[3]-y[1]) * dt
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
                        working_save01[i_, j, k, ll] = a_save
                        final_theta1 = y[0]
                        final_theta2 = y[2]
                        flag = 1
                        initial_center_x = (m1 * l1 * np.cos(initial_theta1) +
                                            m2 * (l1_r * np.cos(initial_theta1) + l2 * np.sin(initial_theta2))) / (
                                                       m1 + m2)
                        final_center_x = (m1 * l1 * np.cos(final_theta1) +
                                          m2 * (l1_r * np.cos(final_theta1) + l2 * np.sin(final_theta2))) / (m1 + m2)
                        D = abs(final_center_x - initial_center_x)
                        W = (m1 + m2) * g
                        E = Energy_save01[i_, j, k, ll]
                        Cmt_save01[i_, j, k, ll] = E / (W * D) if (W * D) > 0 else 0
                        break
# with h5py.File(f'{ENERGY_COMPARISON_DATA_ROOT}/60,30(0.33m)/working_save(0,1)-10-30', 'w') as h5f:
#     h5f.create_dataset('working_save', data=working_save01)
# with h5py.File(f'{ENERGY_COMPARISON_DATA_ROOT}/60,30(0.33m)/Cmt_save(0,1)-10-30', 'w') as h5f:
#     h5f.create_dataset('Cmt_save', data=Cmt_save01)
# with h5py.File(f'{ENERGY_COMPARISON_DATA_ROOT}/60,30(0.33m)/Energy_save(0,1)-10-30', 'w') as h5f:
#     h5f.create_dataset('Energy_save', data=Energy_save01)

N1 = 10
N2 = 30
N = N1
a_old = 0
speed_nondim = 3.5
chek_save = np.zeros(4)
tht1s = np.linspace(60, 120, N1) * np.pi / 180
dtht1s = np.linspace(-speed_nondim, speed_nondim, N2)
tht2s = np.linspace(-60, 60, N1) * np.pi / 180
dtht2s = np.linspace(-speed_nondim, speed_nondim, N2)
working_save0_1 = np.zeros((N1, N2, N1, N2))
Energy_save0_1 = np.zeros((N1, N2, N1, N2))
Cmt_save0_1 = np.zeros((N1, N2, N1, N2))
for i_ in tqdm(range(N1)):
    for j in range(N2):
        for k in range(N1):
            for ll in range(N2):
                working_save0_1[i_, j, k, ll] = -2
                y = np.array([tht1s[i_], dtht1s[j], tht2s[k], dtht2s[ll]])
                initial_theta1 = y[0]
                initial_theta2 = y[2]
                flag = 0
                con = 0
                a_save = 0
                for step in range(steps):
                    if (abs(y[0] - target1) < settle and abs(y[2] - target2) < settle) or \
                            (abs(y[0] - target3) < settle and abs(y[2] - target4) < settle):
                        working_save0_1[i_, j, k, ll] = a_save
                        flag = 1
                        final_theta1 = y[0]
                        final_theta2 = y[2]
                        initial_center_x = (m1 * l1 * np.cos(initial_theta1) +
                                            m2 * (l1_r * np.cos(initial_theta1) + l2 * np.sin(initial_theta2))) / (
                                                   m1 + m2)
                        final_center_x = (m1 * l1 * np.cos(final_theta1) +
                                          m2 * (l1_r * np.cos(final_theta1) + l2 * np.sin(final_theta2))) / (m1 + m2)
                        D = abs(final_center_x - initial_center_x)
                        W = (m1 + m2) * g
                        E = Energy_save0_1[i_, j, k, ll]
                        Cmt_save0_1[i_, j, k, ll] = E / (W * D) if (W * D) > 0 else 0
                        break
                    inverse_A, B = calc_new_a_b(y, m1, m2, j1, j2, l1_r, l2_r, l1, l2, g)
                    state_in_net__ = np.array([(y[0] - np.pi / 2) / rad_theta1_range,
                                               y[1] / speed_range, y[2] / rad_theta2_range, y[3] / speed_range])
                    prob = policy_passive_discrete_1(torch.FloatTensor(state_in_net__).reshape(1, 4))[
                        0].cpu().detach().numpy()
                    a = actions1[np.argmax(prob)]
                    if a != a_old and a_old != 0 and a != 0:
                        chek_save[0] = i_
                        chek_save[1] = j
                        chek_save[2] = k
                        chek_save[3] = ll
                    a_old = a
                    if a * (y[3]-y[1]) > 0:
                        Energy_save0_1[i_, j, k, ll] += abs(a * torque) * abs(y[3]-y[1]) * dt
                    # Energy_save0_1[i_, j, k, ll] += abs(a * torque) * (abs(y[1]) + abs(y[3])) * dt
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
                        working_save0_1[i_, j, k, ll] = a_save
                        flag = 1
                        final_theta1 = y[0]
                        final_theta2 = y[2]
                        initial_center_x = (m1 * l1 * np.cos(initial_theta1) +
                                            m2 * (l1_r * np.cos(initial_theta1) + l2 * np.sin(initial_theta2))) / (
                                                   m1 + m2)
                        final_center_x = (m1 * l1 * np.cos(final_theta1) +
                                          m2 * (l1_r * np.cos(final_theta1) + l2 * np.sin(final_theta2))) / (m1 + m2)
                        D = abs(final_center_x - initial_center_x)
                        W = (m1 + m2) * g
                        E = Energy_save0_1[i_, j, k, ll]
                        Cmt_save0_1[i_, j, k, ll] = E / (W * D) if (W * D) > 0 else 0
                        break
with h5py.File(f'{ENERGY_COMPARISON_DATA_ROOT}/action_weight=0.04/60,30(0.33m)/working_save(-1,0)-10-30', 'w') as h5f:
    h5f.create_dataset('working_save', data=working_save0_1)
with h5py.File(f'{ENERGY_COMPARISON_DATA_ROOT}/action_weight=0.04/60,30(0.33m)/Cmt_save(-1,0)-10-30', 'w') as h5f:
    h5f.create_dataset('Cmt_save', data=Cmt_save0_1)
with h5py.File(f'{ENERGY_COMPARISON_DATA_ROOT}/action_weight=0.04/60,30(0.33m)/Energy_save(-1,0)-10-30', 'w') as h5f:
    h5f.create_dataset('Energy_save', data=Energy_save0_1)

working_save_passive = np.zeros((N1, N2, N1, N2))
Cmt_save_passive = np.zeros((N1, N2, N1, N2))
for i in tqdm(range(N1)):
    for j in range(N2):
        for k in range(N1):
            for ll in range(N2):
                if working_save0_1[i, j, k, ll] == -1 and working_save01[i, j, k, ll] != 0:
                    working_save_passive[i, j, k, ll] = -1
                elif working_save0_1[i, j, k, ll] != 0 and working_save01[i, j, k, ll] == 1:
                    working_save_passive[i, j, k, ll] = 1
                elif working_save0_1[i, j, k, ll] == 0 or working_save01[i, j, k, ll] == 0:
                    working_save_passive[i, j, k, ll] = 0
                elif working_save0_1[i, j, k, ll] == -1 and working_save01[i, j, k, ll] == 1:
                    if Cmt_save01[i, j, k, ll] > Cmt_save0_1[i, j, k, ll]:
                        working_save_passive[i, j, k, ll] = -1
                    else:
                        working_save_passive[i, j, k, ll] = 1
                elif working_save0_1[i, j, k, ll] == -2 and working_save01[i, j, k, ll] == -2:
                    working_save_passive[i, j, k, ll] = -2
for i in tqdm(range(N1)):
    for j in range(N2):
        for k in range(N1):
            for ll in range(N2):
                if working_save_passive[i, j, k, ll] == -1:
                    Cmt_save_passive[i, j, k, ll] = Cmt_save0_1[i, j, k, ll]
                elif working_save_passive[i, j, k, ll] == 1:
                    Cmt_save_passive[i, j, k, ll] = Cmt_save01[i, j, k, ll]
                elif working_save_passive[i, j, k, ll] == 0:
                    Cmt_save_passive[i, j, k, ll] = min(Cmt_save01[i, j, k, ll], Cmt_save0_1[i, j, k, ll])
                else:
                    Cmt_save_passive[i, j, k, ll] = -2
with h5py.File(f'{ENERGY_COMPARISON_DATA_ROOT}/action_weight=0.04/60,30(0.33m)/working_save_passive-10-30', 'w') as h5f:
    h5f.create_dataset('working_save_passive', data=working_save_passive)
with h5py.File(f'{ENERGY_COMPARISON_DATA_ROOT}/action_weight=0.04/60,30(0.33m)/Cmt_save_passive-10-30', 'w') as h5f:
    h5f.create_dataset('Cmt_save_passive', data=Cmt_save_passive)

N1 = 10
N2 = 30
N = N1
speed_nondim = 3.5
tht1s = np.linspace(60, 120, N1) * np.pi / 180
dtht1s = np.linspace(-speed_nondim, speed_nondim, N2)
tht2s = np.linspace(-60, 60, N1) * np.pi / 180
dtht2s = np.linspace(-speed_nondim, speed_nondim, N2)
working_save_active_discrete = np.zeros((N1, N2, N1, N2))
Energy_save_active_discrete = np.zeros((N1, N2, N1, N2))
Cmt_save_active_discrete = np.zeros((N1, N2, N1, N2))
for i_ in tqdm(range(N1)):
    for j in range(N2):
        for k in range(N1):
            for ll in range(N2):
                working_save_active_discrete[i_, j, k, ll] = -2
                y = np.array([tht1s[i_], dtht1s[j], tht2s[k], dtht2s[ll]])
                initial_theta1 = y[0]
                initial_theta2 = y[2]
                flag = 0
                con = 0
                a_save = 0
                for step in range(steps):
                    if (abs(y[0] - target1) < settle and abs(y[2] - target2) < settle) or \
                            (abs(y[0] - target3) < settle and abs(y[2] - target4) < settle):
                        working_save_active_discrete[i_, j, k, ll] = a_save
                        flag = 1
                        final_theta1 = y[0]
                        final_theta2 = y[2]
                        initial_center_x = (m1 * l1 * np.cos(initial_theta1) +
                                            m2 * (l1_r * np.cos(initial_theta1) + l2 * np.sin(initial_theta2))) / (
                                                   m1 + m2)
                        final_center_x = (m1 * l1 * np.cos(final_theta1) +
                                          m2 * (l1_r * np.cos(final_theta1) + l2 * np.sin(final_theta2))) / (m1 + m2)
                        D = abs(final_center_x - initial_center_x)
                        W = (m1 + m2) * g
                        E = Energy_save_active_discrete[i_, j, k, ll]
                        Cmt_save_active_discrete[i_, j, k, ll] = E / (W * D) if (W * D) > 0 else 0
                        break
                    inverse_A, B = calc_new_a_b(y, m1, m2, j1, j2, l1_r, l2_r, l1, l2, g)
                    state_in_net_ = np.array([(y[0] - np.pi / 2) / rad_theta1_range,
                                              y[1] / speed_range, y[2] / rad_theta2_range, y[3] / speed_range])
                    prob = policy_active_discrete(torch.FloatTensor(state_in_net_).reshape(1, 4))[
                        0].cpu().detach().numpy()
                    a = actions[np.argmax(prob)]
                    if a * (y[3]-y[1]) > 0:
                        Energy_save_active_discrete[i_, j, k, ll] += abs(a * torque) * abs(y[3]-y[1]) * dt
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
                        # print("1")
                        working_save_active_discrete[i_, j, k, ll] = a_save
                        flag = 1
                        final_theta1 = y[0]
                        final_theta2 = y[2]
                        initial_center_x = (m1 * l1 * np.cos(initial_theta1) +
                                            m2 * (l1_r * np.cos(initial_theta1) + l2 * np.sin(initial_theta2))) / (
                                                   m1 + m2)
                        final_center_x = (m1 * l1 * np.cos(final_theta1) +
                                          m2 * (l1_r * np.cos(final_theta1) + l2 * np.sin(final_theta2))) / (m1 + m2)
                        D = abs(final_center_x - initial_center_x)
                        W = (m1 + m2) * g
                        E = Energy_save_active_discrete[i_, j, k, ll]
                        Cmt_save_active_discrete[i_, j, k, ll] = E / (W * D) if (W * D) > 0 else 0
                        break
# with h5py.File(f'{ENERGY_COMPARISON_DATA_ROOT}/60,30(0.33m)/working_save_active_discrete-10-30', 'w') as h5f:
#     h5f.create_dataset('working_save_active_discrete', data=working_save_active_discrete)
# with h5py.File(f'{ENERGY_COMPARISON_DATA_ROOT}/60,30(0.33m)/Energy_save_active_discrete-10-30', 'w') as h5f:
#     h5f.create_dataset('Energy_save_active_discrete', data=Energy_save_active_discrete)
# with h5py.File(f'{ENERGY_COMPARISON_DATA_ROOT}/60,30(0.33m)/Cmt_save_active_discrete(0,1)-10-30', 'w') as h5f:
#     h5f.create_dataset('Cmt_save', data=Cmt_save_active_discrete)

N1 = 10
N2 = 30
N = N1
speed_nondim = 3.5
tht1s = np.linspace(60, 120, N1) * np.pi / 180
dtht1s = np.linspace(-speed_nondim, speed_nondim, N2)
tht2s = np.linspace(-60, 60, N1) * np.pi / 180
dtht2s = np.linspace(-speed_nondim, speed_nondim, N2)
working_save_active_continuous = np.zeros((N1, N2, N1, N2))
Energy_save_active_continuous = np.zeros((N1, N2, N1, N2))
Cmt_save_active_continuous = np.zeros((N1, N2, N1, N2))
for i_ in tqdm(range(N1)):
    for j in range(N2):
        for k in range(N1):
            for ll in range(N2):
                working_save_active_continuous[i_, j, k, ll] = -6
                y = np.array([tht1s[i_], dtht1s[j], tht2s[k], dtht2s[ll]])
                initial_theta1 = y[0]
                initial_theta2 = y[2]
                con = 0
                a_save = 0
                for step in range(steps):
                    if (abs(y[0] - target1) < settle and abs(y[2] - target2) < settle) or \
                            (abs(y[0] - target3) < settle and abs(y[2] - target4) < settle):
                        working_save_active_continuous[i_, j, k, ll] = a_save
                        flag = 1
                        final_theta1 = y[0]
                        final_theta2 = y[2]
                        initial_center_x = (m1 * l1 * np.cos(initial_theta1) +
                                            m2 * (l1_r * np.cos(initial_theta1) + l2 * np.sin(initial_theta2))) / (
                                                   m1 + m2)
                        final_center_x = (m1 * l1 * np.cos(final_theta1) +
                                          m2 * (l1_r * np.cos(final_theta1) + l2 * np.sin(final_theta2))) / (m1 + m2)
                        D = abs(final_center_x - initial_center_x)
                        W = (m1 + m2) * g
                        E = Energy_save_active_continuous[i_, j, k, ll]
                        Cmt_save_active_continuous[i_, j, k, ll] = E / (W * D) if (W * D) > 0 else 0
                        break
                    inverse_A, B = calc_new_a_b(y, m1, m2, j1, j2, l1_r, l2_r, l1, l2, g)
                    state_in_net_ = np.array([(y[0] - np.pi / 2) / rad_theta1_range,
                                              y[1] / speed_range, y[2] / rad_theta2_range, y[3] / speed_range])
                    output = policy_active_continuous(torch.FloatTensor(state_in_net_).reshape(1, 4).to(device))
                    mean, log_std = output.chunk(2, dim=1)
                    std = torch.exp(log_std)
                    dist = torch.distributions.Normal(mean, std)
                    action_value = dist.sample().item()
                    action_value = np.clip(action_value, -torque, torque)
                    Energy_save_active_continuous[i_, j, k, ll] += positive_actuator_work_increment(
                        action_value, y[3] - y[1], dt
                    )
                    if con == 0:
                        a_save = action_value
                        con += 1
                    arr = np.array([[-action_value], [action_value]])
                    temp = np.dot(inverse_A, B + arr)
                    y = update_y(y, temp, dt, c1, c2)
                    if y[0] < 0 or y[0] > np.pi or y[2] < -np.pi / 2 or y[2] > np.pi / 2:
                        break
                    if (abs(y[0] - theta1_target1) < settle and abs(y[2] - theta2_target1) < settle) or \
                            (abs(y[0] - theta1_target2) < settle and abs(y[2] - theta2_target2) < settle):
                        # print("1")
                        working_save_active_continuous[i_, j, k, ll] = a_save
                        flag = 1
                        final_theta1 = y[0]
                        final_theta2 = y[2]
                        initial_center_x = (m1 * l1 * np.cos(initial_theta1) +
                                            m2 * (l1_r * np.cos(initial_theta1) + l2 * np.sin(initial_theta2))) / (
                                                   m1 + m2)
                        final_center_x = (m1 * l1 * np.cos(final_theta1) +
                                          m2 * (l1_r * np.cos(final_theta1) + l2 * np.sin(final_theta2))) / (m1 + m2)
                        D = abs(final_center_x - initial_center_x)
                        W = (m1 + m2) * g
                        E = Energy_save_active_continuous[i_, j, k, ll]
                        Cmt_save_active_continuous[i_, j, k, ll] = E / (W * D) if (W * D) > 0 else 0
                        break

# with h5py.File(f'{ENERGY_COMPARISON_DATA_ROOT}/60,30(0.33m)/working_save_active_continuous-10-30', 'w') as h5f:
#     h5f.create_dataset('working_save_active_continuous', data=working_save_active_continuous)
# with h5py.File(f'{ENERGY_COMPARISON_DATA_ROOT}/60,30(0.33m)/Energy_save_active_continuous-10-30', 'w') as h5f:
#     h5f.create_dataset('Energy_save_active_continuous', data=Energy_save_active_continuous)
# with h5py.File(f'{ENERGY_COMPARISON_DATA_ROOT}/60,30(0.33m)/Cmt_save_active_continuous-10-30', 'w') as h5f:
#     h5f.create_dataset('Cmt_save', data=Cmt_save_active_continuous)

Passive_energy = 0
Active_energy_discrete = 0
Active_energy_continuous = 0
count = 0
Passive = 0
Active_discrete = 0
Active_continuous = 0
error = 0
error1 = 0
error2 = 0
error_old = 0
error_old_1 = 0
error_save = np.zeros(4)
for i_ in tqdm(range(N1)):
    for j in range(N2):
        for k in range(N1):
            for ll in range(N2):
                if working_save_passive[i_, j, k, ll] != -2:
                    Passive += 1
                if working_save_active_discrete[i_, j, k, ll] != -2:
                    Active_discrete += 1
                if working_save_active_continuous[i_, j, k, ll] != -6:
                    Active_continuous += 1
                if working_save_passive[i_, j, k, ll] != -2 and working_save_active_discrete[i_, j, k, ll] != -2 and \
                        working_save_active_continuous[i_, j, k, ll] != -6:
                    error1 = Cmt_save_active_discrete[i_, j, k, ll] - Cmt_save_passive[i_, j, k, ll]
                    error2 = Cmt_save_active_continuous[i_, j, k, ll] - Cmt_save_passive[i_, j, k, ll]
                    error = error1 + error2
                    if error1 > 0 and error1 > error_old_1 and 5 < i_ < 35 \
                            and 5 < j < 45 and 5 < k < 35 and 5 < ll < 45:
                        # if i_ == 6 and j == 15 and k == 9 and ll == 7:
                        #     continue
                        # elif i_ == 6 and j == 9 and k == 9 and ll == 15:
                        #     continue
                        # elif i_ == 6 and j == 13 and k == 9 and ll == 18:
                        #     continue
                        # else:
                        error_save[0] = i_
                        error_save[1] = j
                        error_save[2] = k
                        error_save[3] = ll
                        error_old = error
                        error_old_1 = error1
                    count += 1
                    Passive_energy += Cmt_save_passive[i_, j, k, ll]
                    Active_energy_discrete += Cmt_save_active_discrete[i_, j, k, ll]
                    Active_energy_continuous += Cmt_save_active_continuous[i_, j, k, ll]

print("Passive_energy")
print(Passive_energy / count)
# print(Passive / (10 * 30 * 10 * 30))
print("Active_energy_discrete")
print(Active_energy_discrete / count)
# print(Active_discrete / (10 * 30 * 10 * 30))
print("Active_energy_continuous")
print(Active_energy_continuous / count)
# print(Active_continuous / (10 * 30 * 10 * 30))
print(error_save)
