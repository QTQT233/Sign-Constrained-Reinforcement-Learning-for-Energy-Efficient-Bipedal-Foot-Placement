import os
import torch
import random
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from numba import njit
import h5py
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

with h5py.File('D:/L&S/Mas/Project/Walking-RL/result/PPO-continuous/working_save_continuous(-1,0,1)-10,30', 'r') as h5f:
    data1 = h5f['working_save'][:]
working_save_continuous = data1

with h5py.File('C:/Users/Admin/Desktop/paper-code - F/working_save(-1,0,1)-10,30', 'r') as h5f:
    data0 = h5f['working_save'][:]
working_save_discrete = data0

N1 = 10
N2 = 30
N = N1
speed_nondim = 3.5
tht1s = np.linspace(60, 120, N1) * np.pi / 180
dtht1s = np.linspace(-speed_nondim, speed_nondim, N2)
tht2s = np.linspace(-60, 60, N1) * np.pi / 180
dtht2s = np.linspace(-speed_nondim, speed_nondim, N2)
plt.figure(figsize=(N, N))
plt.title("Continuous", fontsize=30)
for ii_ in range(len(tht1s)):
    for jj_ in range(len(tht1s)):
        plt.subplot(N, N, ((N - 1) - ii_) * N + jj_ + 1)
        plt.imshow(working_save_continuous[ii_, :, jj_, :], cmap='jet', origin='lower', vmin=-2, vmax=1,
                   extent=(float(dtht1s[0]), float(dtht1s[-1]), float(dtht2s[0]), float(dtht2s[-1])))
        plt.xticks([])
        plt.yticks([])
        # if ii_ == (N - 1):
        #     plt.title(', '.join(map(str, [ii_, jj_])))
plt.tight_layout()
plt.show()
plt.pause(1e0)


plt.figure(figsize=(N, N))
plt.title("Discrete", fontsize=30)
for ii_ in range(len(tht1s)):
    for jj_ in range(len(tht1s)):
        plt.subplot(N, N, ((N - 1) - ii_) * N + jj_ + 1)
        plt.imshow(working_save_discrete[ii_, :, jj_, :], cmap='jet', origin='lower', vmin=-2, vmax=1,
                   extent=(float(dtht1s[0]), float(dtht1s[-1]), float(dtht2s[0]), float(dtht2s[-1])))
        plt.xticks([])
        plt.yticks([])
        # if ii_ == (N - 1):
        #     plt.title(', '.join(map(str, [ii_, jj_])))
plt.tight_layout()
plt.show()
plt.pause(1e0)

# check1 = 0
# check2 = 0
# for i_ in tqdm(range(N1)):
#     for j in range(N2):
#         for k in range(N1):
#             for ll in range(N2):
#                 if working_save[i_, j, k, ll] != -2:
#                     check1 += 1
#                 if working_save_active[i_, j, k, ll] != -2:
#                     check2 += 1
# print(check1/90000)
# print(check2/90000)

# Code for simulation
# @njit
# def update_y(y_, temp_, dt_, c1_, c2_):
#     y_[1] += temp_[0][0] * dt_ - c1_ * y_[0]
#     y_[0] += y_[1] * dt_
#     y_[3] += temp_[1][0] * dt_ - c2_ * y_[2]
#     y_[2] += y_[3] * dt_
#     return y_
#
#
# @njit
# def calc_new_a_b(y_, m1_, m2_, l1_, l2_, l1_1, l2_1, g_):
#     a_ = np.array([[(m1_ + m2_) * l1_1 ** 2, m2_ * l1_1 * l2_1 * np.sin(y_[0] - y_[2])],
#                    [m2_ * l1_1 * l2_1 * np.sin(y_[0] - y_[2]), m2_ * l2_1 ** 2]])
#     b = np.array([[-m2_ * l1_1 * l2_1 * y_[3] * np.cos(y_[0] - y_[2]) * (y_[1] - y_[3])
#                    + m2_ * l1_1 * l2_1 * y_[1] * y_[3] * np.cos(y_[0] - y_[2])
#                    - m1_ * g_ * l1_1 * np.cos(y_[0]) - m2_ * g_ * l1_ * np.cos(y_[0])],
#                   [-m2_ * l1_1 * l2_1 * y_[1] * np.cos(y_[0] - y_[2]) * (y_[1] - y_[3])
#                    - m2_ * l1_1 * l2_1 * y_[1] * y_[3] * np.cos(y_[0] - y_[2])
#                    - m2_ * g_ * l2_1 * np.sin(y_[2])]])
#     identity_a = np.eye(a_.shape[0])
#     inverse_a = np.linalg.solve(a_, identity_a)
#     return inverse_a, b
#
#
# settle = np.deg2rad(5)  # the acceptable error for the target location
# m1 = 1.4122
# m2 = 0.0839
# l1 = 0.22
# l2 = 0.31
# l1_r = 0.33
# l2_r = 0.33
# c1 = 0  # 0.1
# c2 = 0  # 0.95
# gamma = 1
# gamma1 = 1
# g = 9.8
# dt = 0.01
# torque = 5
# simu_time = 3.2
# steps = int(simu_time / dt)
# # target location
# target1 = 60
# target2 = 30
# target3 = 120
# target4 = -30
# theta1_target1 = 60 * np.pi / 180
# theta1_target2 = 120 * np.pi / 180
# theta2_target1 = 30 * np.pi / 180
# theta2_target2 = -30 * np.pi / 180
# # episode and training parameters
# episode = 50000
# critic_training_times = 20
# critic_training_steps = 50
# actor_training_times = 10
# playing_times = 2000
# concentrated_sample_times = 15
# batch_size = 5000
# reward_scale = 5
# action_value_weight = 0.04
# policy_entropy_coefficient = 0.001
#
# # Terminate conditions
# theta1_range = 90
# theta2_range = 90
# rad_theta1_range = np.deg2rad(theta1_range)
# rad_theta2_range = np.deg2rad(theta2_range)
# speed_range = 3.5
# # ========================================================================================================================
# device = torch.device("cpu")
# nnn = 512
#
#
# # Neural Networks initialize
# class Model(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.s = torch.nn.Sequential(
#             torch.nn.Linear(4, nnn * 2),
#             torch.nn.ReLU(),
#             torch.nn.Linear(nnn * 2, nnn),
#             torch.nn.ReLU(),
#         )
#         self.mu = torch.nn.Sequential(
#             torch.nn.Linear(nnn, 1),
#             torch.nn.ReLU(),
#         )
#         self.sigma = torch.nn.Sequential(
#             torch.nn.Linear(nnn, 1),
#             torch.nn.ReLU(),
#         )
#
#     def forward(self, state):
#         state = self.s(state)
#
#         return self.mu(state), self.sigma(state).exp()
#
#
# # Neural Networks initialize
# model = torch.nn.Sequential(torch.nn.Linear(4, nnn * 2), torch.nn.ReLU(),
#                             torch.nn.Linear(nnn * 2, nnn), torch.nn.ReLU(),
#                             torch.nn.Linear(nnn, 1))
# for module in model.modules():
#     if isinstance(module, torch.nn.Linear):
#         torch.nn.init.orthogonal_(module.weight)
# policy = Model()
# model.to(device)
# policy.to(device)
# optimizer_value = torch.optim.Adam(model.parameters(), lr=1e-3)
# optimizer_policy = torch.optim.Adam(policy.parameters(), lr=1e-3)
# loss_fn = torch.nn.MSELoss()
#
#
# class PendulumEnv:
#     def __init__(self, g_=g, m1_=m1, l1_=l1_r, m2_=m2, l2_=l2_r, c1_=c1, c2_=c2, dt_=dt, torque_=torque,
#                  l1_1=l1, l2_1=l2):
#         self.state = None
#         self.next_state = None
#         self.max_torque = torque_
#         self.dt = dt_
#         self.g = g_
#         self.m1 = m1_
#         self.m2 = m2_
#         self.l1 = l1_
#         self.l2 = l2_
#         self.l1_s = l1_1
#         self.l2_s = l2_1
#         self.c1 = c1_
#         self.c2 = c2_
#         self.reward = None
#         self.over = None
#         self.y = np.zeros(4)
#
#     def step(self, act_index):
#         self.y[0], self.y[1], self.y[2], self.y[3] = self.state
#         action_value = act_index
#         a = np.array([[(self.m1 + self.m2) * self.l1_s ** 2,
#                        self.m2 * self.l1_s * self.l2_s * np.sin(self.y[0] - self.y[2])],
#                       [self.m2 * self.l1_s * self.l2_s * np.sin(self.y[0] - self.y[2]), self.m2 * self.l2_s ** 2]])
#         b = np.array(
#             [[-self.m2 * self.l1_s * self.l2_s * self.y[3] * np.cos(self.y[0] - self.y[2]) * (self.y[1] - self.y[3])
#               + self.m2 * self.l1_s * self.l2_s * self.y[1] * self.y[3] * np.cos(self.y[0] - self.y[2])
#               - self.m1 * self.g * self.l1_s * np.cos(self.y[0]) - self.m2 * self.g * self.l1 * np.cos(self.y[0])],
#              [-self.m2 * self.l1_s * self.l2_s * self.y[1] * np.cos(self.y[0] - self.y[2]) * (self.y[1] - self.y[3])
#               - self.m2 * self.l1_s * self.l2_s * self.y[1] * self.y[3] * np.cos(self.y[0] - self.y[2])
#               - self.m2 * self.g * self.l2_s * np.sin(self.y[2])]])
#         inverse_a = np.linalg.inv(a)
#         arr = np.array([[-action_value], [action_value]])
#         temp = inverse_a @ (b + arr)
#         self.y[1] += temp[0][0] * self.dt - self.c1 * self.y[0]
#         self.y[0] += self.y[1] * self.dt
#         self.y[3] += temp[1][0] * self.dt - self.c2 * self.y[2]
#         self.y[2] += self.y[3] * self.dt
#         reward1 = abs(self.y[0] - np.deg2rad(target1)) < settle and abs(self.y[2] - np.deg2rad(target2)) < settle
#         reward2 = abs(self.y[0] - np.deg2rad(target3)) < settle and abs(self.y[2] - np.deg2rad(target4)) < settle
#         if reward1 or reward2:
#             self.reward = reward_scale
#             success.append(1)
#             self.over = True
#         elif abs(self.y[0] - np.pi / 2) > rad_theta1_range or abs(self.y[2]) > rad_theta2_range:
#             self.reward = -reward_scale
#             self.over = True
#         else:
#             self.reward = 0
#             self.over = False
#         self.reward += -abs(action_value) * action_value_weight
#         self.next_state = np.array([self.y[0], self.y[1], self.y[2], self.y[3]])
#         self.state = np.copy(self.next_state)
#         return self.next_state, self.reward, self.over
#
#     def reset(self):
#         theta_1_initial = np.deg2rad(np.random.uniform(60, 120))
#         theta_2_initial = np.deg2rad(np.random.uniform(-60, 60))  # rads
#         theta_1_dot_initial = np.random.uniform(-speed_range, speed_range)
#         theta_2_dot_initial = np.random.uniform(-speed_range, speed_range)
#         self.state = np.array([theta_1_initial, theta_1_dot_initial, theta_2_initial, theta_2_dot_initial])
#         return self.state
#
#     def define(self, theta1, dtheta1, theta2, dtheta2):
#         self.state = np.array([theta1, dtheta1, theta2, dtheta2])
#         return self.state
#
#
# env = PendulumEnv()
#
# policy.load_state_dict(
#     torch.load('Policy_Net_Pytorch(-1,0,1)_1443.pth'))
# policy.to(device)
# N1 = 10
# N2 = 30
# N = N1
# speed_nondim = 3.5
# tht1s = np.linspace(60, 120, N1) * np.pi / 180
# dtht1s = np.linspace(-speed_nondim, speed_nondim, N2)
# tht2s = np.linspace(-60, 60, N1) * np.pi / 180
# dtht2s = np.linspace(-speed_nondim, speed_nondim, N2)
# working_save = np.zeros((N1, N2, N1, N2))
# for i_ in tqdm(range(N1)):
#     for j in range(N2):
#         for k in range(N1):
#             for ll in range(N2):
#                 working_save[i_, j, k, ll] = -2
#                 y = np.array([tht1s[i_], dtht1s[j], tht2s[k], dtht2s[ll]])
#                 flag = 0
#                 con = 0
#                 a_save = 0
#                 for step in range(steps):
#                     if (abs(y[0] - target1) < settle and abs(y[2] - target2) < settle) or \
#                             (abs(y[0] - target3) < settle and abs(y[2] - target4) < settle):
#                         working_save[i_, j, k, ll] = a_save
#                         # print("1")
#                         flag = 1
#                         break
#                     inverse_A, B = calc_new_a_b(y, m1, m2, l1_r, l2_r, l1, l2, g)
#                     state_in_net_ = np.array([(y[0] - np.pi / 2) / rad_theta1_range,
#                                               y[1] / speed_range, y[2] / rad_theta2_range, y[3] / speed_range])
#                     mu, sigma = policy(torch.FloatTensor(state_in_net_).reshape(1, 4).to(device))
#                     a = random.normalvariate(mu=mu.item(), sigma=sigma.item())
#                     if con == 0:
#                         a_save = a
#                         con += 1
#                     arr = np.array([[-a], [a]])
#                     temp = np.dot(inverse_A, B + arr)
#                     y = update_y(y, temp, dt, c1, c2)
#                     if y[0] < 0 or y[0] > np.pi or y[2] < -np.pi / 2 or y[2] > np.pi / 2:
#                         break
#                     if (abs(y[0] - theta1_target1) < settle and abs(y[2] - theta2_target1) < settle) or \
#                             (abs(y[0] - theta1_target2) < settle and abs(y[2] - theta2_target2) < settle):
#                         # print("1")
#                         working_save[i_, j, k, ll] = a_save
#                         flag = 1
#                         break

# with h5py.File('working_save_continuous(-1,0,1)-10,30', 'w') as h5f:
#     h5f.create_dataset('working_save', data=working_save)

