import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pandas as pd
import h5py
from matplotlib.patches import Rectangle
import h5py
import imageio
import torch
from tqdm import tqdm
import random
from numba import njit
import os

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Running on the GPU")
else:
    device = torch.device("cpu")
    print("Running on the CPU")
torch.cuda.set_device(device)
nnn = 512


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
                                        '/Policy_Net_Pytorch(-1,0,1)_1612.pth'))
policy_active_discrete.to(device)

m1 = 1.4122
m2 = 0.0839
l1 = 0.22
l2 = 0.31
l1_r = 0.33
l2_r = 0.31
c1 = 0  # 0.1
c2 = 0  # 0.95
j1 = m1 * l1 ** 2
j2 = m2 * l2 ** 2
gamma = 1
gamma1 = 1
g = 9.8
simu_time = 3.2
dt_plot = 0.005
steps = int(simu_time / dt_plot)
dt = 0.005
torque = 4
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
settle = np.deg2rad(5)

N1 = 10
N2 = 30
N = N1
speed_nondim = 3.5
tht1s = np.linspace(60, 120, N1) * np.pi / 180
dtht1s = np.linspace(-speed_nondim, speed_nondim, N2)
tht2s = np.linspace(-60, 60, N1) * np.pi / 180
dtht2s = np.linspace(-speed_nondim, speed_nondim, N2)


def set_state(y, tht1s_, dtht1s_, tht2s_, dtht2s_):
    y[0] = tht1s_[6]  # theta1
    y[1] = dtht1s_[16]  # dtheta1
    y[2] = tht2s_[9]  # theta2
    y[3] = dtht2s_[6]
    return y


def calculate_energy(y, m1__, m2__, l1__, l2__, l1_r_, g__):
    """计算系统的总机械能"""
    theta1_std = np.pi / 2 - y[0]  # 您的θ₁转换为标准θ₁
    theta2_std = y[2]  # 您的θ₂与标准θ₂一致
    dtheta1 = y[1]
    dtheta2 = y[3]

    # 计算转动惯量
    j1 = m1__ * l1__ ** 2
    j2 = m2__ * l2__ ** 2

    # 动能
    v1_sq = (l1__ * dtheta1) ** 2
    v2_sq = (l1__ * dtheta1) ** 2 + (l2__ * dtheta2) ** 2 + 2 * l1__ * l2__ * dtheta1 * dtheta2 * np.cos(
        theta1_std - theta2_std)

    ke = 0.5 * m1__ * v1_sq + 0.5 * m2__ * v2_sq

    # 势能
    h1 = l1__ * np.sin(y[0])
    h2 = l1_r_ * np.sin(y[0]) - l2__ * np.cos(y[2])

    pe = m1__ * g__ * h1 + m2__ * g * h2

    return ke + pe


def load_rl_trajectory():
    with h5py.File('D:/L&S/Mas/Project/Paper1/Energy/working_save_passive-10-30', 'r') as h5f:
        working_save_passive = np.array(h5f['working_save_passive'][:])
    # with h5py.File('D:/L&S/Mas/Project/Paper1/Energy/working_save_passive', 'r') as h5f:
    #     working_save_passive = np.array(h5f['working_save_passive'][:])
    y = np.zeros(4)
    # y[0] = 88.2 * np.pi / 180  # theta1
    # y[1] = -19.98 * np.pi / 180  # dtheta1
    # y[2] = 0 * np.pi / 180  # theta2
    # y[3] = 0 * np.pi / 180  # dtheta2
    y = set_state(y, tht1s, dtht1s, tht2s, dtht2s)

    # 存储结果
    y_trace = np.zeros((steps + 1, 4))
    y_trace[0, :] = y
    a_save = np.zeros((steps + 1, 1))
    e_save = np.zeros(steps + 1)
    total_energy = np.zeros(steps + 1)
    ddtheta2_save = np.zeros(steps + 1)
    i = np.argmin(abs(y[0] - tht1s))
    j = np.argmin(abs(y[1] - dtht1s))
    k = np.argmin(abs(y[2] - tht2s))
    ll = np.argmin(abs(y[3] - dtht2s))
    if working_save_passive[i, j, k, ll] == 1 or working_save_passive[i, j, k, ll] == 0:
        print("1")
        for step in range(steps):
            if (abs(y[0] - target1) < settle and abs(y[2] - target2) < settle) or \
                    (abs(y[0] - target3) < settle and abs(y[2] - target4) < settle):
                working_save0_1[i_, j, k, ll] = a_save
                break
            inverse_A, B = calc_new_a_b(y, m1, m2, j1, j2, l1_r, l2_r, l1, l2, g)
            state_in_net__ = np.array([(y[0] - np.pi / 2) / rad_theta1_range,
                                       y[1] / speed_range, y[2] / rad_theta2_range, y[3] / speed_range])
            prob = policy_passive_discrete(torch.FloatTensor(state_in_net__).reshape(1, 4))[0].cpu().detach().numpy()
            a = np.argmax(prob)
            arr = np.array([[-a * torque], [a * torque]])
            temp = np.dot(inverse_A, B + arr)
            y = update_y(y, temp, dt, c1, c2)
            ddtheta2_save[step + 1] = temp[1] * dt_plot
            y_trace[step + 1, :] = y
            a_save[step + 1, 0] = a
            e_save[step + 1] = e_save[step] + abs(a) * torque * (abs(y[1]) + abs(y[3])) * dt_plot

            if (abs(y[0] - np.pi / 3) < 0.08 and abs(y[2] - np.pi / 6) < 0.08) \
                    or (abs(y[0] - 2 * np.pi / 3) < 0.08 and abs(y[2] - (-np.pi / 6)) < 0.08):
                print(f'\nAchieved the goal at step {step}!')
                break
    else:
        for step in range(steps):
            if (abs(y[0] - target1) < settle and abs(y[2] - target2) < settle) or \
                    (abs(y[0] - target3) < settle and abs(y[2] - target4) < settle):
                break
            inverse_A, B = calc_new_a_b(y, m1, m2, j1, j2, l1_r, l2_r, l1, l2, g)
            state_in_net__ = np.array([(y[0] - np.pi / 2) / rad_theta1_range,
                                       y[1] / speed_range, y[2] / rad_theta2_range, y[3] / speed_range])
            prob = policy_passive_discrete_1(torch.FloatTensor(state_in_net__).reshape(1, 4))[0].cpu().detach().numpy()
            a = actions1[np.argmax(prob)]
            arr = np.array([[-a * torque], [a * torque]])
            temp = np.dot(inverse_A, B + arr)
            y = update_y(y, temp, dt, c1, c2)
            ddtheta2_save[step + 1] = temp[1] * dt_plot
            y_trace[step + 1, :] = y
            a_save[step + 1, 0] = a
            e_save[step + 1] = e_save[step] + abs(a) * torque * (abs(y[1]) + abs(y[3])) * dt_plot

            if (abs(y[0] - theta1_target1) < settle and abs(y[2] - theta2_target1) < settle) or \
                    (abs(y[0] - theta1_target2) < settle and abs(y[2] - theta2_target2) < settle):
                print(f'\nAchieved the goal at step {step}!')
                break
    t = np.arange(0, (step + 1) * dt_plot, dt_plot)
    for i in range(steps):
        total_energy[i + 1] = calculate_energy(y_trace[i], m1, m2, l1, l2, l1_r, g)
    return t, y_trace[:step + 1, :], a_save[:step + 1, 0], e_save[:step + 1], ddtheta2_save[:step + 1]\
        , total_energy[:step + 1]


def load_rl_active_trajectory():
    # 初始化状态（与原始代码相同）
    y = np.zeros(4)
    # y[0] = 88.2 * np.pi / 180  # theta1
    # y[1] = -19.98 * np.pi / 180  # dtheta1
    # y[2] = 0 * np.pi / 180  # theta2
    # y[3] = 0 * np.pi / 180  # dtheta2
    # y[0] = tht1s[9]  # theta1
    # y[1] = dtht1s[8]  # dtheta1
    # y[2] = tht2s[6]  # theta2
    # y[3] = dtht2s[9]  # dtheta2
    y = set_state(y, tht1s, dtht1s, tht2s, dtht2s)
    # 存储结果
    y_trace = np.zeros((steps + 1, 4))
    y_trace[0, :] = y
    a_save = np.zeros(steps + 1)
    e_save = np.zeros(steps + 1)
    ddtheta2_save = np.zeros(steps + 1)
    total_energy = np.zeros(steps + 1)
    # 主仿真循环 - 使用连续动作策略
    for step in range(steps):
        # 归一化状态
        if (abs(y[0] - target1) < settle and abs(y[2] - target2) < settle) or \
                (abs(y[0] - target3) < settle and abs(y[2] - target4) < settle):
            working_save_active_discrete[i_, j, k, ll] = a_save
            break
        inverse_A, B = calc_new_a_b(y, m1, m2, j1, j2, l1_r, l2_r, l1, l2, g)
        state_in_net_ = np.array([(y[0] - np.pi / 2) / rad_theta1_range,
                                  y[1] / speed_range, y[2] / rad_theta2_range, y[3] / speed_range])
        prob = policy_active_discrete(torch.FloatTensor(state_in_net_).reshape(1, 4))[0].cpu().detach().numpy()
        a = actions[np.argmax(prob)]
        arr = np.array([[-a * torque], [a * torque]])
        temp = np.dot(inverse_A, B + arr)
        y = update_y(y, temp, dt, c1, c2)

        # 存储结果
        ddtheta2_save[step + 1] = temp[1] * dt_plot
        y_trace[step + 1, :] = y
        a_save[step + 1] = a
        e_save[step + 1] = e_save[step] + abs(a) * torque * (abs(y[1]) + abs(y[3])) * dt_plot

        # 检查终止条件
        if (abs(y[0] - theta1_target1) < settle and abs(y[2] - theta2_target1) < settle) or \
                (abs(y[0] - theta1_target2) < settle and abs(y[2] - theta2_target2) < settle):
            print(f'\nAchieved the goal at step {step}!')
            break
    t = np.arange(0, (step + 1) * dt_plot, dt_plot)
    for i in range(steps):
        total_energy[i + 1] = calculate_energy(y_trace[i], m1, m2, l1, l2, l1_r, g)
    return t, y_trace[:step + 1, :], a_save[:step + 1], e_save[:step + 1], ddtheta2_save[:step + 1]\
        , total_energy[:step + 1]


# 加载原始强化学习仿真的状态轨迹
def load_rl_continuous_trajectory():
    y = np.zeros(4)
    # y[0] = 88.2 * np.pi / 180  # theta1
    # y[1] = -19.98 * np.pi / 180  # dtheta1
    # y[2] = 0 * np.pi / 180  # theta2
    # y[3] = 0 * np.pi / 180  # dtheta2
    # y[0] = tht1s[9]  # theta1
    # y[1] = dtht1s[8]  # dtheta1
    # y[2] = tht2s[6]  # theta2
    # y[3] = dtht2s[9]  # dtheta2
    y = set_state(y, tht1s, dtht1s, tht2s, dtht2s)
    y_trace = np.zeros((steps + 1, 4))
    y_trace[0, :] = y
    a_save = np.zeros(steps + 1)
    e_save = np.zeros(steps + 1)
    ddtheta2_save = np.zeros(steps + 1)
    total_energy = np.zeros(steps + 1)
    for step in range(steps):
        if (abs(y[0] - target1) < settle and abs(y[2] - target2) < settle) or \
                (abs(y[0] - target3) < settle and abs(y[2] - target4) < settle):
            working_save_active_continuous[i_, j, k, ll] = a_save
            break
        inverse_a, b = calc_new_a_b(y, m1, m2, j1, j2, l1_r, l2_r, l1, l2, g)
        state_in_net_ = np.array([(y[0] - np.pi / 2) / rad_theta1_range,
                                  y[1] / speed_range, y[2] / rad_theta2_range, y[3] / speed_range])
        output = policy_active_continuous(torch.FloatTensor(state_in_net_).reshape(1, 4).to(device))
        mean, log_std = output.chunk(2, dim=1)
        std = torch.exp(log_std)
        dist = torch.distributions.Normal(mean, std)
        action_value = dist.sample().item()
        action_value = np.clip(action_value, -torque, torque)
        a = np.clip(action_value, -1.0, 1.0)
        arr = np.array([[-a * torque], [a * torque]])
        temp = np.dot(inverse_a, b + arr)
        y = update_y(y, temp, dt, c1, c2)
        ddtheta2_save[step + 1] = temp[1] * dt_plot
        y_trace[step + 1, :] = y
        a_save[step + 1] = a
        e_save[step + 1] = e_save[step] + abs(a) * torque * (abs(y[1]) + abs(y[3])) * dt_plot
        if (abs(y[0] - theta1_target1) < settle and abs(y[2] - theta2_target1) < settle) or \
                (abs(y[0] - theta1_target2) < settle and abs(y[2] - theta2_target2) < settle):
            print(f'\nAchieved the goal at step {step}!')
            break

    t = np.arange(0, (step + 1) * dt_plot, dt_plot)
    for i in range(steps):
        total_energy[i + 1] = calculate_energy(y_trace[i], m1, m2, l1, l2, l1_r, g)
    return t, y_trace[:step + 1, :], a_save[:step + 1], e_save[:step + 1], ddtheta2_save[:step + 1]\
        , total_energy[:step + 1]


# PID控制器类（用于跟踪RL轨迹）
class TrajectoryTrackingPID:
    def __init__(self, Kp, Ki, Kd, max_output, min_output, dt):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.max_output = max_output
        self.min_output = min_output
        self.dt = dt
        self.reset()

    def reset(self):
        self.integral = 0
        self.previous_error = 0

    def compute(self, target_theta1, target_theta2, current_theta1, current_theta2):
        # 计算角度误差
        # error1 = target_theta1 - current_theta1
        error2 = target_theta2 - current_theta2

        # 组合误差（主要关注θ1，因为θ1的变化较大）
        # combined_error = 0.4 * error1 + 0.6 * error2
        combined_error = error2

        # 比例项
        P = self.Kp * combined_error

        # 积分项
        # self.integral += combined_error * self.dt
        # I = self.Ki * self.integral

        # 微分项
        derivative = (combined_error - self.previous_error) / self.dt
        D = self.Kd * derivative

        # 计算总输出
        # output = P + I + D
        output = P + D

        # 限幅
        output = np.clip(output, self.min_output, self.max_output)
        self.previous_error = combined_error

        return output


# 主函数
def main():
    # 加载RL轨迹
    rl_time_continuous, rl_trajectory_continuous, rl_actions_continuous, rl_energy_continuous, \
        rl_ddtheta2_continuous, rl_total_energy_continuous = load_rl_continuous_trajectory()
    rl_time, rl_trajectory, rl_actions, rl_energy, rl_ddtheta2, rl_total_energy = load_rl_trajectory()
    rl_time_active, rl_trajectory_active, rl_actions_active, rl_energy_active, \
        rl_ddtheta2_active, rl_total_energy_active = load_rl_active_trajectory()
    steps = min(len(rl_time_continuous), len(rl_time), len(rl_time_active))

    y = np.zeros(4)
    # y[0] = 88.2 * np.pi / 180  # theta1
    # y[1] = -19.98 * np.pi / 180  # dtheta1
    # y[2] = 0 * np.pi / 180  # theta2
    # y[3] = 0 * np.pi / 180  # dtheta2
    # y[0] = tht1s[9]  # theta1
    # y[1] = dtht1s[8]  # dtheta1
    # y[2] = tht2s[6]  # theta2
    # y[3] = dtht2s[9]
    y = set_state(y, tht1s, dtht1s, tht2s, dtht2s)

    # 创建PID控制器（用于跟踪RL轨迹）
    pid = TrajectoryTrackingPID(Kp=80.0, Ki=0.05, Kd=2,
                                max_output=1.0, min_output=-1.0, dt=dt_plot)

    # 存储结果
    pid_trace = np.zeros((steps + 1, 4))
    pid_trace[0, :] = y
    pid_actions = np.zeros(steps + 1)
    pid_energy = np.zeros(steps + 1)
    # 主仿真循环
    for step in range(steps):
        target_theta1 = rl_trajectory_continuous[step, 0]
        target_theta2 = rl_trajectory_continuous[step, 2]

        a = pid.compute(target_theta1, target_theta2, y[0], y[2])

        inverse_a, b = calc_new_a_b(y, m1, m2, j1, j2, l1_r, l2_r, l1, l2, g)

        arr = np.array([[-a * torque], [a * torque]])
        temp = np.dot(inverse_a, b + arr)

        y[1] += temp[0] * dt_plot - c1 * y[1]
        y[0] += y[1] * dt_plot
        y[3] += temp[1] * dt_plot - c2 * y[3]
        y[2] += y[3] * dt_plot

        pid_trace[step + 1, :] = y
        pid_actions[step] = a
        if step >= 1:
            pid_energy[step] = pid_energy[step - 1] + abs(a) * torque * (abs(y[1]) + abs(y[3])) * dt_plot
        else:
            pid_energy[step] = abs(a) * torque * (y[1] + y[3])
        if step % 50 == 0:
            print(f"Step: {step}/{steps}, Theta1: {y[0] * 180 / np.pi:.2f}°, Theta2: {y[2] * 180 / np.pi:.2f}°")
    total_energy_pid = np.zeros(steps + 1)
    for i in range(steps):
        total_energy_pid[i] = calculate_energy(pid_trace[i], m1, m2, l1, l2, l1_r, g)
    rl_time_continuous = rl_time_continuous[:steps]
    rl_trajectory_continuous = rl_trajectory_continuous[:steps, :]
    rl_actions_continuous = rl_actions_continuous[:steps]
    rl_energy_continuous = rl_energy_continuous[:steps]
    rl_total_energy_continuous = rl_total_energy_continuous[:steps]
    rl_ddtheta2_continuous = rl_ddtheta2_continuous[:steps]

    rl_time = rl_time[:steps]
    rl_trajectory = rl_trajectory[:steps, :]
    rl_actions = rl_actions[:steps]
    rl_energy = rl_energy[:steps]
    rl_total_energy = rl_total_energy[:steps]
    rl_ddtheta2 = rl_ddtheta2[:steps]

    rl_time_active = rl_time_active[:steps]
    rl_trajectory_active = rl_trajectory_active[:steps, :]
    rl_actions_active = rl_actions_active[:steps]
    rl_energy_active = rl_energy_active[:steps]
    rl_total_energy_active = rl_total_energy_active[:steps]
    rl_ddtheta2_active = rl_ddtheta2_active[:steps]

    rl__total_energy__active = rl_time_active[:steps]
    rl_trajectory_active = rl_trajectory_active[:steps, :]
    rl_actions_active = rl_actions_active[:steps]
    rl_energy_active = rl_energy_active[:steps]

    # 绘制结果比较
    plt.figure(figsize=(12, 8))

    # θ1比较
    plt.subplot(221)
    plt.plot(rl_time_continuous, rl_trajectory_continuous[:, 0] * 180 / np.pi, 'b-', label='RL_continuous Theta1')
    plt.plot(rl_time_continuous, pid_trace[1:steps + 1, 0] * 180 / np.pi, 'r--', label='PID Theta1')
    plt.plot(rl_time, rl_trajectory[:, 0] * 180 / np.pi, 'y--', label='RL_passive_discrete Theta1')
    plt.plot(rl_time_active, rl_trajectory_active[:, 0] * 180 / np.pi, 'k--', label='RL_active_discrete Theta1')
    plt.ylabel(r'$\theta_1$ [deg]')
    plt.legend()
    plt.title('Theta1 Comparison')

    # θ2比较
    plt.subplot(222)
    plt.plot(rl_time_continuous, rl_trajectory_continuous[:, 2] * 180 / np.pi, 'b-', label='RL_continuous Theta2')
    plt.plot(rl_time_continuous, pid_trace[1:steps + 1, 2] * 180 / np.pi, 'r--', label='PID Theta2')
    plt.plot(rl_time, rl_trajectory[:, 2] * 180 / np.pi, 'y--', label='RL_passive_discrete Theta1')
    plt.plot(rl_time_active, rl_trajectory_active[:, 2] * 180 / np.pi, 'k--',
             label='RL_active_discrete Theta1')
    plt.ylabel(r'$\theta_2$ [deg]')
    plt.legend()
    plt.title('Theta2 Comparison')

    # 角速度比较
    plt.subplot(223)
    plt.plot(rl_time_continuous, rl_trajectory_continuous[:, 1] * 180 / np.pi, 'b-', label='RL dTheta1')
    plt.plot(rl_time_continuous, pid_trace[1:steps + 1, 1] * 180 / np.pi, 'r--', label='PID dTheta1')
    plt.plot(rl_time, rl_trajectory[:, 1] * 180 / np.pi, 'y--', label='RL_passive_discrete Theta1')
    plt.plot(rl_time_active, rl_trajectory_active[:, 1] * 180 / np.pi, 'k--',
             label='RL_active_discrete Theta1')
    plt.ylabel(r'$\dot{\theta}_1$ [deg/s]')
    plt.xlabel('Time [s]')
    plt.legend()

    plt.subplot(224)
    plt.plot(rl_time_continuous, rl_trajectory_continuous[:, 3] * 180 / np.pi, 'b-', label='RL_continuous dTheta2')
    plt.plot(rl_time_continuous, pid_trace[1:steps + 1, 3] * 180 / np.pi, 'r--', label='PID dTheta2')
    plt.plot(rl_time, rl_trajectory[:, 3] * 180 / np.pi, 'y--', label='RL_passive_discrete Theta1')
    plt.plot(rl_time_active, rl_trajectory_active[:, 3] * 180 / np.pi, 'k--',
             label='RL_active_discrete Theta1')
    plt.ylabel(r'$\dot{\theta}_2$ [deg/s]')
    plt.xlabel('Time [s]')
    plt.legend()

    plt.tight_layout()
    plt.show()
    plt.pause(0.5)
    # plt.savefig('PID_vs_RL_comparison.pdf')

    # plt.figure(figsize=(10, 4))
    # plt.plot(rl_time_continuous, rl_ddtheta2_continuous * 180 / np.pi, 'b-', label='RL_continuous ddTheta2')
    # plt.plot(rl_time, rl_ddtheta2 * 180 / np.pi, 'y--', label='RL_passive_discrete ddTheta2')
    # plt.plot(rl_time_active, rl_ddtheta2_active * 180 / np.pi, 'k--', label='RL_active_discrete ddTheta2')
    # plt.ylabel(r'$\ddot{\theta}_2$ [deg/s]')
    # plt.xlabel('Time [s]')
    # plt.legend()
    # plt.title('Acceleration obtained by torque')
    # plt.tight_layout()

    # 控制信号比较
    plt.figure(figsize=(10, 4))
    plt.plot(rl_time_continuous, rl_actions_continuous * torque, 'b-', label='RL Control_continuous')
    plt.plot(rl_time_active, rl_actions_active * torque, 'k-', label='RL Control_active_discrete')
    plt.plot(rl_time, rl_actions * torque, 'y-', label='RL Control_passive_discrete')
    plt.plot(rl_time_continuous, pid_actions[:steps] * torque, 'r--', label='PID Control')
    plt.ylabel('Torque [Nm]')
    plt.xlabel('Time [s]')
    plt.legend()
    plt.title('Control Signal Comparison')
    plt.tight_layout()
    # plt.savefig('control_signal_comparison.pdf')
    plt.show()
    plt.pause(0.5)

    # 能量比较
    plt.figure(figsize=(10, 4))
    plt.plot(rl_time_continuous, rl_energy_continuous, 'b-', label='RL_continuous Energy')
    plt.plot(rl_time, rl_energy, 'y-', label='RL_passive_discrete Energy')
    plt.plot(rl_time_active, rl_energy_active, 'k-', label='RL_active_discrete Energy')
    plt.plot(rl_time_continuous, pid_energy[:steps], 'r--', label='PID Energy')
    plt.ylabel('Energy [J]')
    plt.xlabel('Time [s]')
    plt.legend()
    plt.title('Energy Comparison')
    plt.tight_layout()
    plt.show()
    plt.pause(0.5)

    plt.figure(figsize=(10, 4))
    plt.plot(rl_time_continuous, rl_total_energy_continuous, 'b-', label='RL_continuous Total_Energy')
    plt.plot(rl_time, rl_total_energy, 'y-', label='RL_passive_discrete Total_Energy')
    plt.plot(rl_time_active, rl_total_energy_active, 'k-', label='RL_active_discrete Total_Energy')
    plt.plot(rl_time_continuous, total_energy_pid[:steps], 'r--', label='PID Total_Energy')
    plt.ylabel('Energy [J]')
    plt.xlabel('Time [s]')
    plt.legend()
    plt.title('Total_Energy Comparison')
    plt.tight_layout()
    plt.show()
    plt.pause(0.5)

    # 计算跟踪误差
    theta1_error = np.abs(rl_trajectory_continuous[:, 0] - pid_trace[1:steps + 1, 0]) * 180 / np.pi
    theta2_error = np.abs(rl_trajectory_continuous[:, 2] - pid_trace[1:steps + 1, 2]) * 180 / np.pi

    plt.figure(figsize=(10, 4))
    plt.plot(rl_time_continuous, theta1_error, 'b-', label='Theta1 Error')
    plt.plot(rl_time_continuous, theta2_error, 'r-', label='Theta2 Error')
    plt.ylabel('Tracking Error [deg]')
    plt.xlabel('Time [s]')
    plt.legend()
    plt.title('Tracking Error')
    plt.tight_layout()
    # plt.savefig('tracking_error.pdf')

    # 打印统计信息
    print(f"Mean Theta1 Error: {np.mean(theta1_error):.4f} deg")
    print(f"Max Theta1 Error: {np.max(theta1_error):.4f} deg")
    print(f"Mean Theta2 Error: {np.mean(theta2_error):.4f} deg")
    print(f"Max Theta2 Error: {np.max(theta2_error):.4f} deg")

    plt.show()
    plt.pause(0.5)


if __name__ == "__main__":
    main()
