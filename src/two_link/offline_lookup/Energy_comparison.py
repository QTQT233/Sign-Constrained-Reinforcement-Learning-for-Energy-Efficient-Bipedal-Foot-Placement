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
import os

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Running on the GPU")
else:
    device = torch.device("cpu")
    print("Running on the CPU")
torch.cuda.set_device(device)
nnn = 512

# 值函数网络保持不变
model = torch.nn.Sequential(torch.nn.Linear(4, nnn * 2), torch.nn.ReLU(),
                            torch.nn.Linear(nnn * 2, nnn), torch.nn.ReLU(),
                            torch.nn.Linear(nnn, 1))
for module in model.modules():
    if isinstance(module, torch.nn.Linear):
        torch.nn.init.orthogonal_(module.weight)

# 策略网络修改为输出均值和标准差
policy = torch.nn.Sequential(
    torch.nn.Linear(4, nnn * 2), torch.nn.Tanh(),
    torch.nn.Linear(nnn * 2, nnn), torch.nn.Tanh(),
    torch.nn.Linear(nnn, 2)  # 输出均值和标准差
)

policy.load_state_dict(
    torch.load(
        'D:/L&S/Mas/Project/Paper1/Check_continuous/Policy_Net_Pytorch(-1,0,1)_1560_continuous_old_form.pth'))
model.to(device)
policy.to(device)
# model.train()
optimizer_value = torch.optim.Adam(model.parameters(), lr=1e-4)
optimizer_policy = torch.optim.Adam(policy.parameters(), lr=1e-4)
loss_fn = torch.nn.MSELoss()


# 双摆动力学模型（与原始代码相同）
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


def load_rl_trajectory():
    # 参数（与原始代码相同）
    m1 = 1.4122
    m2 = 0.0839
    l1 = 0.22
    l2 = 0.31
    l1_r = 0.33
    l2_r = 0.33
    c1 = 0.062
    c2 = 0.3
    g = 9.8
    torque = 4
    j1 = m1 * l1 ** 2 / 2
    j2 = m2 * l2 ** 2 / 2

    # 时间参数
    simu_time = 3.2
    dt_plot = 0.005
    steps = int(simu_time / dt_plot)

    # 初始化状态（与原始代码相同）
    y = np.zeros(4)
    y[0] = 88.2 * np.pi / 180  # theta1
    y[1] = -19.98 * np.pi / 180  # dtheta1
    y[2] = 0 * np.pi / 180  # theta2
    y[3] = 0 * np.pi / 180  # dtheta2

    # 加载预训练Q表
    with h5py.File('C:/Users/Admin/Desktop/paper-code - F/working_save-ATC-50', 'r') as h5f:
        Q = np.array(h5f['working_save'][:])

    # 离散化空间（与原始代码相同）
    N = 50
    tht1s = np.linspace(np.pi / 3, 2 * np.pi / 3, N)
    dtht1s = np.linspace(-3.5, 3.5, N)
    tht2s = np.linspace(-np.pi / 3, np.pi / 3, N)
    dtht2s = np.linspace(-3.5, 3.5, N)

    # 存储结果
    y_trace = np.zeros((steps + 1, 4))
    y_trace[0, :] = y
    a_save = np.zeros((steps + 1, 1))
    E_save = np.zeros(steps + 1)
    ddtheta2_save = np.zeros(steps + 1)

    y_old = y[0]
    y2_old = y[2]
    y3_c = y[3]
    cont_old = 0
    i = np.argmin(abs(y[0] - tht1s))
    j = np.argmin(abs(y[1] - dtht1s))
    k = np.argmin(abs(y[2] - tht2s))
    ll = np.argmin(abs(y[3] - dtht2s))
    cont = 0
    con = 0

    # 主仿真循环（与原始代码相同）
    for step in range(steps):
        if y_old - y[0] >= 1.8 * np.pi / 180:
            y3_c = (y[2] - y2_old) / ((cont - cont_old) * dt_plot)
            i = np.argmin(abs(y[0] - tht1s))
            j = np.argmin(abs(y[1] - dtht1s))
            k = np.argmin(abs(y[2] - tht2s))
            ll = np.argmin(abs(y3_c - dtht2s))
            y_old = y[0]
            y2_old = y[2]
            cont_old = cont

        inverse_A, B = calc_new_a_b(y, m1, m2, j1, j2, l1_r, l2_r, l1, l2, g)
        a = Q[i, j, k, ll]
        if a == -2:
            a = 0
        if a != 0:
            con += 1
        if con != 0 and a == 0:
            c2 = 0.0001

        arr = np.array([[-a * torque], [a * torque]])
        temp = np.dot(inverse_A, B + arr)

        y[1] += temp[0] * dt_plot - c1 * y[1]
        y[0] += y[1] * dt_plot
        y[3] += temp[1] * dt_plot - c2 * y[3]
        y[2] += y[3] * dt_plot

        ddtheta2_save[step + 1] = temp[1] * dt_plot
        y_trace[step + 1, :] = y
        a_save[step + 1, 0] = a
        E_save[step + 1] = E_save[step] + abs(a) * torque * (abs(y[1]) + abs(y[3])) * dt_plot
        cont += 1

        if (abs(y[0] - np.pi / 3) < 0.08 and abs(y[2] - np.pi / 6) < 0.08) \
                or (abs(y[0] - 2 * np.pi / 3) < 0.08 and abs(y[2] - (-np.pi / 6)) < 0.08):
            print(f'\nAchieved the goal at step {step}!')
            break

    t = np.arange(0, (step + 1) * dt_plot, dt_plot)
    return t, y_trace[:step + 1, :], a_save[:step + 1, 0], E_save[:step + 1], ddtheta2_save[:step + 1]


def load_rl_active_trajectory():
    # 参数（与原始代码相同）
    m1 = 1.4122
    m2 = 0.0839
    l1 = 0.22
    l2 = 0.31
    l1_r = 0.33
    l2_r = 0.33
    c1 = 0.062
    c2 = 0.3
    g = 9.8
    torque = 4
    j1 = m1 * l1 ** 2
    j2 = m2 * l2 ** 2

    # 时间参数
    simu_time = 3.2
    dt_plot = 0.005
    steps = int(simu_time / dt_plot)

    # 初始化状态（与原始代码相同）
    y = np.zeros(4)
    y[0] = 88.2 * np.pi / 180  # theta1
    y[1] = -19.98 * np.pi / 180  # dtheta1
    y[2] = 0 * np.pi / 180  # theta2
    y[3] = 0 * np.pi / 180  # dtheta2

    # 加载连续动作策略网络
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nnn = 512

    # 策略网络结构
    policy1 = torch.nn.Sequential(torch.nn.Linear(4, nnn * 2), torch.nn.Tanh(),
                                 torch.nn.Linear(nnn * 2, nnn), torch.nn.Tanh(),
                                 torch.nn.Linear(nnn, 3), torch.nn.Softmax(dim=1))  

    # 加载预训练权重
    policy1.load_state_dict(
        torch.load('D:/L&S/Mas/Project/Paper1/Check_continuous/Policy_Net_Pytorch(-1,0,1)_1612.pth',
                   map_location=device))
    policy1.to(device)
    policy1.eval()  # 设置为评估模式

    # 归一化参数
    rad_theta1_range = np.deg2rad(90)
    rad_theta2_range = np.deg2rad(90)
    speed_range = 3.5

    # 存储结果
    y_trace = np.zeros((steps + 1, 4))
    y_trace[0, :] = y
    a_save = np.zeros(steps + 1)
    E_save = np.zeros(steps + 1)
    ddtheta2_save = np.zeros(steps + 1)
    actions = [-1, 0, 1]

    # 主仿真循环 - 使用连续动作策略
    for step in range(steps):
        # 归一化状态
        state_normalized = np.array([
            (y[0] - np.pi / 2) / rad_theta1_range,
            y[1] / speed_range,
            y[2] / rad_theta2_range,
            y[3] / speed_range
        ])

        # 通过策略网络获取动作
        prob_ = policy1(torch.FloatTensor(state_normalized).reshape(1, 4).to(device))[0].cpu().detach().numpy()
        a = actions[np.argmax(prob_)]
        # 计算动力学
        inverse_A, B = calc_new_a_b(y, m1, m2, j1, j2,  l1_r, l2_r, l1, l2, g)

        # 应用控制输入
        arr = np.array([[-a * torque], [a * torque]])
        temp = np.dot(inverse_A, B + arr)

        # 更新状态
        y[1] += temp[0] * dt_plot - c1 * y[1]
        y[0] += y[1] * dt_plot
        y[3] += temp[1] * dt_plot - c2 * y[3]
        y[2] += y[3] * dt_plot

        # 存储结果
        ddtheta2_save[step + 1] = temp[1] * dt_plot
        y_trace[step + 1, :] = y
        a_save[step + 1] = a
        E_save[step + 1] = E_save[step] + abs(a) * torque * (abs(y[1]) + abs(y[3])) * dt_plot

        # 检查终止条件
        if (abs(y[0] - np.pi / 3) < 0.08 and abs(y[2] - np.pi / 6) < 0.08) \
                or (abs(y[0] - 2 * np.pi / 3) < 0.08 and abs(y[2] - (-np.pi / 6)) < 0.08):
            print(f'\nAchieved the goal at step {step}!')
            break

    t = np.arange(0, (step + 1) * dt_plot, dt_plot)
    return t, y_trace[:step + 1, :], a_save[:step + 1], E_save[:step + 1], ddtheta2_save[:step + 1]


# 加载原始强化学习仿真的状态轨迹
def load_rl_continuous_trajectory():
    # 参数设置（与原始代码相同）
    m1 = 1.4122
    m2 = 0.0839
    l1 = 0.22
    l2 = 0.31
    l1_r = 0.33
    l2_r = 0.33
    c1 = 0.062
    c2 = 0.3
    g = 9.8
    torque = 4
    j1 = m1 * l1 ** 2
    j2 = m2 * l2 ** 2

    # 时间参数
    simu_time = 3.2
    dt_plot = 0.005
    steps = int(simu_time / dt_plot)

    # 初始化状态（与原始代码相同）
    y = np.zeros(4)
    y[0] = 88.2 * np.pi / 180  # theta1
    y[1] = -19.98 * np.pi / 180  # dtheta1
    y[2] = 0 * np.pi / 180  # theta2
    y[3] = 0 * np.pi / 180  # dtheta2

    # 加载连续动作策略网络
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nnn = 512

    # 策略网络结构
    policy = torch.nn.Sequential(
        torch.nn.Linear(4, nnn * 2), torch.nn.Tanh(),
        torch.nn.Linear(nnn * 2, nnn), torch.nn.Tanh(),
        torch.nn.Linear(nnn, 2)  # 输出均值和标准差
    )

    # 加载预训练权重
    policy.load_state_dict(
        torch.load('D:/L&S/Mas/Project/Paper1/Check_continuous/Policy_Net_Pytorch(-1,0,1)_1560_continuous_old_form.pth',
                   map_location=device))
    policy.to(device)
    policy.eval()  # 设置为评估模式

    # 归一化参数
    rad_theta1_range = np.deg2rad(90)
    rad_theta2_range = np.deg2rad(90)
    speed_range = 3.5

    # 存储结果
    y_trace = np.zeros((steps + 1, 4))
    y_trace[0, :] = y
    a_save = np.zeros(steps + 1)
    E_save = np.zeros(steps + 1)
    ddtheta2_save = np.zeros(steps + 1)

    # 主仿真循环 - 使用连续动作策略
    for step in range(steps):
        # 归一化状态
        state_normalized = np.array([
            (y[0] - np.pi / 2) / rad_theta1_range,
            y[1] / speed_range,
            y[2] / rad_theta2_range,
            y[3] / speed_range
        ])

        # 通过策略网络获取动作
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state_normalized).unsqueeze(0).to(device)
            output = policy(state_tensor)
            mean, log_std = output.chunk(2, dim=1)
            std = torch.exp(log_std)
            dist = torch.distributions.Normal(mean, std)
            action_value = dist.sample().item()

        # 裁剪动作值
        a = np.clip(action_value, -1.0, 1.0)

        # 计算动力学
        inverse_A, B = calc_new_a_b(y, m1, m2, j1, j2, l1_r, l2_r, l1, l2, g)

        # 应用控制输入
        arr = np.array([[-a * torque], [a * torque]])
        temp = np.dot(inverse_A, B + arr)

        # 更新状态
        y[1] += temp[0] * dt_plot - c1 * y[1]
        y[0] += y[1] * dt_plot
        y[3] += temp[1] * dt_plot - c2 * y[3]
        y[2] += y[3] * dt_plot

        # 存储结果
        ddtheta2_save[step + 1] = temp[1] * dt_plot
        y_trace[step + 1, :] = y
        a_save[step + 1] = a
        E_save[step + 1] = E_save[step] + abs(a) * torque * (abs(y[1]) + abs(y[3])) * dt_plot

        # 检查终止条件
        if (abs(y[0] - np.pi / 3) < 0.08 and abs(y[2] - np.pi / 6) < 0.08) \
                or (abs(y[0] - 2 * np.pi / 3) < 0.08 and abs(y[2] - (-np.pi / 6)) < 0.08):
            print(f'\nAchieved the goal at step {step}!')
            break

    t = np.arange(0, (step + 1) * dt_plot, dt_plot)
    return t, y_trace[:step + 1, :], a_save[:step + 1], E_save[:step + 1], ddtheta2_save[:step + 1]


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
        error1 = target_theta1 - current_theta1
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
        output = P

        # 限幅
        output = np.clip(output, self.min_output, self.max_output)
        self.previous_error = combined_error

        return output


# 主函数
def main():
    # 加载RL轨迹
    rl_time_continuous, rl_trajectory_continuous, rl_actions_continuous, rl_energy_continuous, rl_ddtheta2_continuous = load_rl_continuous_trajectory()
    rl_time, rl_trajectory, rl_actions, rl_energy, rl_ddtheta2 = load_rl_trajectory()
    rl_time_active, rl_trajectory_active, rl_actions_active, rl_energy_active, rl_ddtheta2_active = load_rl_active_trajectory()
    # 系统参数（与原始代码相同）
    m1 = 1.4122
    m2 = 0.0839
    l1 = 0.22
    l2 = 0.31
    l1_r = 0.33
    l2_r = 0.33
    c1 = 0.062
    c2 = 0.3
    g = 9.8
    torque = 4
    j1 = m1 * l1 ** 2
    j2 = m2 * l2 ** 2

    # 时间参数（与RL轨迹相同）
    dt_plot = 0.005
    steps = len(rl_time_continuous)

    # 初始化状态（与原始RL相同）
    y = np.zeros(4)
    y[0] = 88.2 * np.pi / 180  # theta1
    y[1] = -19.98 * np.pi / 180  # dtheta1
    y[2] = 0 * np.pi / 180  # theta2
    y[3] = 0 * np.pi / 180  # dtheta2

    # 创建PID控制器（用于跟踪RL轨迹）
    pid = TrajectoryTrackingPID(Kp=80.0, Ki=0.05, Kd=2.0,
                                max_output=1.0, min_output=-1.0, dt=dt_plot)

    # 存储结果
    pid_trace = np.zeros((steps + 1, 4))
    pid_trace[0, :] = y
    pid_actions = np.zeros(steps + 1)
    pid_energy = np.zeros(steps + 1)
    # 主仿真循环
    for step in range(steps):
        # 获取当前时间点的RL目标状态
        target_theta1 = rl_trajectory_continuous[step, 0]
        target_theta2 = rl_trajectory_continuous[step, 2]

        # 计算PID控制输出
        a = pid.compute(target_theta1, target_theta2, y[0], y[2])

        # 计算动力学
        inverse_A, B = calc_new_a_b(y, m1, m2, j1, j2, l1_r, l2_r, l1, l2, g)

        # 应用控制输入（与原始代码相同的扭矩结构）
        arr = np.array([[-a * torque], [a * torque]])
        temp = np.dot(inverse_A, B + arr)

        # 更新状态（欧拉积分）
        y[1] += temp[0] * dt_plot - c1 * y[1]
        y[0] += y[1] * dt_plot
        y[3] += temp[1] * dt_plot - c2 * y[3]
        y[2] += y[3] * dt_plot

        # 存储结果
        pid_trace[step + 1, :] = y
        pid_actions[step] = a
        if step >= 1:
            pid_energy[step] = pid_energy[step-1] + abs(a) * torque * (abs(y[1]) + abs(y[3])) * dt_plot
        else:
            pid_energy[step] =  abs(a) * torque * (y[1] + y[3])
        # 打印进度
        if step % 50 == 0:
            print(f"Step: {step}/{steps}, Theta1: {y[0] * 180 / np.pi:.2f}°, Theta2: {y[2] * 180 / np.pi:.2f}°")

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
    # plt.savefig('PID_vs_RL_comparison.pdf')

    plt.figure(figsize=(10, 4))
    plt.plot(rl_time_continuous, rl_ddtheta2_continuous * 180 / np.pi, 'b-', label='RL_continuous ddTheta2')
    plt.plot(rl_time, rl_ddtheta2 * 180 / np.pi, 'y--', label='RL_passive_discrete ddTheta2')
    plt.plot(rl_time_active, rl_ddtheta2_active * 180 / np.pi, 'k--', label='RL_active_discrete ddTheta2')
    plt.ylabel(r'$\ddot{\theta}_2$ [deg/s]')
    plt.xlabel('Time [s]')
    plt.legend()
    plt.title('Acceleration obtained by torque')
    plt.tight_layout()


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


if __name__ == "__main__":
    main()
