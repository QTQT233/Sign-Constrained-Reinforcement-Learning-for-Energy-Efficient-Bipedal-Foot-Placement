import os
import h5py
import torch
import imageio
import numpy as np
from numba import njit
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib.patches import Rectangle

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


def compute_trig(l1_val, d, l2):
    """
    根据当前 l1 计算所有需要的三角函数
    返回: sin_alpha, cos_alpha, sin_theta2, cos_theta2, sin_theta1, cos_theta1
    """
    sin_theta2 = (d ** 2 + l2 ** 2 - l1_val ** 2) / (2 * d * l2)
    sin_theta2 = np.clip(sin_theta2, -1, 1)
    cos_theta2 = np.sqrt(1 - sin_theta2 ** 2)  # θ2 ∈ (-π/2, π/2) => cosθ2 > 0

    sin_theta1 = l2 * cos_theta2 / l1_val
    cos_theta1 = (d - l2 * sin_theta2) / l1_val

    # 直接计算 α = θ1 - θ2 的正余弦，避免开方符号歧义
    sin_alpha = sin_theta1 * cos_theta2 - cos_theta1 * sin_theta2
    cos_alpha = cos_theta1 * cos_theta2 + sin_theta1 * sin_theta2

    return sin_alpha, cos_alpha, sin_theta2, cos_theta2, sin_theta1, cos_theta1


def M_func(l1_val, J1, m2, l2, d, C):
    """
    计算等效质量 M(l1)
    """
    sin_alpha, cos_alpha, _, _, _, _ = compute_trig(l1_val, d, l2)
    if abs(cos_alpha) < 1e-12:
        return np.inf
    tan_alpha = sin_alpha / cos_alpha
    sec_alpha = 1 / cos_alpha
    term1 = (J1 / l1_val ** 2 + m2) * tan_alpha ** 2
    term2 = C * sec_alpha ** 2
    return m2 + term1 + term2


def dVdl1_func(l1_val, m1, l1_prime, l2, m2, l2_prime, g, d):
    """
    计算势能导数 dV/dl1
    """
    sin_alpha, cos_alpha, sin_theta2, cos_theta2, sin_theta1, cos_theta1 = compute_trig(l1_val, d, l2)
    A = m1 * l1_prime * l2 / l1_val + m2 * (l2 - l2_prime)
    dA_dl1 = - m1 * l1_prime * l2 / l1_val ** 2
    if cos_theta2 < 1e-12:
        return np.inf
    dcos_theta2_dl1 = (l1_val * sin_theta2) / (d * l2 * cos_theta2)
    return -g * (dcos_theta2_dl1 * A + cos_theta2 * dA_dl1)


def dMdl1_num(l1_val, J1, m2, l2, d, C, eps=1e-6):
    """
    数值微分求 dM/dl1
    """
    M1 = M_func(l1_val + eps, J1, m2, l2, d, C)
    M2 = M_func(l1_val - eps, J1, m2, l2, d, C)
    if np.isinf(M1) or np.isinf(M2):
        return 0.0
    return (M1 - M2) / (2 * eps)


def dynamics(t, y, m1, m2, l2, l1_prime, l2_prime, J1, J2, g, d, C, F):
    """
    系统动力学函数
    y = [theta1, theta2, l1, omega1, omega2, v1]
    返回导数 [omega1, omega2, v1, omega1_dot, omega2_dot, v1_dot]
    """
    _, _, l1_val, _, _, v1 = y

    # 防止 l1 超出物理范围
    if l1_val <= 0.2 or l1_val >= 0.8:
        return [0, 0, 0, 0, 0, 0]

    # 根据 l1 计算约束下的角度
    sin_alpha, cos_alpha, sin_theta2, cos_theta2, sin_theta1, cos_theta1 = compute_trig(l1_val, d, l2)
    theta1 = np.arctan2(sin_theta1, cos_theta1)
    theta2 = np.arcsin(sin_theta2)  # cos_theta2>0 确保返回在 [-π/2, π/2]

    # 角速度表达式 (9)
    omega1 = - (v1 / l1_val) * (sin_alpha / cos_alpha)
    omega2 = - (v1 / l2) * (1 / cos_alpha)

    # 计算等效质量、导数、势能导数
    M = M_func(l1_val, J1, m2, l2, d, C)
    dM = dMdl1_num(l1_val, J1, m2, l2, d, C)
    dV = dVdl1_func(l1_val, m1, l1_prime, l2, m2, l2_prime, g, d)

    if np.isinf(M) or np.isinf(dV):
        return [0, 0, 0, 0, 0, 0]

    # 单自由度运动方程 (14) 求 v1_dot
    v1_dot = (F - 0.5 * dM * v1 ** 2 - dV) / M

    # 计算 dα/dt
    dalpha_dt = omega1 - omega2

    # 角加速度表达式（对 (9) 求导）
    omega1_dot = - ((v1_dot / l1_val - v1 ** 2 / l1_val ** 2) * (sin_alpha / cos_alpha)
                    + (v1 / l1_val) * (1 / cos_alpha ** 2) * dalpha_dt)
    omega2_dot = - ((v1_dot / l2) * (1 / cos_alpha)
                    + (v1 / l2) * (sin_alpha / cos_alpha ** 2) * dalpha_dt)

    return [omega1, omega2, v1, omega1_dot, omega2_dot, v1_dot]


def collision_dynamics_full(theta1, theta2, dtheta1_pre, dtheta2_pre, params_pre):
    # theta1 = np.radians(theta1_pre)
    # theta2 = np.radians(theta2_pre)

    m1 = params_pre['m1']
    m2 = params_pre['m2']
    l1 = params_pre['l1']
    l2 = params_pre['l2']
    l1_prime = params_pre['l1_prime']
    l2_prime = params_pre['l2_prime']
    J1_root = params_pre['J1']  # 杆1绕根部的惯量
    J2_cm = params_pre['J2']  # 杆2绕质心的惯量
    J1_cm = J1_root - m1 * l1_prime ** 2
    qdot_minus = np.array([0.0, 0.0, dtheta1_pre, dtheta2_pre])
    s1 = np.sin(theta1)
    c1 = np.cos(theta1)
    s2 = np.sin(theta2)
    c2 = np.cos(theta2)
    # 杆1的动能贡献
    # T1 = 1/2 m1 (ẋ^2 + ẏ^2 + l1_prime^2 θ̇1^2 - 2 ẋ l1_prime s1 θ̇1 + 2 ẏ l1_prime c1 θ̇1) + 1/2 J1_cm θ̇1^2
    # 杆2的动能贡献
    # T2 = 1/2 m2 [ (ẋ - l1 s1 θ̇1 + l2_prime c2 θ̇2)^2 + (ẏ + l1 c1 θ̇1 + l2_prime s2 θ̇2)^2 ] + 1/2 J2_cm θ̇2^2
    M = np.zeros((4, 4))

    M[0, 0] = m1 + m2
    M[1, 1] = m1 + m2
    # ẋ 与 ẏ 无耦合，所以 M[0,1]=0

    # ẋ 与 θ̇1
    M[0, 2] = -m1 * l1_prime * s1 + m2 * (-l1 * s1)  # 来自杆1的 -m1 l1_prime s1 和杆2的 -m2 l1 s1

    M[0, 2] = -m1 * l1_prime * s1 - m2 * l1 * s1

    # ẏ 与 θ̇1
    M[1, 2] = m1 * l1_prime * c1 + m2 * l1 * c1
    M[0, 3] = m2 * l2_prime * c2
    M[1, 3] = m2 * l2_prime * s2
    M[2, 2] = m1 * l1_prime ** 2 + J1_cm + m2 * (l1 ** 2)
    M[3, 3] = m2 * l2_prime ** 2 + J2_cm
    M[2, 3] = m2 * l1 * l2_prime * np.sin(theta2 - theta1)
    M[3, 2] = M[2, 3]

    # 填充对称部分
    M[1, 0] = M[0, 1]
    M[2, 0] = M[0, 2]
    M[2, 1] = M[1, 2]
    M[3, 0] = M[0, 3]
    M[3, 1] = M[1, 3]

    # ---------- 约束雅可比 J_c (2x4) ----------
    # 约束: v_xE = ẋ - l1 sinθ1 θ̇1 + l2 cosθ2 θ̇2 = 0
    #       v_yE = ẏ + l1 cosθ1 θ̇1 + l2 sinθ2 θ̇2 = 0
    J_c = np.array([
        [1, 0, -l1 * s1, l2 * c2],
        [0, 1, l1 * c1, l2 * s2]
    ])

    # ---------- 构建冲击方程组 ----------
    # 未知数: q̇⁺ (4个) 和 λ (2个) -> 共6个
    A = np.zeros((6, 6))
    # 左上块: M
    A[0:4, 0:4] = M
    # 右上块: -J_c^T
    A[0:4, 4:6] = -J_c.T
    # 左下块: J_c
    A[4:6, 0:4] = J_c
    # 右下块: 0

    # 右端项 b = [M q̇⁻; 0]
    b = np.zeros(6)
    b[0:4] = M @ qdot_minus

    # 求解线性方程组
    try:
        x = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        print("警告：矩阵奇异，使用最小二乘求解。")
        x = np.linalg.lstsq(A, b, rcond=None)[0]
    vx_root = x[0]
    vy_root = x[1]
    dtheta1_post_abs = x[2]
    dtheta2_post_abs = x[3]
    lam = x[4:6]
    theta1_new = np.arctan2(l2 * c2, -l2 * s2)
    theta2_new = theta1 - np.pi / 2
    v = np.array([-l1 * c1, -l1 * s1])
    dtheta1_new = dtheta2_post_abs
    dtheta2_new = dtheta1_post_abs

    return theta1_new, theta2_new, dtheta1_new, dtheta2_new, vx_root, vy_root


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


step = 3
stop_flag = 0
settle2 = np.deg2rad(1)
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

m1_l2 = 0.6503
m2_l2 = 1.5981
l1_l2 = 0.521
l2_l2 = 0.481
l1_prime_l2 = 0.30300000000000005
l2_prime_l2 = 0.05916
J1_l2 = 0.34
J2_l2 = 0.1
c1_l2 = 0.001
c2_l2 = 0.0001

torque_motor = 2
gear_radius = 0.021
F = torque_motor / gear_radius
flag = 0
g = 9.8
dt = 0.01
torque = 4
simu_time = 3.2
steps = int(simu_time / dt)
reward_scale = 5
action_value_weight = 0.03
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


class PendulumEnv:
    def __init__(self, g_=g, m1_=m1, l1_=l1_prime, m2_=m2, l2_=l2_prime, c1_=c1, c2_=c2, dt_=dt, torque_=torque,
                 l1_1=l1, l2_1=l2, j1_=J1, j2_=J2):
        self.state = None
        self.next_state = None
        self.max_torque = torque_
        self.dt = dt_
        self.g = g_
        self.m1 = m1_
        self.m2 = m2_
        self.l1 = l1_1
        self.l2 = l2_1
        self.l1_s = l1_
        self.l2_s = l2_
        self.J1 = j1_
        self.J2 = j2_
        self.c1 = c1_
        self.c2 = c2_
        self.reward = None
        self.over = None
        self.check = 0
        self.y = np.zeros(4)
        self.action = np.array([0, self.max_torque])

    def step(self, act_index):
        self.y[0], self.y[1], self.y[2], self.y[3] = self.state
        action_value = self.action[act_index]
        a = np.array([[(self.J1 + self.m2 * self.l1 ** 2),
                       self.m2 * self.l1 * self.l2_s * np.sin(self.y[2] - self.y[0])],
                      [self.m2 * self.l1 * self.l2_s * np.sin(self.y[2] - self.y[0]),
                       (self.J2 + self.m2 * self.l2_s ** 2)]])
        b = np.array(
            [[-self.m2 * self.y[1] * self.l1 * self.y[3] * self.l2_s * np.cos(self.y[2] - self.y[0])
              - self.m1 * self.g * self.l1_s * np.cos(self.y[0]) - self.m2 * self.g * self.l1 * np.cos(self.y[0])
              - self.m2 * self.l1 * self.l2_s * self.y[3] * np.cos(self.y[2] - self.y[0]) * (self.y[3] - self.y[1])],
             [-self.m2 * self.l1 * self.l2_s * self.y[1] * np.cos(self.y[2] - self.y[0]) * (self.y[3] - self.y[1])
              + self.m2 * self.y[1] * self.l1 * self.y[3] * self.l2_s * np.cos(self.y[2] - self.y[0])
              - self.m2 * self.g * self.l2_s * np.sin(self.y[2])]])
        inverse_a = np.linalg.inv(a)
        arr = np.array([[-action_value], [action_value]])
        temp = inverse_a @ (b + arr)
        self.y[1] += (temp[0][0] - self.c1 * self.y[1]) * self.dt
        self.y[0] += self.y[1] * self.dt
        self.y[3] += (temp[1][0] - self.c2 * self.y[3]) * self.dt
        self.y[2] += self.y[3] * self.dt
        reward1 = abs(self.y[0] - np.deg2rad(target1)) < settle and abs(self.y[2] - np.deg2rad(target2)) < settle
        reward2 = abs(self.y[0] - np.deg2rad(target3)) < settle2 and abs(self.y[2] - np.deg2rad(target4)) < settle22
        if reward1:
            self.reward = reward_scale * 3
            self.over = True
        elif reward2:
            self.reward = reward_scale * 2
            self.over = True
        elif abs(self.y[0] - np.pi / 2) > rad_theta1_range or abs(self.y[2]) > rad_theta2_range:
            self.reward = -reward_scale * 5
            self.over = True
        # elif (self.l1 * np.sin(self.y[0]) - self.l2 * np.cos(self.y[2])) < 0.03:
        #     self.reward = -reward_scale * 5
        #     self.over = True
        else:
            self.reward = 0
            self.over = False
        self.reward += -abs(action_value) * action_value_weight
        self.next_state = np.array([self.y[0], self.y[1], self.y[2], self.y[3]])
        self.state = np.copy(self.next_state)
        return self.next_state, self.reward, self.over

    def reset(self):
        theta_1_initial = np.deg2rad(np.random.uniform(target1, target3))
        theta_2_initial = np.deg2rad(np.random.uniform(-60, 60))  # rads
        theta_1_dot_initial = np.random.uniform(-speed_range, speed_range)
        theta_2_dot_initial = np.random.uniform(-speed_range, speed_range)
        self.state = np.array([theta_1_initial, theta_1_dot_initial, theta_2_initial, theta_2_dot_initial])
        return self.state

    def define(self, theta1, dtheta1, theta2, dtheta2):
        self.state = np.array([theta1, dtheta1, theta2, dtheta2])
        return self.state


class Pendulumenv:
    def __init__(self, g_=g, m1_=m1, l1_=l1_prime, m2_=m2, l2_=l2_prime, c1_=c1, c2_=c2, dt_=dt, torque_=torque,
                 l1_1=l1, l2_1=l2, j1_=J1, j2_=J2):
        self.state = None
        self.next_state = None
        self.max_torque = torque_
        self.dt = dt_
        self.g = g_
        self.m1 = m1_
        self.m2 = m2_
        self.l1 = l1_1
        self.l2 = l2_1
        self.l1_s = l1_
        self.l2_s = l2_
        self.J1 = j1_
        self.J2 = j2_
        self.c1 = c1_
        self.c2 = c2_
        self.reward = None
        self.over = None
        self.check = 0
        self.y = np.zeros(4)
        self.action = np.array([0, -self.max_torque])

    def step(self, act_index):
        self.y[0], self.y[1], self.y[2], self.y[3] = self.state
        action_value = self.action[act_index]
        a = np.array([[(self.J1 + self.m2 * self.l1 ** 2),
                       self.m2 * self.l1 * self.l2_s * np.sin(self.y[2] - self.y[0])],
                      [self.m2 * self.l1 * self.l2_s * np.sin(self.y[2] - self.y[0]),
                       (self.J2 + self.m2 * self.l2_s ** 2)]])
        b = np.array(
            [[-self.m2 * self.y[1] * self.l1 * self.y[3] * self.l2_s * np.cos(self.y[2] - self.y[0])
              - self.m1 * self.g * self.l1_s * np.cos(self.y[0]) - self.m2 * self.g * self.l1 * np.cos(self.y[0])
              - self.m2 * self.l1 * self.l2_s * self.y[3] * np.cos(self.y[2] - self.y[0]) * (self.y[3] - self.y[1])],
             [-self.m2 * self.l1 * self.l2_s * self.y[1] * np.cos(self.y[2] - self.y[0]) * (self.y[3] - self.y[1])
              + self.m2 * self.y[1] * self.l1 * self.y[3] * self.l2_s * np.cos(self.y[2] - self.y[0])
              - self.m2 * self.g * self.l2_s * np.sin(self.y[2])]])
        inverse_a = np.linalg.inv(a)
        arr = np.array([[-action_value], [action_value]])
        temp = inverse_a @ (b + arr)
        self.y[1] += (temp[0][0] - self.c1 * self.y[1]) * self.dt
        self.y[0] += self.y[1] * self.dt
        self.y[3] += (temp[1][0] - self.c2 * self.y[3]) * self.dt
        self.y[2] += self.y[3] * self.dt
        reward1 = abs(self.y[0] - np.deg2rad(target1)) < settle and abs(self.y[2] - np.deg2rad(target2)) < settle
        reward2 = abs(self.y[0] - np.deg2rad(target3)) < settle2 and abs(self.y[2] - np.deg2rad(target4)) < settle2
        if reward1:
            self.reward = reward_scale * 3
            self.over = True
        elif reward2:
            self.reward = reward_scale * 2
            self.over = True
        elif abs(self.y[0] - np.pi / 2) > rad_theta1_range or abs(self.y[2]) > rad_theta2_range:
            self.reward = -reward_scale * 5
            self.over = True
        # elif (self.l1 * np.sin(self.y[0]) - self.l2 * np.cos(self.y[2])) < 0.03:
        #     self.reward = -reward_scale * 5
        #     self.over = True
        else:
            self.reward = 0
            self.over = False
        self.reward += -abs(action_value) * action_value_weight
        self.next_state = np.array([self.y[0], self.y[1], self.y[2], self.y[3]])
        self.state = np.copy(self.next_state)
        return self.next_state, self.reward, self.over

    def reset(self):
        theta_1_initial = np.deg2rad(np.random.uniform(target1, target3))
        theta_2_initial = np.deg2rad(np.random.uniform(-60, 60))  # rads
        theta_1_dot_initial = np.random.uniform(-speed_range, speed_range)
        theta_2_dot_initial = np.random.uniform(-speed_range, speed_range)
        self.state = np.array([theta_1_initial, theta_1_dot_initial, theta_2_initial, theta_2_dot_initial])
        return self.state

    def define(self, theta1, dtheta1, theta2, dtheta2):
        self.state = np.array([theta1, dtheta1, theta2, dtheta2])
        return self.state


class PendulumEnv1:
    def __init__(self, g_=g, m1_=m1_l2, l1_=l1_prime_l2, m2_=m2_l2, l2_=l2_prime_l2, c1_=c1_l2, c2_=c2_l2, dt_=dt,
                 torque_=torque, l1_1=l1_l2, l2_1=l2_l2, j1_=J1_l2, j2_=J2_l2):
        self.state = None
        self.next_state = None
        self.max_torque = torque_
        self.dt = dt_
        self.g = g_
        self.m1 = m1_
        self.m2 = m2_
        self.l1 = l1_1
        self.l2 = l2_1
        self.l1_s = l1_
        self.l2_s = l2_
        self.J1 = j1_
        self.J2 = j2_
        self.c1 = c1_
        self.c2 = c2_
        self.reward = None
        self.over = None
        self.check = 0
        self.y = np.zeros(4)
        self.action = np.array([0, self.max_torque])

    def step(self, act_index):
        self.y[0], self.y[1], self.y[2], self.y[3] = self.state
        action_value = self.action[act_index]
        a = np.array([[(self.J1 + self.m2 * self.l1 ** 2),
                       self.m2 * self.l1 * self.l2_s * np.sin(self.y[2] - self.y[0])],
                      [self.m2 * self.l1 * self.l2_s * np.sin(self.y[2] - self.y[0]),
                       (self.J2 + self.m2 * self.l2_s ** 2)]])
        b = np.array(
            [[-self.m2 * self.y[1] * self.l1 * self.y[3] * self.l2_s * np.cos(self.y[2] - self.y[0])
              - self.m1 * self.g * self.l1_s * np.cos(self.y[0]) - self.m2 * self.g * self.l1 * np.cos(self.y[0])
              - self.m2 * self.l1 * self.l2_s * self.y[3] * np.cos(self.y[2] - self.y[0]) * (self.y[3] - self.y[1])],
             [-self.m2 * self.l1 * self.l2_s * self.y[1] * np.cos(self.y[2] - self.y[0]) * (self.y[3] - self.y[1])
              + self.m2 * self.y[1] * self.l1 * self.y[3] * self.l2_s * np.cos(self.y[2] - self.y[0])
              - self.m2 * self.g * self.l2_s * np.sin(self.y[2])]])
        inverse_a = np.linalg.inv(a)
        arr = np.array([[-action_value], [action_value]])
        temp = inverse_a @ (b + arr)
        self.y[1] += (temp[0][0] - self.c1 * self.y[1]) * self.dt
        self.y[0] += self.y[1] * self.dt
        self.y[3] += (temp[1][0] - self.c2 * self.y[3]) * self.dt
        self.y[2] += self.y[3] * self.dt
        reward1 = abs(self.y[0] - np.deg2rad(target1)) < settle and abs(self.y[2] - np.deg2rad(target2)) < settle
        reward2 = abs(self.y[0] - np.deg2rad(target3)) < settle2 and abs(self.y[2] - np.deg2rad(target4)) < settle22
        if reward1:
            self.reward = reward_scale * 3
            self.over = True
        elif reward2:
            self.reward = reward_scale * 2
            self.over = True
        elif abs(self.y[0] - np.pi / 2) > rad_theta1_range or abs(self.y[2]) > rad_theta2_range:
            self.reward = -reward_scale * 5
            self.over = True
        # elif (self.l1 * np.sin(self.y[0]) - self.l2 * np.cos(self.y[2])) < 0.03:
        #     self.reward = -reward_scale * 5
        #     self.over = True
        else:
            self.reward = 0
            self.over = False
        self.reward += -abs(action_value) * action_value_weight
        self.next_state = np.array([self.y[0], self.y[1], self.y[2], self.y[3]])
        self.state = np.copy(self.next_state)
        return self.next_state, self.reward, self.over

    def reset(self):
        theta_1_initial = np.deg2rad(np.random.uniform(target1, target3))
        theta_2_initial = np.deg2rad(np.random.uniform(-60, 60))  # rads
        theta_1_dot_initial = np.random.uniform(-speed_range, speed_range)
        theta_2_dot_initial = np.random.uniform(-speed_range, speed_range)
        self.state = np.array([theta_1_initial, theta_1_dot_initial, theta_2_initial, theta_2_dot_initial])
        return self.state

    def define(self, theta1, dtheta1, theta2, dtheta2):
        self.state = np.array([theta1, dtheta1, theta2, dtheta2])
        return self.state


class Pendulumenv1:
    def __init__(self, g_=g, m1_=m1_l2, l1_=l1_prime_l2, m2_=m2_l2, l2_=l2_prime_l2, c1_=c1_l2, c2_=c2_l2, dt_=dt,
                 torque_=torque, l1_1=l1_l2, l2_1=l2_l2, j1_=J1_l2, j2_=J2_l2):
        self.state = None
        self.next_state = None
        self.max_torque = torque_
        self.dt = dt_
        self.g = g_
        self.m1 = m1_
        self.m2 = m2_
        self.l1 = l1_1
        self.l2 = l2_1
        self.l1_s = l1_
        self.l2_s = l2_
        self.J1 = j1_
        self.J2 = j2_
        self.c1 = c1_
        self.c2 = c2_
        self.reward = None
        self.over = None
        self.check = 0
        self.y = np.zeros(4)
        self.action = np.array([0, -self.max_torque])

    def step(self, act_index):
        self.y[0], self.y[1], self.y[2], self.y[3] = self.state
        action_value = self.action[act_index]
        a = np.array([[(self.J1 + self.m2 * self.l1 ** 2),
                       self.m2 * self.l1 * self.l2_s * np.sin(self.y[2] - self.y[0])],
                      [self.m2 * self.l1 * self.l2_s * np.sin(self.y[2] - self.y[0]),
                       (self.J2 + self.m2 * self.l2_s ** 2)]])
        b = np.array(
            [[-self.m2 * self.y[1] * self.l1 * self.y[3] * self.l2_s * np.cos(self.y[2] - self.y[0])
              - self.m1 * self.g * self.l1_s * np.cos(self.y[0]) - self.m2 * self.g * self.l1 * np.cos(self.y[0])
              - self.m2 * self.l1 * self.l2_s * self.y[3] * np.cos(self.y[2] - self.y[0]) * (self.y[3] - self.y[1])],
             [-self.m2 * self.l1 * self.l2_s * self.y[1] * np.cos(self.y[2] - self.y[0]) * (self.y[3] - self.y[1])
              + self.m2 * self.y[1] * self.l1 * self.y[3] * self.l2_s * np.cos(self.y[2] - self.y[0])
              - self.m2 * self.g * self.l2_s * np.sin(self.y[2])]])
        inverse_a = np.linalg.inv(a)
        arr = np.array([[-action_value], [action_value]])
        temp = inverse_a @ (b + arr)
        self.y[1] += (temp[0][0] - self.c1 * self.y[1]) * self.dt
        self.y[0] += self.y[1] * self.dt
        self.y[3] += (temp[1][0] - self.c2 * self.y[3]) * self.dt
        self.y[2] += self.y[3] * self.dt
        reward1 = abs(self.y[0] - np.deg2rad(target1)) < settle and abs(self.y[2] - np.deg2rad(target2)) < settle
        reward2 = abs(self.y[0] - np.deg2rad(target3)) < settle2 and abs(self.y[2] - np.deg2rad(target4)) < settle22
        if reward1:
            self.reward = reward_scale * 3
            self.over = True
        elif reward2:
            self.reward = reward_scale * 2
            self.over = True
        elif abs(self.y[0] - np.pi / 2) > rad_theta1_range or abs(self.y[2]) > rad_theta2_range:
            self.reward = -reward_scale * 5
            self.over = True
        # elif (self.l1 * np.sin(self.y[0]) - self.l2 * np.cos(self.y[2])) < 0.03:
        #     self.reward = -reward_scale * 5
        #     self.over = True
        else:
            self.reward = 0
            self.over = False
        self.reward += -abs(action_value) * action_value_weight
        self.next_state = np.array([self.y[0], self.y[1], self.y[2], self.y[3]])
        self.state = np.copy(self.next_state)
        return self.next_state, self.reward, self.over

    def reset(self):
        theta_1_initial = np.deg2rad(np.random.uniform(target1, target3))
        theta_2_initial = np.deg2rad(np.random.uniform(-60, 60))  # rads
        theta_1_dot_initial = np.random.uniform(-speed_range, speed_range)
        theta_2_dot_initial = np.random.uniform(-speed_range, speed_range)
        self.state = np.array([theta_1_initial, theta_1_dot_initial, theta_2_initial, theta_2_dot_initial])
        return self.state

    def define(self, theta1, dtheta1, theta2, dtheta2):
        self.state = np.array([theta1, dtheta1, theta2, dtheta2])
        return self.state


with h5py.File('D:/L&S/Mas/Project/Paper2/working_l1_stand_save_passive_1(-1,0,1)-30,60', 'r') as h5f:
    Q1 = np.array(h5f['working_save_passive'][:])
with h5py.File('D:/L&S/Mas/Project/Paper2/working_l2_stand_save_passive_1(-1,0,1)-30,60', 'r') as h5f:
    Q2 = np.array(h5f['working_save_passive'][:])
with h5py.File('D:/L&S/Mas/Project/Paper2/reward_check_l1_stand_passive_1(-1,0,1)-30,60', 'r') as h5f:
    C1 = np.array(h5f['reward_check'][:])
with h5py.File('D:/L&S/Mas/Project/Paper2/reward_check_l2_stand_passive_1(-1,0,1)-30,60', 'r') as h5f:
    C2 = np.array(h5f['reward_check'][:])
N1 = 30
N2 = 60
N = N1
speed_nondim = 2
tht1s = np.linspace(60, 120, N1) * np.pi / 180
dtht1s = np.linspace(-speed_nondim, speed_nondim, N2)
tht2s = np.linspace(-60, 60, N1) * np.pi / 180
dtht2s = np.linspace(-speed_nondim, speed_nondim, N2)
plt.figure(figsize=(4, 2), dpi=200)
l1_draw = 0.24
l2_draw = 0.24
count = 0
thetas1 = []
thetas2 = []
dthetas1 = []
dthetas2 = []
actions = []
stance_leg = 1
theta1_new, theta2_new, dtheta1_new, dtheta2_new = 0, 0, 0, 0
while step != 0:
    y = np.zeros(4)
    if count == 0:
        y[0] = tht1s[19]
        y[1] = dtht1s[11]
        y[2] = tht2s[15]
        y[3] = dtht2s[7]
        stance_leg = 1
    else:
        y[0] = theta1_new
        y[1] = dtheta1_new
        y[2] = theta2_new
        y[3] = dtheta2_new
    i = np.argmin(abs(y[0] - tht1s))
    j = np.argmin(abs(y[1] - dtht1s))
    k = np.argmin(abs(y[2] - tht2s))
    ll = np.argmin(abs(y[3] - dtht2s))
    if stance_leg == 1:
        if Q1[i, j, k, ll] == -1:
            env = Pendulumenv()
            actions = [0, -1]
            device = torch.device("cpu")
            nnn = 256
            policy = torch.nn.Sequential(torch.nn.Linear(4, nnn * 2), torch.nn.Tanh(),
                                         torch.nn.Linear(nnn * 2, nnn), torch.nn.Tanh(),
                                         torch.nn.Linear(nnn, 2), torch.nn.Softmax(dim=1))
            policy.to(device)
            optimizer_policy = torch.optim.Adam(policy.parameters(), lr=1e-3)
            loss_fn = torch.nn.MSELoss()

            policy.load_state_dict(
                torch.load('D:/L&S/Mas/Project/Paper2/l1_stand(0,-1)_696.pth'))
            policy.to(device)

        else:
            env = PendulumEnv()
            actions = [0, 1]
            device = torch.device("cpu")
            nnn = 256
            policy = torch.nn.Sequential(torch.nn.Linear(4, nnn * 2), torch.nn.Tanh(),
                                         torch.nn.Linear(nnn * 2, nnn), torch.nn.Tanh(),
                                         torch.nn.Linear(nnn, 2), torch.nn.Softmax(dim=1))
            policy.to(device)
            optimizer_policy1 = torch.optim.Adam(policy.parameters(), lr=1e-3)
            loss_fn1 = torch.nn.MSELoss()

            policy.load_state_dict(
                torch.load('D:/L&S/Mas/Project/Paper2/l1_stand(0,1)_676.pth'))
            policy.to(device)
    else:
        if Q2[i, j, k, ll] == -1:
            env = Pendulumenv1()
            actions = [0, -1]
            device = torch.device("cpu")
            nnn = 256
            policy = torch.nn.Sequential(torch.nn.Linear(4, nnn * 2), torch.nn.Tanh(),
                                         torch.nn.Linear(nnn * 2, nnn), torch.nn.Tanh(),
                                         torch.nn.Linear(nnn, 2), torch.nn.Softmax(dim=1))
            policy.to(device)
            optimizer_policy = torch.optim.Adam(policy.parameters(), lr=1e-3)
            loss_fn = torch.nn.MSELoss()

            policy.load_state_dict(
                torch.load('D:/L&S/Mas/Project/Paper2/l2_stand(0,-1)_829.pth'))
        else:
            env = PendulumEnv1()
            actions = [0, 1]
            device = torch.device("cpu")
            nnn = 256
            policy = torch.nn.Sequential(torch.nn.Linear(4, nnn * 2), torch.nn.Tanh(),
                                         torch.nn.Linear(nnn * 2, nnn), torch.nn.Tanh(),
                                         torch.nn.Linear(nnn, 2), torch.nn.Softmax(dim=1))
            policy.to(device)
            optimizer_policy1 = torch.optim.Adam(policy.parameters(), lr=1e-3)
            loss_fn1 = torch.nn.MSELoss()

            policy.load_state_dict(
                torch.load('D:/L&S/Mas/Project/Paper2/l2_stand(0,1)_815.pth'))
            policy.to(device)

    over = False
    state = env.reset()
    state[0] = y[0]
    state[1] = y[1]
    state[2] = y[2]
    state[3] = y[3]
    env.state[0] = y[0]
    env.state[1] = y[1]
    env.state[2] = y[2]
    env.state[3] = y[3]
    next_state = state
    count_time = 0
    while not over:
        reward1 = abs(y[0] - np.deg2rad(target1)) < settle and abs(
            y[2] - np.deg2rad(target2)) < settle
        reward2 = abs(y[0] - np.deg2rad(target3)) < settle2 and abs(
            y[2] - np.deg2rad(target4)) < settle2
        state_in_net_ = np.array([(state[0] - np.pi / 2) / rad_theta1_range,
                                  state[1] / speed_range, state[2] / rad_theta2_range, state[3] / speed_range])
        prob = policy(torch.FloatTensor(state_in_net_).reshape(1, 4))[0].cpu().detach().numpy()
        action_index = actions[np.argmax(prob)]
        next_state, reward, over = env.step(action_index)
        thetas1.append(state[0] * 180 / np.pi)
        thetas2.append(state[2] * 180 / np.pi)
        dthetas1.append(state[1] * 180 / np.pi)
        dthetas2.append(state[3] * 180 / np.pi)
        actions.append(actions[action_index] * torque)
        state = np.copy(next_state)

    reward1 = abs(state[0] - np.deg2rad(target1)) < settle and \
              abs(state[2] - np.deg2rad(target2)) < settle
    reward2 = abs(state[0] - np.deg2rad(target3)) < settle2 and \
              abs(state[2] - np.deg2rad(target4)) < settle2
    # print(state[1])
    # print(state[3])
    # print("")
    if reward1:
        print("碰撞触发！计算碰撞后状态...")
        params_pre = {
            'm1': env.m1,
            'm2': env.m2,
            'l1': env.l1,
            'l2': env.l2,
            'l1_prime': env.l1_s,
            'l2_prime': env.l2_s,
            'J1': env.J1,
            'J2': env.J2
        }
        print("碰撞前状态：")
        print(f"  站立腿与地面夹角 = {np.rad2deg(state[0]):.2f}°")
        print(f"  摆动腿与竖直方向夹角 = {np.rad2deg(state[2]):.2f}°")
        print(f"  站立腿角速度 = {state[1]:.4f} rad/s")
        print(f"  摆动腿角速度 = {state[3]:.4f} rad/s")

        # 调用碰撞函数
        theta1_new, theta2_new, dtheta1_new, dtheta2_new, _, _ = collision_dynamics_full(
            state[0], state[2], state[1], state[3], params_pre)

        # 输出结果（转换为度便于观察）
        print("碰撞后新状态（新定义）：")
        print(f"  新站立腿与地面夹角 = {np.rad2deg(theta1_new):.2f}°")
        print(f"  新摆动腿与竖直方向夹角 = {np.rad2deg(theta2_new):.2f}°")
        print(f"  新站立腿角速度 = {dtheta1_new:.4f} rad/s")
        print(f"  新摆动腿角速度 = {dtheta2_new:.4f} rad/s")

        if stance_leg == 1:
            i = np.argmin(abs(np.rad2deg(theta1_new) - tht1s))
            j = np.argmin(abs(dtheta1_new - dtht1s))
            k = np.argmin(abs(np.rad2deg(theta2_new) - tht2s))
            ll = np.argmin(abs(dtheta2_new - dtht2s))
            # while C2[i, j, k, ll, 0] != 1:
            dtheta1_new -= 0.5
            stop_flag += 1
            i = np.argmin(abs(theta1_new - tht1s))
            j = np.argmin(abs(dtheta1_new - dtht1s))
            k = np.argmin(abs(theta2_new - tht2s))
            ll = np.argmin(abs(dtheta2_new - dtht2s))
            # print("")
            # print(i)
            # print(j)
            # print(k)
            # print(ll)

            print(Q2[i, j, k, ll])
            print(C2[i, j, k, ll, 0])
            print("蹬地后新状态（新定义）：")
            print(f"  新站立腿与地面夹角 = {np.rad2deg(theta1_new):.2f}°")
            print(f"  新摆动腿与竖直方向夹角 = {np.rad2deg(theta2_new):.2f}°")
            print(f"  新站立腿角速度 = {dtheta1_new:.4f} rad/s")
            print(f"  新摆动腿角速度 = {dtheta2_new:.4f} rad/s")
            # if stop_flag > 6:
            #     step = 0
            #     break
            stance_leg = 2
        elif stance_leg == 2:
            i = np.argmin(abs(np.rad2deg(theta1_new) - tht1s))
            j = np.argmin(abs(dtheta1_new - dtht1s))
            k = np.argmin(abs(np.rad2deg(theta2_new) - tht2s))
            ll = np.argmin(abs(dtheta2_new - dtht2s))
            # while C1[i, j, k, ll, 0] != 1:
            dtheta1_new -= 0.53
            # dtheta2_new += 0.8
            stop_flag += 1
            i = np.argmin(abs(theta1_new - tht1s))
            j = np.argmin(abs(dtheta1_new - dtht1s))
            k = np.argmin(abs(theta2_new - tht2s))
            ll = np.argmin(abs(dtheta2_new - dtht2s))
            # print("")
            # print(i)
            # print(j)
            # print(k)
            # print(ll)

            print(Q1[i, j, k, ll])
            print(C1[i, j, k, ll, 0])
            print("蹬地后新状态（新定义）：")
            print(f"  新站立腿与地面夹角 = {np.rad2deg(theta1_new):.2f}°")
            print(f"  新摆动腿与竖直方向夹角 = {np.rad2deg(theta2_new):.2f}°")
            print(f"  新站立腿角速度 = {dtheta1_new:.4f} rad/s")
            print(f"  新摆动腿角速度 = {dtheta2_new:.4f} rad/s")
            # if stop_flag > 6:
            #     step = 0
            #     break
            stance_leg = 1
        step -= 1
        count += 1
