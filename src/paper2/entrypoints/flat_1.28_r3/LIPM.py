import os
import h5py
import torch
import imageio
import numpy as np
from numba import njit
import matplotlib.pyplot as plt
import cvxpy as cp
from scipy.integrate import solve_ivp
from matplotlib.patches import Rectangle

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
torch.manual_seed(0)
np.random.seed(0)


def collision_dynamics_full(theta1, theta2, dtheta1_pre, dtheta2_pre, params_pre):
    m1, m2 = params_pre['m1'], params_pre['m2']
    l1, l2 = params_pre['l1'], params_pre['l2']
    l1_prime, l2_prime = params_pre['l1_prime'], params_pre['l2_prime']
    J1_root, J2_cm = params_pre['J1'], params_pre['J2']
    J1_cm = J1_root - m1 * l1_prime ** 2
    qdot_minus = np.array([0.0, 0.0, dtheta1_pre, dtheta2_pre])
    s1, c1 = np.sin(theta1), np.cos(theta1)
    s2, c2 = np.sin(theta2), np.cos(theta2)
    M = np.zeros((4, 4))
    M[0, 0] = m1 + m2
    M[1, 1] = m1 + m2
    M[0, 2] = -m1 * l1_prime * s1 + m2 * (-l1 * s1)
    M[0, 2] = -m1 * l1_prime * s1 - m2 * l1 * s1
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

    theta1_new_fun = np.arctan2(l2 * c2, -l2 * s2)
    theta2_new_fun = theta1 - np.pi / 2
    dtheta1_new_fun = x[3]  # 注意顺序：x[2]是原dtheta1, x[3]是原dtheta2，根据碰撞后定义交换
    dtheta2_new_fun = x[2]

    return theta1_new_fun, theta2_new_fun, dtheta1_new_fun, dtheta2_new_fun, x[0], x[1]


def kinetic_energy(theta1, theta2, dtheta1, dtheta2, params_fun):
    m2 = params_fun['m2']
    l1 = params_fun['l1']
    l2_prime = params_fun['l2_prime']
    J1 = params_fun['J1']
    J2 = params_fun['J2']

    # 杆1的转动动能
    T_rot1 = 0.5 * J1 * dtheta1 ** 2

    v2_sq = (dtheta1 * l1) ** 2 + (dtheta2 * l2_prime) ** 2 + \
            2 * dtheta1 * l1 * dtheta2 * l2_prime * np.sin(theta2 - theta1)

    T_trans2 = 0.5 * m2 * v2_sq

    T_rot2 = 0.5 * J2 * dtheta2 ** 2

    T = T_rot1 + T_trans2 + T_rot2

    return T


def load_policy_and_env(stance_leg_fun, params1, params2):
    """根据站立腿和Q值返回环境、策略和动作列表"""
    if stance_leg_fun == 1:
        params_fun = params1
        policy_path = 'D:/L&S/Mas/Project/Paper2/l1_stand_continuous_1252.pth'
        actions_fun = [-1, 0, 1]

    else:
        params_fun = params2
        policy_path = 'D:/L&S/Mas/Project/Paper2/l2_stand_continuous_1329.pth'
        actions_fun = [-1, 0, 1]

    env_fun = PendulumEnv(params_fun)

    device_fun = torch.device("cpu")
    nnn_fun = 512
    policy_fun = torch.nn.Sequential(torch.nn.Linear(4, nnn_fun * 2), torch.nn.Tanh(),
                                     torch.nn.Linear(nnn_fun * 2, nnn_fun), torch.nn.Tanh(),
                                     torch.nn.Linear(nnn_fun, 2))
    policy_fun.load_state_dict(torch.load(policy_path, map_location=device_fun))
    policy_fun.to(device_fun)
    policy_fun.eval()  # 设置为评估模式，避免影响 dropout/batch norm

    return env_fun, policy_fun, actions_fun


step = 3
stop_flag = 0
device = torch.device("cpu")
BASE_PARAMS = {
    'g': 9.8, 'dt': 0.01, 'max_torque': 4,
    'target1': 60, 'target2': 30, 'target3': 120, 'target4': -30,
    'theta1_range': 90, 'theta2_range': 90,
    'speed_range': 2, 'settle': 5, 'reward_scale': 5, 'action_value_weight': 0.03
}
PARAMS1 = BASE_PARAMS.copy()
PARAMS1.update({
    'm1': 1.5981, 'm2': 0.6503, 'l1': 0.521, 'l2': 0.481,
    'l1_prime': 0.4114, 'l2_prime': 0.218, 'J1': 0.5, 'J2': 0.105,
    'c1': 0.00047630386389081107, 'c2': 0.0001
})
PARAMS2 = BASE_PARAMS.copy()
PARAMS2.update({
    'm1': 0.6503, 'm2': 1.5981, 'l1': 0.521, 'l2': 0.481,
    'l1_prime': 0.30300000000000005, 'l2_prime': 0.05916,
    'J1': 0.34, 'J2': 0.1, 'c1': 0.001, 'c2': 0.0001
})

W = (PARAMS1['m1'] + PARAMS1['m2']) * PARAMS1['g']
flag = 0
simu_time = 3.2
steps = int(simu_time / PARAMS1['dt'])


class PendulumEnv:
    def __init__(self, param):
        self.param = param
        self.max_torque = param['max_torque']
        self.dt = param['dt']
        self.m1 = param['m1']
        self.m2 = param['m2']
        self.l1 = param['l1']
        self.l2 = param['l2']
        self.l1_s = param['l1_prime']
        self.l2_s = param['l2_prime']
        self.J1 = param['J1']
        self.J2 = param['J2']
        self.c1 = param['c1']
        self.c2 = param['c2']
        self.g = param.get('g', 9.8)
        self.settle = np.deg2rad(param['settle'])
        self.settle2 = self.settle / 5
        self.target1_rad = np.deg2rad(param['target1'])
        self.target2_rad = np.deg2rad(param['target2'])
        self.target3_rad = np.deg2rad(param['target3'])
        self.target4_rad = np.deg2rad(param['target4'])
        self.speed_range = param['speed_range']
        self.rad_theta1_range = np.deg2rad(param['theta1_range'])
        self.rad_theta2_range = np.deg2rad(param['theta2_range'])
        self.reward_scale = param['reward_scale']
        self.action_value_weight = param['action_value_weight']
        self.state = None
        self.next_state = None
        self.reward = None
        self.over = None
        self.y = np.zeros(4)

    def step(self, action_value):
        self.y[0], self.y[1], self.y[2], self.y[3] = self.state
        action_value = np.clip(action_value, -self.max_torque, self.max_torque)
        a_ = np.array([[(self.J1 + self.m2 * self.l1 ** 2),
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
        inverse_a = np.linalg.inv(a_)
        arr = np.array([[-action_value], [action_value]])
        temp = inverse_a @ (b + arr)
        self.y[1] += (temp[0][0] - self.c1 * self.y[1]) * self.dt
        self.y[0] += self.y[1] * self.dt
        self.y[3] += (temp[1][0] - self.c2 * self.y[3]) * self.dt
        self.y[2] += self.y[3] * self.dt
        cond1 = abs(self.y[0] - self.target1_rad) < self.settle and abs(self.y[2] - self.target2_rad) < self.settle
        cond2 = abs(self.y[0] - self.target3_rad) < self.settle2 and abs(self.y[2] - self.target4_rad) < self.settle2
        if cond1:
            self.reward = self.reward_scale * 3
            self.over = True
        elif cond2:
            self.reward = self.reward_scale * 2
            self.over = True
        elif abs(self.y[0] - np.pi / 2) > self.rad_theta1_range or abs(self.y[2]) > self.rad_theta2_range:
            self.reward = -self.reward_scale * 5
            self.over = True
        # elif (self.l1 * np.sin(self.y[0]) - self.l2 * np.cos(self.y[2])) < 0.03:
        #     self.reward = -reward_scale * 5
        #     self.over = True
        else:
            self.reward = 0
            self.over = False
        self.reward += -abs(action_value) * self.action_value_weight
        self.next_state = np.array([self.y[0], self.y[1], self.y[2], self.y[3]])
        self.state = np.copy(self.next_state)
        return self.next_state, self.reward, self.over

    def reset(self):
        theta_1_initial = np.random.uniform(self.target1_rad, self.target3_rad)
        theta_2_initial = np.deg2rad(np.random.uniform(-60, 60))  # rads
        theta_1_dot_initial = np.random.uniform(-self.speed_range, self.speed_range)
        theta_2_dot_initial = np.random.uniform(-self.speed_range, self.speed_range)
        self.state = np.array([theta_1_initial, theta_1_dot_initial, theta_2_initial, theta_2_dot_initial])
        return self.state

    def define(self, theta1, dtheta1, theta2, dtheta2):
        self.state = np.array([theta1, dtheta1, theta2, dtheta2])
        return self.state


N1, N2 = 30, 60
tht1s = np.linspace(BASE_PARAMS['target1'], BASE_PARAMS['target3'], N1) * np.pi / 180
dtht1s = np.linspace(-2, 2, N2)
tht2s = np.linspace(-60, 60, N1) * np.pi / 180
dtht2s = np.linspace(-2, 2, N2)
count = 0
thetas1, thetas2, dthetas1, dthetas2, actions_save = [], [], [], [], []
st_leg_store = []
stance_leg = 1
Energy_PPO = 0
Energy_pre = 0
Energy_new = 0
theta1_new, theta2_new, dtheta1_new, dtheta2_new = 0, 0, 0, 0
init_idx = (27, 3, 15, 9)  # (i, j, k, ll)
y = np.array([tht1s[init_idx[0]], dtht1s[init_idx[1]], tht2s[init_idx[2]], dtht2s[init_idx[3]]])
initial_theta1, initial_theta2 = y[0], y[2]
final_theta1 = 0
final_theta2 = 0
E_push = 0
D = 0
while step != 0:
    if count > 0:
        y = np.array([theta1_new, dtheta1_new, theta2_new, dtheta2_new])
    env, policy, actions = load_policy_and_env(stance_leg, PARAMS1, PARAMS2)
    over = False
    state = env.reset()
    state[:] = y
    env.state = state.copy()
    next_state = state
    count_time = 0
    while not over:
        state_in_net_ = np.array([(state[0] - np.pi / 2) / env.rad_theta1_range, state[1] / env.speed_range,
                                  state[2] / env.rad_theta2_range, state[3] / env.speed_range])
        output = policy(torch.FloatTensor(state_in_net_).reshape(1, 4).to(device))
        mean, log_std = output.chunk(2, dim=1)
        std = torch.exp(log_std)
        dist = torch.distributions.Normal(mean, std)
        action_value = dist.sample().item()
        action_value = np.clip(action_value, -env.max_torque, env.max_torque)
        log_prob = dist.log_prob(torch.tensor([action_value]).to(device)).item()
        next_state, reward, over = env.step(action_value)
        if action_value * (state[3] - state[1]) > 0:
            Energy_PPO += abs(action_value) * abs(state[3] - state[1]) * env.dt
        thetas1.append(state[0] * 180 / np.pi)
        thetas2.append(state[2] * 180 / np.pi)
        dthetas1.append(state[1] * 180 / np.pi)
        dthetas2.append(state[3] * 180 / np.pi)
        actions_save.append(action_value)
        st_leg_store.append(stance_leg)
        state = np.copy(next_state)
    reward1 = abs(state[0] - env.target1_rad) < env.settle and abs(state[2] - env.target2_rad) < env.settle
    reward2 = abs(state[0] - env.target3_rad) < env.settle2 and abs(state[2] - env.target4_rad) < env.settle2
    final_theta1 = state[0]
    final_theta2 = state[2]
    initial_center_x = (env.m1 * env.l1_s * np.cos(initial_theta1) +
                        env.m2 * (env.l1 * np.cos(initial_theta1) + env.l2_s * np.sin(initial_theta2))) / (
                               env.m1 + env.m2)
    final_center_x = (env.m1 * env.l1_s * np.cos(final_theta1) + env.m2 *
                      (env.l1 * np.cos(final_theta1) + env.l2_s * np.sin(final_theta2))) / (env.m1 + env.m2)
    D += abs(final_center_x - initial_center_x)
    if reward1:
        print("碰撞触发！计算碰撞后状态...")
        params = {
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
            state[0], state[2], state[1], state[3], params)
        # 输出结果（转换为度便于观察）
        print("碰撞后新状态（新定义）：")
        print(f"  新站立腿与地面夹角 = {np.rad2deg(theta1_new):.2f}°")
        print(f"  新摆动腿与竖直方向夹角 = {np.rad2deg(theta2_new):.2f}°")
        print(f"  新站立腿角速度 = {dtheta1_new:.4f} rad/s")
        print(f"  新摆动腿角速度 = {dtheta2_new:.4f} rad/s")
        initial_theta1 = theta1_new
        initial_theta2 = theta2_new
        Energy_pre = kinetic_energy(theta1_new, theta2_new, dtheta1_new, dtheta2_new, params)
        if stance_leg == 1:
            dtheta1_new -= 0.88
            stop_flag += 1
            Energy_new = kinetic_energy(theta1_new, theta2_new, dtheta1_new, dtheta2_new, params)
            E_push += Energy_new - Energy_pre
            Energy_PPO += Energy_new - Energy_pre
            print("蹬地后新状态（新定义）：")
            print(f"  新站立腿与地面夹角 = {np.rad2deg(theta1_new):.2f}°")
            print(f"  新摆动腿与竖直方向夹角 = {np.rad2deg(theta2_new):.2f}°")
            print(f"  新站立腿角速度 = {dtheta1_new:.4f} rad/s")
            print(f"  新摆动腿角速度 = {dtheta2_new:.4f} rad/s")
            stance_leg = 2
        elif stance_leg == 2:
            dtheta1_new -= 0.67
            stop_flag += 1
            Energy_new = kinetic_energy(theta1_new, theta2_new, dtheta1_new, dtheta2_new, params)
            E_push += Energy_new - Energy_pre
            Energy_PPO += Energy_new - Energy_pre
            print("蹬地后新状态（新定义）：")
            print(f"  新站立腿与地面夹角 = {np.rad2deg(theta1_new):.2f}°")
            print(f"  新摆动腿与竖直方向夹角 = {np.rad2deg(theta2_new):.2f}°")
            print(f"  新站立腿角速度 = {dtheta1_new:.4f} rad/s")
            print(f"  新摆动腿角速度 = {dtheta2_new:.4f} rad/s")
            stance_leg = 1
        step -= 1
        count += 1
Cmt_active_continuous = Energy_PPO / (W * D)
print("连续力矩Cmt：", Cmt_active_continuous)
theta1_rad = np.array(thetas1) * np.pi / 180
theta2_rad = np.array(thetas2) * np.pi / 180
dtheta1_rad = np.array(dthetas1) * np.pi / 180
dtheta2_rad = np.array(dthetas2) * np.pi / 180
st_leg_arr = np.array(st_leg_store)

T = len(theta1_rad)  # 总时间步数
dt = 0.01  # 与仿真一致

# 系统参数（两组）
params1 = {
    'm_sup': 1.5981,
    'm_swing': 0.6503,
    'l_sup': 0.521,
    'l_swing': 0.481,
    'l_sup_prime': 0.4114,
    'l_swing_prime': 0.218
}
params2 = {
    'm_sup': 0.6503,
    'm_swing': 1.5981,
    'l_sup': 0.521,
    'l_swing': 0.481,
    'l_sup_prime': 0.303,
    'l_swing_prime': 0.05916
}
M_total = 1.5981 + 0.6503

# 初始化全局坐标
sup_global_x = 0.0
sup_global_y = 0.0
x_com_global = np.zeros(T)
y_com_global = np.zeros(T)
p_global = np.zeros(T)

# 递推计算
for k in range(T):
    leg = st_leg_arr[k]
    p = params1 if leg == 1 else params2
    th1 = theta1_rad[k]
    th2 = theta2_rad[k]

    # 髋关节相对支撑点
    hip_rel_x = p['l_sup'] * np.cos(th1)
    hip_rel_y = p['l_sup'] * np.sin(th1)
    hip_global_x = sup_global_x + hip_rel_x
    hip_global_y = sup_global_y + hip_rel_y  # sup_global_y 始终为0

    # 站立腿质心相对
    com_sup_rel_x = p['l_sup_prime'] * np.cos(th1)
    com_sup_rel_y = p['l_sup_prime'] * np.sin(th1)
    # 摆动腿质心相对
    com_swing_rel_x = p['l_sup'] * np.cos(th1) + p['l_swing_prime'] * np.sin(th2)
    com_swing_rel_y = p['l_sup'] * np.sin(th1) - p['l_swing_prime'] * np.cos(th2)
    # 整体质心相对支撑点
    com_rel_x = (p['m_sup'] * com_sup_rel_x + p['m_swing'] * com_swing_rel_x) / M_total
    com_rel_y = (p['m_sup'] * com_sup_rel_y + p['m_swing'] * com_swing_rel_y) / M_total

    x_com_global[k] = sup_global_x + com_rel_x
    y_com_global[k] = sup_global_y + com_rel_y
    p_global[k] = sup_global_x

    # 判断下一步是否切换腿
    if k < T - 1 and st_leg_arr[k + 1] != leg:
        # 计算当前摆动腿末端位置作为新支撑点
        foot_rel_x = p['l_swing'] * np.sin(th2)
        foot_rel_y = - p['l_swing'] * np.cos(th2)
        foot_global_x = hip_global_x + foot_rel_x
        sup_global_x = foot_global_x
        sup_global_y = 0.0

# 总路程（位移绝对值累加，与PPO的D定义一致）
D_path = np.sum(np.abs(np.diff(x_com_global)))
# 平均高度
h = np.mean(y_com_global)
g = 9.8

# ========== LIPM 优化 ==========
N = T - 1  # 控制步数
x_ref = x_com_global
v_ref = np.gradient(x_com_global, dt)  # 数值速度
p_ref = p_global[:N]  # 支撑点参考（最后一个状态无对应控制）

# 权重（可根据跟踪效果调整）
w_x = 1.0
w_v = 0.1
w_u = 0.01
w_du = 0.1

# 离散LIPM动力学
A = np.array([[1.0, dt],
              [g * dt / h, 1.0]])
B = np.array([[0.0],
              [-g * dt / h]])

x = cp.Variable(T)
v = cp.Variable(T)
u = cp.Variable(N)

cost = (w_x * cp.sum_squares(x - x_ref) +
        w_v * cp.sum_squares(v - v_ref) +
        w_u * cp.sum_squares(u - p_ref) +
        w_du * cp.sum_squares(u[1:] - u[:-1]))

constraints = [x[0] == x_ref[0], v[0] == v_ref[0]]
for k in range(N):
    constraints += [x[k + 1] == A[0, 0] * x[k] + A[0, 1] * v[k] + B[0, 0] * u[k],
                    v[k + 1] == A[1, 0] * x[k] + A[1, 1] * v[k] + B[1, 0] * u[k]]

prob = cp.Problem(cp.Minimize(cost), constraints)
prob.solve(solver=cp.SCS, verbose=False, eps_abs=1e-4, eps_rel=1e-4)

if prob.status not in ["optimal", "optimal_inaccurate"]:
    print("优化失败，使用参考值")
    x_opt, v_opt, u_opt = x_ref, v_ref, p_ref
else:
    x_opt = x.value
    v_opt = v.value
    u_opt = u.value

# 计算LIPM能耗（仅正功）
# 支撑点速度（中心差分）
# 支撑力水平分量 Fx = M*g/h * (x - p)
Fx = (M_total * g / h) * (x_opt[:N] - u_opt)

# 支撑点速度（中心差分）
v_p = np.zeros_like(u_opt)
if N > 2:
    v_p[1:-1] = (u_opt[2:] - u_opt[:-2]) / (2 * dt)
if N > 1:
    v_p[0] = (u_opt[1] - u_opt[0]) / dt
    v_p[-1] = (u_opt[-1] - u_opt[-2]) / dt

# 瞬时功率
P = Fx * v_p

# 能耗累加：仅当该控制步内未发生腿切换时计入正功
E_lipm = 0.0
for k in range(N):          # k 对应从时刻 k 到 k+1 的控制步
    if st_leg_arr[k] == st_leg_arr[k+1]:   # 未切换腿，连续运动
        E_lipm += dt * max(0.0, P[k])
    # 若切换，则该步支撑点瞬间移动，不计入能耗

# ----------------- 与 PPO 的 Cmt 对比 -----------------
Cmt_lipm = E_lipm / (M_total * g * D_path)
Cmt_ppo = Cmt_active_continuous  # 之前计算的PPO的Cmt

print(f"LIPM 能耗 E = {E_lipm:.4f} J")
print(f"总路程 D = {D_path:.4f} m")
print(f"LIPM 的 Cmt = {Cmt_lipm:.6f}")
print(f"PPO 的 Cmt  = {Cmt_ppo:.6f}")
print(f"比值 LIPM/PPO = {Cmt_lipm / Cmt_ppo:.3f}")

# 绘图（可选）
t = np.arange(T) * dt
plt.figure(figsize=(12, 8))
plt.subplot(3, 1, 1)
plt.plot(t, x_com_global, label='双摆质心 x')
plt.plot(t, x_opt, '--', label='LIPM 优化 x')
plt.ylabel('x (m)')
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(t[:-1], p_global[:-1], label='双摆支撑点')
plt.plot(t[:-1], u_opt, '--', label='LIPM 优化 p')
plt.ylabel('p (m)')
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(t[:-1], P, label='功率 P')
plt.xlabel('t (s)')
plt.ylabel('Power (W)')
plt.legend()
plt.tight_layout()
plt.show()
