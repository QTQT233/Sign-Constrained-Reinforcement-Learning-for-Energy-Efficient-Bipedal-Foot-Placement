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

    v2_sq = (dtheta1 * l1) ** 2 + (dtheta2 * l2_prime) ** 2 +\
            2 * dtheta1 * l1 * dtheta2 * l2_prime * np.sin(theta2 - theta1)

    T_trans2 = 0.5 * m2 * v2_sq

    T_rot2 = 0.5 * J2 * dtheta2 ** 2

    T = T_rot1 + T_trans2 + T_rot2

    return T


def load_policy_and_env(stance_leg_fun, params1, params2):
    """根据站立腿和Q值返回环境、策略和动作列表"""
    if stance_leg_fun == 1:
        params_fun = params1
        policy_path = 'D:/L&S/Mas/Project/Paper2/l1_stand(-1,0,1)_1417.pth'
        actions_fun = [-1, 0, 1]

    else:
        params_fun = params2
        policy_path = 'D:/L&S/Mas/Project/Paper2/l2_stand(-1,0,1)_1539.pth'
        actions_fun = [-1, 0, 1]

    env_fun = PendulumEnv(params_fun)

    device_fun = torch.device("cpu")
    nnn_fun = 256
    policy_fun = torch.nn.Sequential(
        torch.nn.Linear(4, nnn_fun * 2), torch.nn.Tanh(),
        torch.nn.Linear(nnn_fun * 2, nnn_fun), torch.nn.Tanh(),
        torch.nn.Linear(nnn_fun, 3), torch.nn.Softmax(dim=1)
    )
    policy_fun.load_state_dict(torch.load(policy_path, map_location=device_fun))
    policy_fun.to(device_fun)
    policy_fun.eval()  # 设置为评估模式，避免影响 dropout/batch norm

    return env_fun, policy_fun, actions_fun


def handle_collision(state_fun, stance_leg_fun, params_fun, energy_ac_discrete_fun):
    """处理碰撞后的状态更新、能量计算和打印"""
    print("碰撞触发！计算碰撞后状态...")
    print(f"碰撞前：θ1={np.rad2deg(state_fun[0]):.2f}°, θ2={np.rad2deg(state_fun[2]):.2f}°, "
          f"ω1={state_fun[1]:.4f}, ω2={state_fun[3]:.4f}")

    theta1_new_fun, theta2_new_fun, dtheta1_new_fun, dtheta2_new_fun, _, _ = collision_dynamics_full(
        state_fun[0], state_fun[2], state_fun[1], state_fun[3], params_fun)

    print(f"碰撞后（新定义）：θ1={np.rad2deg(theta1_new_fun):.2f}°, θ2={np.rad2deg(theta2_new_fun):.2f}°, "
          f"ω1={dtheta1_new_fun:.4f}, ω2={dtheta2_new_fun:.4f}")

    energy_pre_fun = kinetic_energy(theta1_new_fun, theta2_new_fun, dtheta1_new_fun, dtheta2_new_fun, params_fun)
    if stance_leg_fun == 1:
        dtheta1_new_fun -= 0.73
    else:
        dtheta1_new_fun -= 1.06
    energy_new_fun = kinetic_energy(theta1_new_fun, theta2_new_fun, dtheta1_new_fun, dtheta2_new_fun, params_fun)
    energy_ac_discrete_fun += energy_new_fun - energy_pre_fun

    print(f"蹬地后：ω1={dtheta1_new_fun:.4f}, ω2={dtheta2_new_fun:.4f}")
    return theta1_new_fun, theta2_new_fun, dtheta1_new_fun, dtheta2_new_fun, energy_ac_discrete_fun


step = 3
stop_flag = 0

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
        self.action = np.array([-self.max_torque, 0, self.max_torque])

    def step(self, act_index):
        self.y[0], self.y[1], self.y[2], self.y[3] = self.state
        action_value = self.action[act_index]
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
stance_leg = 1
Energy_ac_discrete = 0
Energy_pre = 0
Energy_new = 0
theta1_new, theta2_new, dtheta1_new, dtheta2_new = 0, 0, 0, 0
init_idx = (27, 1, 15, 10)     # (i, j, k, ll)
y = np.array([tht1s[init_idx[0]], dtht1s[init_idx[1]], tht2s[init_idx[2]], dtht2s[init_idx[3]]])
initial_theta1, initial_theta2 = y[0], y[2]
final_theta1 = 0
final_theta2 = 0
D = 0
actions = [-1, 0, 1]
count_step = 0
landing_residuals = []
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
        prob = policy(torch.FloatTensor(state_in_net_).reshape(1, 4))[0].cpu().detach().numpy()
        action_index = np.argmax(prob)
        a = actions[action_index]
        if a * (state[3] - state[1]) > 0:
            Energy_ac_discrete += abs(a * env.max_torque) * abs(state[3] - state[1]) * env.dt
        next_state, reward, over = env.step(action_index)
        thetas1.append(state[0] * 180 / np.pi)
        thetas2.append(state[2] * 180 / np.pi)
        dthetas1.append(state[1] * 180 / np.pi)
        dthetas2.append(state[3] * 180 / np.pi)
        actions_save.append(actions[action_index] * env.max_torque)
        state = np.copy(next_state)
        count_step += 1

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
        actual_foot_x = env.l1 * np.cos(final_theta1) + env.l2 * np.sin(final_theta2)
        target_foot_x = (
            env.l1 * np.cos(env.target1_rad)
            + env.l2 * np.sin(env.target2_rad)
        )
        landing_residuals.append(actual_foot_x - target_foot_x)
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
        theta1_new, theta2_new, dtheta1_new, dtheta2_new, Energy_ac_discrete = handle_collision(
            state, stance_leg, params, Energy_ac_discrete)
        initial_theta1, initial_theta2 = theta1_new, theta2_new
        stance_leg = 2 if stance_leg == 1 else 1
        step -= 1
        count += 1
Cmt_ac_discrete = Energy_ac_discrete / (W * D)
print("离散主动力矩Cmt：", Cmt_ac_discrete)
print("Time:", count_step * 0.01)
landing_mae = (
    float(np.mean(np.abs(landing_residuals)))
    if landing_residuals
    else float("nan")
)
print("Landing_MAE:", landing_mae)