import os
import torch
import numpy as np
import matplotlib.pyplot as plt

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
        policy_path = 'D:/L&S/Mas/Project/Paper2/60,30(0.33m),0.01m/l1_stand_continuous_1207.pth'
        actions_fun = [-1, 0, 1]

    else:
        params_fun = params2
        policy_path = 'D:/L&S/Mas/Project/Paper2/60,30(0.33m),0.01m/l2_stand_continuous_1199.pth'
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
    'target1': 61.8, 'target2': 31.7, 'target3': 118.2, 'target4': -31.7,
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
init_idx = (27, 0, 15, 21)  # (i, j, k, ll)
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
            dtheta1_new -= 1
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
            dtheta1_new -= 0.82
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

print("\n开始分段时变LQR跟踪仿真...")

# ---------- 1. 定义系统参数 ----------
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
g = 9.8
dt = 0.01

# 从存储的列表中提取参考轨迹（已完成）
theta1_rad = np.array(thetas1) * np.pi / 180
theta2_rad = np.array(thetas2) * np.pi / 180
dtheta1_rad = np.array(dthetas1) * np.pi / 180
dtheta2_rad = np.array(dthetas2) * np.pi / 180
actions_arr = np.array(actions_save)          # 原始动作数组，长度 = T_steps
st_leg_ref = np.array(st_leg_store)       # 支撑腿标志

T_steps = len(theta1_rad)                 # 状态数量
u_ref = actions_arr[:T_steps-1]            # 控制序列，长度 T_steps-1
print(f"状态数量: {T_steps}, 控制数量: {len(u_ref)}")

# 构建参考状态矩阵（T_steps 个状态）
x_ref = np.zeros((T_steps, 4))
x_ref[:, 0] = theta1_rad
x_ref[:, 1] = dtheta1_rad
x_ref[:, 2] = theta2_rad
x_ref[:, 3] = dtheta2_rad

# 找到支撑腿切换点（即碰撞时刻的索引）
change_idx = np.where(np.diff(st_leg_ref) != 0)[0] + 1  # 切换后的第一帧索引
print(f"检测到腿切换时刻（碰撞后第一帧）: {change_idx}")

# 将轨迹分段：从0到change_idx[0]-1为第一段，
# 从change_idx[0]到change_idx[1]-1为第二段，
# 从change_idx[1]到结束为第三段。
segments = []
start = 0
for idx in change_idx:
    segments.append((start, idx))
    start = idx
segments.append((start, T_steps))  # 最后一段
print(f"轨迹分为 {len(segments)} 段，区间为: {segments}")


# 定义数值线性化函数（同前，略）
def numerical_AB(x, u, env, dt=0.01, eps=1e-6):
    env.state = x.copy()
    x_nom, _, _ = env.step(u)
    A = np.zeros((4, 4))
    for i in range(4):
        x_pert = x.copy()
        x_pert[i] += eps
        env.state = x_pert
        x_next_pert, _, _ = env.step(u)
        A[:, i] = (x_next_pert - x_nom) / eps
    B = np.zeros((4, 1))
    u_pert = u + eps
    env.state = x.copy()
    x_next_pert_u, _, _ = env.step(u_pert)
    B[:, 0] = (x_next_pert_u - x_nom) / eps
    return A, B


# 设置LQR权重
Q = np.diag([10.0, 1.0, 10.0, 1.0])
R = np.array([[0.01]])
Qf = Q.copy()

# 准备存储LQR跟踪结果
x_lqr_total = np.zeros((T_steps, 4))
u_lqr_total = np.zeros(T_steps - 1)
# 初始状态直接用参考轨迹
x_lqr_total[0] = x_ref[0].copy()

# 总能耗
Energy_LQR = 0.0
dt = 0.01

# 对每一段分别处理
for seg_idx, (seg_start, seg_end) in enumerate(segments):
    print(f"\n处理第 {seg_idx+1} 段，时间步 [{seg_start}, {seg_end})")
    seg_len = seg_end - seg_start
    if seg_len <= 1:
        continue

    # 该段内的支撑腿类型（取第一帧即可）
    leg = st_leg_ref[seg_start]

    env_seg, _, _ = load_policy_and_env(leg, PARAMS1, PARAMS2)

    # 该段内的参考轨迹
    x_ref_seg = x_ref[seg_start:seg_end]
    u_ref_seg = u_ref[seg_start:seg_end-1]   # 控制序列长度少1

    # 计算该段内的线性化矩阵 A_k, B_k
    A_list = []
    B_list = []
    for k in range(seg_len - 1):
        env_seg.state = x_ref_seg[k].copy()
        A, B = numerical_AB(x_ref_seg[k], u_ref_seg[k], env_seg, dt)
        A_list.append(A)
        B_list.append(B)

    # 逆向递推求解该段的时变增益
    P = Qf
    K_list = [None] * (seg_len - 1)
    for k in reversed(range(seg_len - 1)):
        A = A_list[k]
        B = B_list[k]
        S = R + B.T @ P @ B
        K = np.linalg.solve(S, B.T @ P @ A)
        K_list[k] = K
        P = Q + A.T @ P @ (A - B @ K)

    # 对该段进行跟踪仿真
    # 该段起始状态：如果是第一段，直接用x_lqr_total[seg_start]（即参考初值）
    # 如果是后续段，则起始状态应设为参考轨迹中该段的第一帧（因为碰撞瞬间状态由参考给出，不控制）
    if seg_start == 0:
        x_lqr_total[seg_start] = x_ref[seg_start].copy()
    else:
        # 直接将上一段结束后的状态覆盖为参考轨迹的该段起始（模拟碰撞瞬移）
        x_lqr_total[seg_start] = x_ref[seg_start].copy()

    env_seg.state = x_lqr_total[seg_start].copy()

    for k in range(seg_len - 1):
        global_k = seg_start + k
        # 计算控制误差（基于该段参考）
        delta_x = x_lqr_total[global_k] - x_ref_seg[k]
        delta_u = -K_list[k] @ delta_x
        u_lqr = u_ref_seg[k] + delta_u.item()
        u_lqr_total[global_k] = u_lqr

        # 应用控制
        env_seg.state = x_lqr_total[global_k].copy()
        x_next, _, _ = env_seg.step(u_lqr)
        x_lqr_total[global_k+1] = x_next.copy()

        # 计算能耗
        rel_vel = x_lqr_total[global_k][3] - x_lqr_total[global_k][1]
        if u_lqr * rel_vel > 0:
            Energy_LQR += abs(u_lqr) * abs(rel_vel) * dt

print("\n分段LQR跟踪完成。")

M_total = 1.5981 + 0.6503
sup_global_x = 0.0
x_com_global = np.zeros(T_steps)
for k in range(T_steps):
    leg = st_leg_ref[k]
    if leg == 1:
        p = params1
    else:
        p = params2
    th1 = x_ref[k, 0]
    th2 = x_ref[k, 2]
    # 站立腿质心相对
    com_sup_rel_x = p['l_sup_prime'] * np.cos(th1)
    # 摆动腿质心相对
    com_swing_rel_x = p['l_sup'] * np.cos(th1) + p['l_swing_prime'] * np.sin(th2)
    com_rel_x = (p['m_sup'] * com_sup_rel_x + p['m_swing'] * com_swing_rel_x) / M_total
    x_com_global[k] = sup_global_x + com_rel_x
    # 判断下一步是否切换腿（用于更新支撑点）
    if k < T_steps - 1 and st_leg_ref[k+1] != leg:
        # 计算当前摆动腿末端位置作为新支撑点
        hip_rel_x = p['l_sup'] * np.cos(th1)
        hip_rel_y = p['l_sup'] * np.sin(th1)
        foot_rel_x = p['l_swing'] * np.sin(th2)
        foot_global_x = sup_global_x + hip_rel_x + foot_rel_x
        sup_global_x = foot_global_x

D_path = np.sum(np.abs(np.diff(x_com_global)))

# 计算Cmt
Energy_LQR += E_push
g = 9.8
W = M_total * g
Cmt_lqr = Energy_LQR / (W * D_path)
Cmt_ppo = Cmt_active_continuous  # 假设Energy_PPO已定义

# 输出结果
print(f"\n====== 能耗对比 ======")
print(f"总路程 D = {D_path:.4f} m")
print(f"PPO 能耗 E = {Energy_PPO:.4f} J")
print(f"LQR 能耗 E = {Energy_LQR:.4f} J")
print(f"PPO 的 Cmt  = {Cmt_ppo:.6f}")
print(f"LQR 的 Cmt  = {Cmt_lqr:.6f}")
print(f"比值 LQR/PPO = {Cmt_lqr / Cmt_ppo:.3f}")

t = np.arange(T_steps) * dt
plt.figure(figsize=(14, 10))

plt.subplot(3, 1, 1)
plt.plot(t, x_ref[:, 0] * 180/np.pi, label='Ref θ1')
plt.plot(t, x_lqr_total[:, 0] * 180/np.pi, '--', label='LQR θ1')
plt.ylabel('θ1 (deg)')
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(t, x_ref[:, 2] * 180/np.pi, label='Ref θ2')
plt.plot(t, x_lqr_total[:, 2] * 180/np.pi, '--', label='LQR θ2')
plt.ylabel('θ2 (deg)')
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(t[:-1], u_ref, label='Ref u')
plt.plot(t[:-1], u_lqr_total, '--', label='LQR u')
plt.xlabel('t (s)')
plt.ylabel('Torque (Nm)')
plt.legend()
plt.tight_layout()
plt.show()