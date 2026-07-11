import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import h5py
import torch
import imageio
import numpy as np
from numba import njit
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


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
        a = np.array([[(self.J1 + self.m2 * self.l1**2),
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
        reward2 = abs(self.y[0] - np.deg2rad(target3)) < settle and abs(self.y[2] - np.deg2rad(target4)) < settle
        if reward1:
            self.reward = reward_scale * 3
            success.append(1)
            self.over = True
        elif reward2:
            self.reward = reward_scale * 2
            success.append(1)
            # self.over = True
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
        a = np.array([[(self.J1 + self.m2 * self.l1**2),
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
        reward2 = abs(self.y[0] - np.deg2rad(target3)) < settle and abs(self.y[2] - np.deg2rad(target4)) < settle
        if reward1:
            self.reward = reward_scale * 3
            success.append(1)
            self.over = True
        elif reward2:
            self.reward = reward_scale * 2
            success.append(1)
            # self.over = True
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
    Q = np.array(h5f['working_save_passive'][:])
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
frames = []
success = []
y = np.zeros(4)
y_old = np.zeros(4)
y[0] = 116.08 * np.pi / 180
y[1] = -1.3755
y[2] = -27.08 * np.pi / 180
y[3] = -1.5888
# y[0] = tht1s[19]
# y[1] = dtht1s[11]
# y[2] = tht2s[15]
# y[3] = dtht2s[7]
i = np.argmin(abs(y[0] - tht1s))
j = np.argmin(abs(y[1] - dtht1s))
k = np.argmin(abs(y[2] - tht2s))
ll = np.argmin(abs(y[3] - dtht2s))
if Q[i, j, k, ll] == -1:
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
thetas1 = []
thetas2 = []
dthetas1 = []
dthetas2 = []
actions_1 = []
thetas1.append(y[0] * 180 / np.pi)
thetas2.append(y[2] * 180 / np.pi)
dthetas1.append(y[1] * 180 / np.pi)
dthetas2.append(y[3] * 180 / np.pi)
# count += 1
# count_time += 1
while not over:
    state_in_net_ = np.array([(state[0] - np.pi / 2) / rad_theta1_range,
                              state[1] / speed_range, state[2] / rad_theta2_range, state[3] / speed_range])
    prob = policy(torch.FloatTensor(state_in_net_).reshape(1, 4))[0].cpu().detach().numpy()
    action_index = actions[np.argmax(prob)]
    next_state, reward, over = env.step(action_index)
    plt.clf()
    plt.axis('equal')
    plt.xlabel('X [m]')
    plt.ylabel('Y [m]')
    # plt.title("4Nm, l=0.33m")
    plt.ylim([-0.12, 0.35])
    plt.xlim([-0.45, 0.45])
    plt.tight_layout()
    plt.plot([-0.33, 0.33], [0, 0], 'k--')
    plt.plot([0, l1_draw * np.cos(np.deg2rad(target1))], [0, l1_draw * np.sin(np.deg2rad(target1))], "k:")
    plt.plot([l1_draw * np.cos(np.deg2rad(target1)), l1_draw * np.cos(np.deg2rad(target1)) + l2_draw * np.sin(np.deg2rad(target2))],
             [l1_draw * np.sin(np.deg2rad(target1)), l1_draw * np.sin(np.deg2rad(target1)) - l2_draw * np.cos(np.deg2rad(target2))],
             "k:")
    plt.plot([0, l1_draw * np.cos(next_state[0])], [0, l1_draw * np.sin(next_state[0])], "m-")
    plt.plot([l1_draw * np.cos(next_state[0]), l1_draw * np.cos(next_state[0]) + l2_draw * np.sin(next_state[2])],
             [l1_draw * np.sin(next_state[0]), l1_draw * np.sin(next_state[0]) - l2_draw * np.cos(next_state[2])], "m-")
    if action_index == 0:
        plt.text(l1_draw * np.cos(next_state[0]), l1_draw * np.sin(next_state[0]), '0', color="y", fontsize=20)
    elif action_index == 1:
        plt.text(l1_draw * np.cos(next_state[0]), l1_draw * np.sin(next_state[0]), '1', color="r", fontsize=20)
    thetas1.append(state[0] * 180 / np.pi)
    thetas2.append(state[2] * 180 / np.pi)
    dthetas1.append(state[1] * 180 / np.pi)
    dthetas2.append(state[3] * 180 / np.pi)
    actions_1.append(actions[action_index] * torque)
    plt.show()
    plt.pause(1e-5)
    state = np.copy(next_state)
    plt.savefig(f'D:/L&S/Mas/Project/Paper2/temp/img_{count}.png', transparent=False,
                facecolor='white')
    image = imageio.imread(f'D:/L&S/Mas/Project/Paper2/temp/img_{count}.png')
    count += 1
    count_time += 1
    frames.append(image)
imageio.mimsave('D:/L&S/Mas/Project/Paper2/temp/example-1.gif', frames, duration=60, loop=0)

plt.figure(figsize=(4, 4), dpi=200)
plt.xticks([])
plt.yticks([])
plt.axis('off')
# plt.title("4Nm, l=0.33M")
horizontal_line_28 = [25] * (count_time + 1)
horizontal_line_32 = [35] * (count_time + 1)  # * np.pi / 180
horizontal_line_58 = [55] * (count_time + 1)  # * np.pi / 180
horizontal_line_62 = [65] * (count_time + 1)
start_point = 0
plt.subplot(3, 1, 1)
plt.ylabel(r'$\theta$ [$^\circ$]')
# plt.ylim([-1, 2.5])
plt.xlim([0, count_time * dt])
plt.plot(np.array(range(count_time + 1)) * dt, thetas1, 'b-+', label=r'$\theta_1$')
plt.plot(np.array(range(count_time + 1)) * dt, thetas2, 'r-', label=r'$\theta_2$')
plt.gca().add_patch(Rectangle((0, horizontal_line_28[start_point]), dt * (count_time + 1),
                              horizontal_line_32[start_point] - horizontal_line_28[start_point],
                              edgecolor='none', facecolor=[1, 0, 0], alpha=0.2))
plt.gca().add_patch(Rectangle((0, horizontal_line_58[start_point]), dt * (count_time + 1),
                              horizontal_line_62[start_point] - horizontal_line_58[start_point],
                              edgecolor='none', facecolor=[0, 0, 1], alpha=0.2))
# plt.plot([dt * (8 - 1), dt * (8 - 1)], [-100, 230], 'k:')
plt.legend(ncol=2)
plt.xticks([])
plt.tight_layout()
plt.subplot(3, 1, 2)
plt.ylabel(r'$\dot{\theta}$  [$^\circ$/s]')
plt.xlim([0, count_time * dt])
plt.plot(np.array(range(count_time + 1)) * dt, dthetas1, 'b-', label=r'$\dot{\theta}_1$')
plt.plot(np.array(range(count_time + 1)) * dt, dthetas2, 'r-', label=r'$\dot{\theta}_2$')
plt.legend(ncol=2)
plt.xticks([])
# plt.plot([dt * (8 - 1), dt * (8 - 1)], [-2000, 400], 'k:')
plt.tight_layout()
plt.subplot(3, 1, 3)
plt.xlabel('Time [s]')
plt.ylabel(r'$\tau$  [Nm]')
plt.ylim([-(torque + 0.5), torque + 0.5])
# plt.plot([dt * (8 - 1), dt * (8 - 1)], [-5.5, 5.5], 'k:')
plt.xlim([0, count_time * dt])
plt.step(np.array(range(count_time)) * dt, actions_1, 'k-', label='Torque')
plt.tight_layout()
# plt.savefig("C:/Users/Admin/Desktop/Figure2.svg")

