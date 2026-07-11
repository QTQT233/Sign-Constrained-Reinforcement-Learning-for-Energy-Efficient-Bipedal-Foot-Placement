import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import imageio
import torch
import numpy as np
from tqdm import tqdm
import random
import matplotlib.pyplot as plt

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Running on the GPU")
else:
    device = torch.device("cpu")
    print("Running on the CPU")
torch.cuda.set_device(device)
nnn = 512
model = torch.nn.Sequential(torch.nn.Linear(4, nnn * 2), torch.nn.ReLU(),
                            torch.nn.Linear(nnn * 2, nnn), torch.nn.ReLU(),
                            torch.nn.Linear(nnn, 1))
for module in model.modules():
    if isinstance(module, torch.nn.Linear):
        torch.nn.init.orthogonal_(module.weight)
policy = torch.nn.Sequential(torch.nn.Linear(4, nnn * 2), torch.nn.Tanh(),
                             torch.nn.Linear(nnn * 2, nnn), torch.nn.Tanh(),
                             torch.nn.Linear(nnn, 2))
for module in policy.modules():
    if isinstance(module, torch.nn.Linear):
        torch.nn.init.orthogonal_(module.weight)
policy.load_state_dict(
    torch.load('D:/L&S/Mas/Project/Paper1/Check_continuous/Policy_Net_Pytorch(-1,0,1)_1560_continuous_old_form.pth'))
# policy.load_state_dict(
#     torch.load('D:/L&S/Mas/Project/Paper2/l1_stand_continuous_452.pth'))
# model.load_state_dict(
#     torch.load('D:/L&S/Mas/Project/Paper2/l1_stand_continuous_critic_452.pth'))
model.to(device)
policy.to(device)
# model.train()
optimizer_value = torch.optim.Adam(model.parameters(), lr=1e-4)
optimizer_policy = torch.optim.Adam(policy.parameters(), lr=1e-4)
loss_fn = torch.nn.MSELoss()

# 系统参数
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
gamma = 1
gamma1 = 1
g = 9.8
dt = 0.01
torque = 4

episode = 50000
critic_training_times = 20
critic_training_steps = 50
actor_training_times = 10
playing_times = 2000
concentrated_sample_times = 15
batch_size = 5000
reward_scale = 5
action_value_weight = 0.04
policy_entropy_coefficient = 0.001

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
        # self.count = 0
        self.y = np.zeros(4)

    def step(self, action_value):
        self.y[0], self.y[1], self.y[2], self.y[3] = self.state
        action_value = np.clip(action_value, -self.max_torque, self.max_torque)
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
        # self.count += 1
        reward1 = abs(self.y[0] - np.deg2rad(target1)) < settle and abs(self.y[2] - np.deg2rad(target2)) < settle
        reward2 = abs(self.y[0] - np.deg2rad(target3)) < settle and abs(self.y[2] - np.deg2rad(target4)) < settle
        if reward1:
            self.reward = reward_scale * 3
            success.append(1)
            self.over = True
        elif reward2:
            self.reward = reward_scale * 2
            success.append(1)
            self.over = True
        elif abs(self.y[0] - np.pi / 2) > rad_theta1_range or abs(self.y[2]) > rad_theta2_range:
            self.reward = -reward_scale * 5
            self.over = True
        # elif self.count > 150:
        #     self.reward = -reward_scale * 6
        #     self.over = True
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
        # self.count = 0
        theta_1_initial = np.deg2rad(np.random.uniform(target1, target3))
        theta_2_initial = np.deg2rad(np.random.uniform(-60, 60))  # rads
        theta_1_dot_initial = np.random.uniform(-speed_range, speed_range)
        theta_2_dot_initial = np.random.uniform(-speed_range, speed_range)
        self.state = np.array([theta_1_initial, theta_1_dot_initial, theta_2_initial, theta_2_dot_initial])
        return self.state

    def define(self, theta1, dtheta1, theta2, dtheta2):
        self.state = np.array([theta1, dtheta1, theta2, dtheta2])
        return self.state


env = PendulumEnv()


def play():
    global experience_buffer_for_policy, experience_buffer_for_value, rad_theta1_range, speed_range, rad_theta2_range
    state_ = env.reset()
    over_ = False
    experience_buffer_ = []
    while not over_:
        state_in_net_ = np.array(
            [(state_[0] - np.pi / 2) / rad_theta1_range, state_[1] / speed_range, state_[2] / rad_theta2_range,
             state_[3] / speed_range])
        output = policy(torch.FloatTensor(state_in_net_).reshape(1, 4).to(device))
        mean, log_std = output.chunk(2, dim=1)
        std = torch.exp(log_std)
        dist = torch.distributions.Normal(mean, std)
        action_value = dist.sample().item()
        action_value = np.clip(action_value, -torque, torque)  # 限制在-5到5之间
        log_prob = dist.log_prob(torch.tensor([action_value]).to(device)).item()

        next_state_, reward_, over_ = env.step(action_value)
        state_ = np.copy(next_state_)
        next_state_[0], next_state_[1], next_state_[2], next_state_[3] = (
            (next_state_[0] - np.pi / 2) / rad_theta1_range, next_state_[1] / speed_range,
            next_state_[2] / rad_theta2_range, next_state_[3] / speed_range)
        experience_buffer_.append([state_in_net_, action_value, reward_, next_state_, over_, log_prob, 0])
        if over_:
            delta_ = []
            target_value_ = (1 - torch.FloatTensor(
                np.array([experience_buffer_[i][4] for i in range(len(experience_buffer_))])).reshape(-1, 1).to(
                device)) * model(
                torch.FloatTensor(np.array([experience_buffer_[i][3] for i in range(len(experience_buffer_))])).to(
                    device)) + torch.FloatTensor(
                np.array([experience_buffer_[i][2] for i in range(len(experience_buffer_))]).reshape(-1, 1)).to(
                device)
            current_value_ = model(
                torch.FloatTensor(np.array([experience_buffer_[i][0] for i in range(len(experience_buffer_))])).to(
                    device))
            target_value_ = (target_value_ - current_value_).reshape(-1, ).cpu().detach().numpy()
            for _i in range(len(experience_buffer_)):
                s = target_value_[_i]
                for _j in range(_i, len(target_value_)):
                    s += target_value_[_j] * gamma1 ** (_j - _i)
                delta_.append(s)
            for _i in range(len(experience_buffer_)):
                experience_buffer_[_i][-1] = delta_[_i]
            experience_buffer_for_policy.extend(experience_buffer_)
            experience_buffer_for_value.extend(experience_buffer_)
            for _i in range(concentrated_sample_times):
                experience_buffer_for_value.append([experience_buffer_[-1][0],
                                                    experience_buffer_[-1][1],
                                                    experience_buffer_[-1][2],
                                                    experience_buffer_[-1][3],
                                                    experience_buffer_[-1][4],
                                                    experience_buffer_[-1][5],
                                                    experience_buffer_[-1][6]])


ini_b = 0
for epoch in range(episode):
    success = []
    experience_buffer_for_policy = []
    experience_buffer_for_value = []
    for _ in tqdm(range(playing_times)):
        play()
    print(f"the {epoch + 1} episode")
    print(f"success {len(success)} times for {playing_times} agents ")
    if epoch > 1:
        ini_a = len(success)
        if ini_a > ini_b:
            torch.save(policy.state_dict(),
                       f'D:/L&S/Mas/Project/Paper2/l1_stand_continuous_{ini_a}.pth')
            torch.save(model.state_dict(),
                       f'D:/L&S/Mas/Project/Paper2/l1_stand_continuous_critic_{ini_a}.pth')
            if os.path.exists(
                    f'D:/L&S/Mas/Project/Paper2/l1_stand_continuous_{ini_b}.pth'):
                os.remove(f'D:/L&S/Mas/Project/Paper2/l1_stand_continuous_{ini_b}.pth')
            if os.path.exists(
                    f'D:/L&S/Mas/Project/Paper2/l1_stand_continuous_critic_{ini_b}.pth'):
                os.remove(f'D:/L&S/Mas/Project/Paper2/l1_stand_continuous_critic_{ini_b}.pth')
            ini_b = ini_a

    for ii in range(critic_training_times):
        experience_buffer_proxy = random.sample(experience_buffer_for_value, batch_size)
        state = torch.FloatTensor(
            np.array([experience_buffer_proxy[i][0] for i in range(len(experience_buffer_proxy))])).to(device)
        reward = torch.FloatTensor(
            np.array([experience_buffer_proxy[i][2] for i in range(len(experience_buffer_proxy))]).reshape(-1, 1)).to(
            device)
        next_state = torch.FloatTensor(
            np.array([experience_buffer_proxy[i][3] for i in range(len(experience_buffer_proxy))])).to(device)
        over = torch.FloatTensor(
            np.array([experience_buffer_proxy[i][4] for i in range(len(experience_buffer_proxy))]).reshape(-1, 1)).to(
            device)

        for i in range(critic_training_steps):
            value = model(state).to(device)
            with torch.no_grad():
                target = model(next_state)
            target = target.to(device)
            target = target * gamma * (1 - over) + reward
            loss = loss_fn(value, target).to(device)
            print(f"\r{loss}", end=" ")
            optimizer_value.zero_grad()
            loss.backward()
            optimizer_value.step()
    state = torch.FloatTensor(
        np.array([experience_buffer_for_policy[i][0] for i in range(len(experience_buffer_for_policy))])).to(device)
    old_log_prob = torch.FloatTensor(
        np.array([experience_buffer_for_policy[i][5] for i in range(len(experience_buffer_for_policy))]).reshape(-1,
                                                                                                                 1)).to(
        device)
    action_current = torch.FloatTensor(
        np.array([experience_buffer_for_policy[i][1] for i in range(len(experience_buffer_for_policy))]).reshape(-1,
                                                                                                                 1)).to(
        device)
    delta = np.array([experience_buffer_for_policy[i][6] for i in range(len(experience_buffer_for_policy))]).reshape(-1,
                                                                                                                     1)
    delta = (delta - np.mean(delta)) / np.std(delta)
    delta = torch.FloatTensor(delta).to(device)
    for ii in range(actor_training_times):
        output = policy(state)
        mean, log_std = output.chunk(2, dim=1)
        std = torch.exp(log_std)
        dist = torch.distributions.Normal(mean, std)
        new_log_prob = dist.log_prob(action_current)
        ratio = torch.exp(new_log_prob - old_log_prob)
        loss1 = ratio * delta
        loss2 = torch.clamp(ratio, 1 - 0.2, 1 + 0.2) * delta  # 裁剪范围设为[0.8, 1.2]
        entropy = dist.entropy()
        loss = -torch.min(loss1, loss2).mean() - policy_entropy_coefficient * entropy.mean()
        loss.backward()
        optimizer_policy.step()
        optimizer_policy.zero_grad()
    torch.save(policy.state_dict(),
               f'D:/L&S/Mas/Project/Paper2/l1_stand_continuous_step1.pth')
    torch.save(model.state_dict(),
               f'D:/L&S/Mas/Project/Paper2/l1_stand_continuous_critic.pth')

