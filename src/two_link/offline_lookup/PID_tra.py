import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pandas as pd
import h5py


# 双摆动力学模型（与原始代码相同）
def calc_new_a_b(y_, m1_, m2_, l1_r, l2_r, l1_1, l2_1, g_):
    j1 = m1_ * l1_1 ** 2 / 2
    j2 = m2_ * l2_1 ** 2 / 2
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


# 加载原始强化学习仿真的状态轨迹
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

        inverse_A, B = calc_new_a_b(y, m1, m2, l1_r, l2_r, l1, l2, g)
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

        y_trace[step + 1, :] = y
        a_save[step + 1, 0] = a
        cont += 1

        if (abs(y[0] - np.pi / 3) < 0.08 and abs(y[2] - np.pi / 6) < 0.08) \
                or (abs(y[0] - 2 * np.pi / 3) < 0.08 and abs(y[2] - (-np.pi / 6)) < 0.08):
            print(f'\nAchieved the goal at step {step}!')
            break

    t = np.arange(0, (step + 1) * dt_plot, dt_plot)
    return t, y_trace[:step + 1, :], a_save[:step + 1, 0]


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
        combined_error = 0.7 * error1 + 0.3 * error2

        # 比例项
        P = self.Kp * combined_error

        # 积分项
        self.integral += combined_error * self.dt
        I = self.Ki * self.integral

        # 微分项
        derivative = (combined_error - self.previous_error) / self.dt
        D = self.Kd * derivative

        # 计算总输出
        output = P + I + D

        # 限幅
        output = np.clip(output, self.min_output, self.max_output)
        self.previous_error = combined_error

        return output


# 主函数
def main():
    # 加载RL轨迹
    rl_time, rl_trajectory, rl_actions = load_rl_trajectory()

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
    j1 = m1 * l1 ** 2 / 2
    j2 = m2 * l2 ** 2 / 2

    # 时间参数（与RL轨迹相同）
    dt_plot = 0.005
    steps = len(rl_time)

    # 初始化状态（与原始RL相同）
    y = np.zeros(4)
    y[0] = 88.2 * np.pi / 180  # theta1
    y[1] = -19.98 * np.pi / 180  # dtheta1
    y[2] = 0 * np.pi / 180  # theta2
    y[3] = 0 * np.pi / 180  # dtheta2

    # 创建PID控制器（用于跟踪RL轨迹）
    pid = TrajectoryTrackingPID(Kp=12.0, Ki=0.05, Kd=2.0,
                                max_output=1.0, min_output=-1.0, dt=dt_plot)

    # 存储结果
    pid_trace = np.zeros((steps + 1, 4))
    pid_trace[0, :] = y
    pid_actions = np.zeros(steps + 1)

    # 主仿真循环
    for step in range(steps):
        # 获取当前时间点的RL目标状态
        target_theta1 = rl_trajectory[step, 0]
        target_theta2 = rl_trajectory[step, 2]

        # 计算PID控制输出
        a = pid.compute(target_theta1, target_theta2, y[0], y[2])

        # 计算动力学
        inverse_A, B = calc_new_a_b(y, m1, m2, l1_r, l2_r, l1, l2, g)

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

        # 打印进度
        if step % 50 == 0:
            print(f"Step: {step}/{steps}, Theta1: {y[0] * 180 / np.pi:.2f}°, Theta2: {y[2] * 180 / np.pi:.2f}°")

    # 读取实验数据用于比较
    data = pd.read_csv("C:/Users/Admin/Desktop/paper-code - F/data-ACexp.csv")
    exp_time = np.array(data['Time'].ravel()) / 1000000 - 2.887
    theta1_exp = np.array(data['Angle1'].ravel())
    theta2_exp = np.array(data['Angle2'].ravel())
    action_exp = np.array(data['Action'].ravel())

    # 绘制结果比较
    plt.figure(figsize=(12, 8))

    # θ1比较
    plt.subplot(221)
    plt.plot(rl_time, rl_trajectory[:, 0] * 180 / np.pi, 'b-', label='RL Theta1')
    plt.plot(rl_time, pid_trace[1:steps + 1, 0] * 180 / np.pi, 'r--', label='PID Theta1')
    plt.plot(exp_time, theta1_exp, 'g-.', label='Exp. Theta1')
    plt.ylabel(r'$\theta_1$ [deg]')
    plt.legend()
    plt.title('Theta1 Comparison')

    # θ2比较
    plt.subplot(222)
    plt.plot(rl_time, rl_trajectory[:, 2] * 180 / np.pi, 'b-', label='RL Theta2')
    plt.plot(rl_time, pid_trace[1:steps + 1, 2] * 180 / np.pi, 'r--', label='PID Theta2')
    plt.plot(exp_time, theta2_exp, 'g-.', label='Exp. Theta2')
    plt.ylabel(r'$\theta_2$ [deg]')
    plt.legend()
    plt.title('Theta2 Comparison')

    # 角速度比较
    plt.subplot(223)
    plt.plot(rl_time, rl_trajectory[:, 1] * 180 / np.pi, 'b-', label='RL dTheta1')
    plt.plot(rl_time, pid_trace[1:steps + 1, 1] * 180 / np.pi, 'r--', label='PID dTheta1')
    plt.ylabel(r'$\dot{\theta}_1$ [deg/s]')
    plt.xlabel('Time [s]')
    plt.legend()

    plt.subplot(224)
    plt.plot(rl_time, rl_trajectory[:, 3] * 180 / np.pi, 'b-', label='RL dTheta2')
    plt.plot(rl_time, pid_trace[1:steps + 1, 3] * 180 / np.pi, 'r--', label='PID dTheta2')
    plt.ylabel(r'$\dot{\theta}_2$ [deg/s]')
    plt.xlabel('Time [s]')
    plt.legend()

    plt.tight_layout()
    plt.savefig('PID_vs_RL_comparison.pdf')

    # 控制信号比较
    plt.figure(figsize=(10, 4))
    plt.plot(rl_time, rl_actions * torque, 'b-', label='RL Control')
    plt.plot(rl_time, pid_actions[:steps] * torque, 'r--', label='PID Control')
    plt.plot(exp_time, action_exp * torque, 'g-.', label='Exp. Control')
    plt.ylabel('Torque [Nm]')
    plt.xlabel('Time [s]')
    plt.legend()
    plt.title('Control Signal Comparison')
    plt.tight_layout()
    plt.savefig('control_signal_comparison.pdf')

    # 计算跟踪误差
    theta1_error = np.abs(rl_trajectory[:, 0] - pid_trace[1:steps + 1, 0]) * 180 / np.pi
    theta2_error = np.abs(rl_trajectory[:, 2] - pid_trace[1:steps + 1, 2]) * 180 / np.pi

    plt.figure(figsize=(10, 4))
    plt.plot(rl_time, theta1_error, 'b-', label='Theta1 Error')
    plt.plot(rl_time, theta2_error, 'r-', label='Theta2 Error')
    plt.ylabel('Tracking Error [deg]')
    plt.xlabel('Time [s]')
    plt.legend()
    plt.title('Tracking Error')
    plt.tight_layout()
    plt.savefig('tracking_error.pdf')

    # 打印统计信息
    print(f"Mean Theta1 Error: {np.mean(theta1_error):.4f} deg")
    print(f"Max Theta1 Error: {np.max(theta1_error):.4f} deg")
    print(f"Mean Theta2 Error: {np.mean(theta2_error):.4f} deg")
    print(f"Max Theta2 Error: {np.max(theta2_error):.4f} deg")

    plt.show()


if __name__ == "__main__":
    main()