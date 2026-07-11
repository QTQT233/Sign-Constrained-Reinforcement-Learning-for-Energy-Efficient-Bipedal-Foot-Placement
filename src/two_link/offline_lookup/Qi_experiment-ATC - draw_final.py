from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import h5py
from numba import njit


@njit
def calc_new_a_b(y_, m1_, m2_, l1_, l2_, l1_1, l2_1, g_):
    a_ = np.array([[(m1_ * l1_1 ** 2 + m2_ * l1_ ** 2), m2_ * l1_ * l2_1 * np.sin(y_[2] - y_[0])],
                   [m2_ * l1 * l2_1 * np.sin(y_[2] - y_[0]), (m2_ * l2_1 ** 2 + m2_ * l2_1 ** 2)]])
    b = np.array([[-m2_ * l1 * l2_1 * y_[3] * np.cos(y_[2] - y_[0]) * (y_[3] - y_[1])
                   - m2_ * l1 * l2_1 * y_[1] * y_[3] * np.cos(y_[2] - y_[0])
                   - m1_ * g_ * l1_1 * np.cos(y_[0]) - m2_ * g_ * l1_ * np.cos(y_[0])],
                  [-m2_ * l1_1 * l2_1 * y_[1] * np.cos(y_[2] - y_[0]) * (y_[3] - y_[1])
                   + m2_ * l1 * l2_1 * y_[1] * y_[3] * np.cos(y_[2] - y_[0])
                   - m2_ * g_ * l2_1 * np.sin(y_[2])]])
    identity_a = np.eye(a_.shape[0])
    inverse_a = np.linalg.solve(a_, identity_a)
    return inverse_a, b


data = pd.read_csv("C:/Users/Admin/Desktop/paper-code - F/data-ACexp.csv")
exp_time = np.array(data['Time'].ravel()) / 1000000 - 2.887
theta1 = np.array(data['Angle1'].ravel())
dtheta1 = np.array(data['Vel'].ravel())
action = np.array(data['Action'].ravel())
theta2 = np.array(data['Angle2'].ravel())
dtheta2 = np.array(data['Vel2'].ravel())
N = 50
m1 = 1.4122
m2 = 0.0839
l1 = 0.22
l2 = 0.31
l1_r = 0.33
l2_r = 0.33
c1 = 0.03
c2 = 0.3
a = 0
g = 9.8
torque = 4
tht1s = np.linspace(np.pi / 3, 2 * np.pi / 3, N)
dtht1s = np.linspace(-3.5, 3.5, N)
tht2s = np.linspace(-np.pi / 3, np.pi / 3, N)
dtht2s = np.linspace(-3.5, 3.5, N)
reward = np.zeros((N, N, N, N))
with h5py.File('C:/Users/Admin/Desktop/paper-code - F/working_save-ATC-50', 'r') as h5f:
    Q = np.array(h5f['working_save'][:])
simu_time = 3.2
dt_plot = 0.005
steps = int(simu_time / dt_plot)
y = np.zeros(4)
y[0] = 88.2 * np.pi / 180
y[1] = -19.98 * np.pi / 180
y[2] = 0 * np.pi / 180
y[3] = 0 * np.pi / 180
y_trace = np.zeros((steps + 1, 4))
y_trace[0, :] = y
a_save = np.zeros((steps + 1, 1))
Energy_save = np.zeros((steps + 1, 1))
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
        c2 = 0.001
    arr = np.array([[-a * torque], [a * torque]])
    temp = np.dot(inverse_A, B + arr)
    Energy_save[step] = 1/2
    y[1] += temp[0] * dt_plot - c1 * y[1]
    y[0] += y[1] * dt_plot
    y[3] += temp[1] * dt_plot - c2 * y[3]
    y[2] += y[3] * dt_plot
    y_trace[step + 1, :] = y
    a_save[step + 1, 0] = a
    cont += 1
    if (abs(y[0] - np.pi / 3) < 0.08 and abs(y[2] - np.pi / 6) < 0.08) \
            or (abs(y[0] - 2 * np.pi / 3) < 0.08 and abs(y[2] - (-np.pi / 6)) < 0.08):
        print('\nAchieve the goal!')
        print(step)
        for repeats in range(2):
            step += 1
            i = np.argmin(abs(y[0] - tht1s))
            j = np.argmin(abs(y[1] - dtht1s))
            k = np.argmin(abs(y[2] - tht2s))
            ll = np.argmin(abs(y[3] - dtht2s))
            i_old = i
            j_old = j
            k_old = k
            ll_old = ll
            A = np.array([[(m1 + m2) * l1 ** 2, m2 * l1 * l2 * np.sin(y[0] - y[2])],
                          [m2 * l1 * l2 * np.sin(y[0] - y[2]), m2 * l2 ** 2]])
            B = np.array([[-m2 * l1 * l2 * y[3] * np.cos(y[0] - y[2]) * (y[1] - y[3])
                           + m2 * l1 * l2 * y[1] * y[3] * np.cos(y[0] - y[2])
                           - m1 * g * l1 * np.cos(y[0]) - m2 * g * l1 * np.cos(y[0]) - c1 * y[1]],
                          [-m2 * l1 * l2 * y[1] * np.cos(y[0] - y[2]) * (y[1] - y[3])
                           - m2 * l1 * l2 * y[1] * y[3] * np.cos(y[0] - y[2])
                           - m2 * g * l2 * np.sin(y[2]) - c2 * y[3]]])
            a = Q[i, j, k, ll]
            if a == -2:
                a = 0
            temp = np.dot(np.linalg.inv(A), B + [[-a * torque], [a * torque]])
            y[1] += temp[0] * dt_plot - c1 * y[1]
            y[0] += y[1] * dt_plot
            y[3] += temp[1] * dt_plot - c2 * y[3]
            y[2] += y[3] * dt_plot
            y_trace[step + 1, :] = y
        break

t = np.arange(0, simu_time + dt_plot, dt_plot)
horizontal_line_25 = [25] * len(t)
horizontal_line_35 = [35] * len(t)
horizontal_line_55 = [55] * len(t)
horizontal_line_65 = [65] * len(t)
start_point = 18

plt.figure(figsize=(4, 4.1), dpi=600)
plt.subplot(311)
plt.plot(t[0:step + 1], y_trace[0:step + 1, 0] * 180 / np.pi, 'b-', label=r'$\theta_1$ Simu.')
plt.plot(exp_time, theta1, 'b--', label=r'$\theta_1$ Exp.')
plt.plot(t[0:step + 2], y_trace[0:step + 2, 2] * 180 / np.pi, 'r-', label=r'$\theta_2$ Simu.')
plt.plot(exp_time, theta2, 'r--', label=r'$\theta_2$ Exp.')
plt.gca().add_patch(Rectangle((0, horizontal_line_25[start_point]), 1,
                              horizontal_line_35[start_point] - horizontal_line_25[start_point],
                              edgecolor='none', facecolor=[1, 0, 0], alpha=0.2))
plt.gca().add_patch(Rectangle((0, horizontal_line_55[start_point]), 1,
                              horizontal_line_65[start_point] - horizontal_line_55[start_point],
                              edgecolor='none', facecolor=[0, 0, 1], alpha=0.2))
plt.ylabel(r'Angle $[^\circ]$')
plt.xticks([])
plt.xlim([0.0, 0.445])
plt.ylim([-10, 90])
plt.legend(ncol=2, loc=[0.01, 0.29])
plt.subplot(312)
plt.plot(t[0:step + 1], y_trace[0:step + 1, 1] * 180 / np.pi, 'b-', label=r'$\dot{\theta}_1$ Simu.')
plt.plot(t[0:step + 2], y_trace[0:step + 2, 3] * 180 / np.pi, 'r-', label=r'$\dot{\theta}_2$ Simu.')
plt.ylabel(r'Angular velo. $[^\circ/s]$')
plt.xticks([])
plt.xlim([0.0, 0.445])
plt.legend(loc=[0.01, 0.4])
plt.subplot(313)
plt.step(t[0:step + 2], a_save[0:step + 2, 0] * torque, 'r-', label='Simu.')
plt.step(exp_time, action * torque, 'b--', where='post', label='Exp.')
plt.ylabel(r'Torque $\tau$ [Nm]')
plt.legend()
plt.xlim([0.0, 0.445])
plt.ylim([-0.1, 5])
plt.xlabel('Time [s]')
plt.tight_layout()
plt.show()
# plt.savefig('temp.pdf')
# plt.savefig("C:/Users/Admin/Desktop/IEEE/Figure13.svg")