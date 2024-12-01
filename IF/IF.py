import numpy as np

V = 0
v_th = 1.0
v_reset = 0
w = 0.5
time_steps = 10
inputs = np.random.randint(0, 2, time_steps)
spikes = []

for t in range(time_steps):
    I_t = w * inputs[t]
    V += I_t
    if V > v_th:
        spikes.append(1)
        V = v_reset
    else:
        spikes.append(0)

print("输入信号为: ", inputs)
print("输出脉冲为: ", spikes)
