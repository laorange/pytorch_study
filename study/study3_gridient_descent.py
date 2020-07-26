# 梯度下降算法
import matplotlib.pyplot as plt
import numpy as np

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w = 1


def forward(x):
    return x * w


def cost(xs, ys):
    cost = 0
    for x, y in zip(xs, ys):
        y_pred = forward(x)
        cost += (y_pred - y) ** 2
    return cost / len(xs)


def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) ** 2


def gradient(xs, ys):
    grad = 0
    for x, y in zip(xs, ys):
        grad += 2 * x * (x * w - y)
    return grad / len(xs)


def stochastic_gradient(x, y):
    return 2 * x * (x * w - y)


epoch_ls = list(np.arange(1, 101, 1))
cost_ls = []

print('Predict (before training)', 4, forward(4))

for epoch in range(100):
    cost_val = cost(x_data, y_data)
    # for x, y in zip(x_data, y_data):
    grad_val = gradient(x_data, y_data)
    w -= 0.01 * grad_val
    print('Epoch:', epoch, 'w=', w, 'cost=', cost_val)
    cost_ls.append(cost_val)

print('Predict (after training)', 4, forward(4))

plt.plot(epoch_ls, cost_ls)
plt.xlabel('epoch')
plt.ylabel('cost')
plt.show()
