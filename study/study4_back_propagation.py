# 梯度下降算法
import matplotlib.pyplot as plt
import numpy as np
import torch

learning_rate = 0.01
x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w = torch.Tensor([1.0])
w.requires_grad = True


def forward(x):
    return x * w


def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) ** 2


# epoch_ls = list(np.arange(1, 101, 1))
# cost_ls = []

print('Predict (before training)', 4, forward(4))

for epoch in range(100):
    for x, y in zip(x_data, y_data):
        l = loss(x, y)
        l.backward()
        print('\tgrad:', x, y, w.grad.item())
        w.data = w.data - learning_rate * w.grad.data

        w.grad.data.zero_()

    print('progress:', epoch, l.item())

print('Predict (after training)', 4, forward(4))
