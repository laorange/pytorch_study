# 线性模型 w * x + b
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

x_data = [1.0, 2.0, 3.0]
y_data = [3.0, 5.0, 7.0]


def forward(x):
    return x * w + b


def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) ** 2


# w_ls = []
# b_ls = []
mse_ls = []

b_ls = list(np.arange(0.0, 2.1, 0.1))
w_ls = list(np.arange(0.0, 4.1, 0.1))

for b in np.arange(0.0, 2.1, 0.1):
    for w in np.arange(0.0, 4.1, 0.1):
        print('w=', w, 'b=', b)
        l_sum = 0
        for x_val, y_val in zip(x_data, y_data):
            y_pred_val = forward(x_val)
            loss_val = loss(x_val, y_val)
            l_sum += loss_val
            print('\t', x_val, y_val, y_pred_val, loss_val)
        print('MSE=', l_sum/3)
        # w_ls.append(w)
        mse_ls.append(l_sum/3)

x, y = np.meshgrid(w_ls, b_ls)
z = mse_ls

fig = plt.figure()

plt.subplot(111)

axis = Axes3D(fig)
axis.scatter(x, y, z)

# plt.plot(w_ls, mse_ls)
# plt.xlabel('w')
# plt.ylabel('loss')
axis.set_xlabel('w', fontdict={'size': 10, 'color': 'red'})
axis.set_ylabel('b', fontdict={'size': 10, 'color': 'blue'})
axis.set_zlabel('mse', fontdict={'size': 10, 'color': 'green'})

# plt.savefig('mse.png', bbox_inches='tight', dpi=300)
plt.show()
