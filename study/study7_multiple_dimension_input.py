# 处理多维特征的输入
import torch
import numpy as np
import matplotlib.pyplot as plt

xy = np.loadtxt('dataset/diabetes.csv.gz', delimiter=',', dtype=np.float32)
x_data = torch.from_numpy(xy[:, :-1])
y_data = torch.from_numpy(xy[:, [-1]])


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(8, 64)
        self.linear2 = torch.nn.Linear(64, 128)
        self.linear3 = torch.nn.Linear(128, 64)
        self.linear4 = torch.nn.Linear(64, 8)
        self.linear5 = torch.nn.Linear(8, 1)
        self.sigmoid = torch.nn.ReLU()   # Sigmoid()
        # self.activate = torch.nn.ReLU()

    def forward(self, x):
        x = torch.sigmoid(self.linear1(x))
        x = torch.sigmoid(self.linear2(x))
        x = torch.sigmoid(self.linear3(x))
        x = torch.sigmoid(self.linear4(x))
        x = torch.sigmoid(self.linear5(x))
        return x


model = Model()

criterion = torch.nn.BCELoss(reduction='sum')
optimiser = torch.optim.Adam(model.parameters(), lr=0.01)

training_epoch = 5000
epoch_ls = list(np.arange(1, training_epoch+1, 1))
loss_ls = []

for epoch in range(training_epoch):
    # Forward
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    print(epoch, loss.item())

    loss_ls.append(loss.item())

    # Backward
    optimiser.zero_grad()
    loss.backward()

    # Update
    optimiser.step()

# print('w=', model.linear.weight.item())
# print('b=', model.linear.bias.item())

# x_test = torch.Tensor([[4.0]])
# y_test = model(x_test)
# print('y_pred=', y_test.data)
#
# x = np.linspace(0, 10, 200)
# x_t = torch.Tensor(x).view((200, 1))
# y_t = model(x_t)
# y = y_t.data.numpy()
# plt.plot(x, y)
# plt.plot([0, 10], [0.5, 0.5], c='r')
# plt.xlabel('Hours')
# plt.ylabel('probability of pass')
# plt.grid()
# plt.show()

plt.plot(epoch_ls, loss_ls)
# plt.plot([0, 10], [0.5, 0.5], c='r')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.grid()
# plt.savefig('study7_diabetes_loss.png', bbox_inches='tight', dpi=300)
plt.show()
