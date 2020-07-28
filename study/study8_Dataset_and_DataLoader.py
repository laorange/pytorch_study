# 加载数据集
import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

filepath = 'dataset/diabetes.csv.gz'


class DiabetesDataset(Dataset):
    def __init__(self, file_path):
        xy = np.loadtxt(file_path, delimiter=',', dtype=np.float32)
        self.len = xy.shape[0]
        self.x_data = torch.from_numpy(xy[:, :-1])
        self.y_data = torch.from_numpy(xy[:, [-1]])

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


training_epoch = 100
batch_size = 100
dataset = DiabetesDataset(filepath)
train_loader = DataLoader(dataset=dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=3)  # mini-batch 读取数据时的并行进程


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(8, 64)
        self.linear2 = torch.nn.Linear(64, 128)
        self.linear3 = torch.nn.Linear(128, 64)
        self.linear4 = torch.nn.Linear(64, 8)
        self.linear5 = torch.nn.Linear(8, 1)
        self.sigmoid = torch.nn.Sigmoid()
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

loss_ls = []

if __name__ == '__main__':
    for epoch in range(training_epoch):
        loss_temp = 0
        for i, data in enumerate(train_loader, 0):
            # 1.prepare date
            inputs, labels = data

            # 2.forward
            y_pred = model(inputs)
            loss = criterion(y_pred, labels)
            print(epoch, i, loss.item())

            loss_temp = loss.item()

            # 3.backward
            optimiser.zero_grad()
            loss.backward()

            # 4.update
            optimiser.step()

        loss_ls.append(loss_temp)

    epoch_ls = list(np.arange(1, len(loss_ls) + 1, 1))
    plt.plot(epoch_ls, loss_ls)
    # plt.plot([0, 10], [0.5, 0.5], c='r')
    plt.xlabel('training_times')
    plt.ylabel('loss')
    plt.grid()
    plt.savefig('study8_diabetes_loss.png', bbox_inches='tight', dpi=300)
    plt.show()
