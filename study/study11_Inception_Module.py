# advanced CNN
import torch
# from torchvision import transforms
# from torchvision import datasets
# from torch.utils.data import DataLoader
# import numpy as np
# import matplotlib.pyplot as plt

pass


class InceptionA(torch.nn.Module):
    def __init__(self, in_channel):
        super(InceptionA, self).__init__()
        self.branch1x1 = torch.nn.Conv2d(in_channel, 16, kernel_size=1)

        self.branch5x5_1 = torch.nn.Conv2d(in_channel, 16, kernel_size=1)
        self.branch5x5_2 = torch.nn.Conv2d(16, 24, kernel_size=5, padding=2)

        self.branch3x3_1 = torch.nn.Conv2d(in_channel, 16, kernel_size=1)
        self.branch3x3_2 = torch.nn.Conv2d(16, 24, kernel_size=3, padding=1)
        self.branch3x3_3 = torch.nn.Conv2d(24, 24, kernel_size=3, padding=1)

        self.branch_pool = torch.nn.Conv2d(in_channel, 24, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)

        branch_pool = torch.nn.functional.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3, branch_pool]
        return torch.cat(outputs, dim=1)


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = torch.nn.Conv2d(88, 20, kernel_size=5)

        self.incep1 = InceptionA(in_channel=10)
        self.incep2 = InceptionA(in_channel=20)

        self.mp = torch.nn.MaxPool2d(2)
        self.fc = torch.nn.Linear(1408, 10)  # 1408是28x28经过网络后计算得出的数据量

    def forward(self, x):
        in_size = x.size(0)
        x = torch.relu(self.mp(self.conv1(x)))
        x = self.incep1(x)
        x = torch.relu(self.mp(self.conv2(x)))
        x = self.incep2(x)
        x = x.view(in_size, -1)
        self.fc(x)
        return x



