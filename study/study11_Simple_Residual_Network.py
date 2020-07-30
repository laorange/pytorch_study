import torch


class ResidualBlock(torch.nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.channels = channels
        self.conv1 = torch.nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        y = torch.relu(self.conv1(x))
        y = self.conv2(y)
        return torch.relu(x + y)


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 16, kernel_size=5)
        self.conv2 = torch.nn.Conv2d(16, 32, kernel_size=5)
        self.mp = torch.nn.MaxPool2d(2)

        self.res_block1 = ResidualBlock(16)
        self.res_block2 = ResidualBlock(32)

        self.fc = torch.nn.Linear(512, 10)

    def forward(self, x):
        in_size = x.size(0)
        x = self.mp(torch.relu(self.conv1(x)))
        x = self.res_block1(x)
        x = self.mp(torch.relu(self.conv2(x)))
        x = self.res_block2(x)
        x = x.view(in_size, -1)
        x = self.fc(x)
        return x
