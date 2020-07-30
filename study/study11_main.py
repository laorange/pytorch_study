# basic CNN
import torch
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

batch_size = 64

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # 均值 & 标准差, MNIST的经验值
])

train_dataset = datasets.MNIST(root='../dataset/',
                               train=True,
                               download=False,
                               transform=transform)

train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)

test_dataset = datasets.MNIST(root='../dataset/',
                              train=True,
                              download=False,
                              transform=transform)

test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)


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


model = Net()

# Move model to GPU
# Define device as the first visible cuda device if we have CUDA available.
device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')  # GPU--------------
print('use GPU:', torch.cuda.is_available())
model.to(device)  # convert parameters and buffers of all modules to CUDA tensor.

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)


def train(epoch):
    running_loss = 0.0
    for batch_idx, data in enumerate(train_loader, 0):
        inputs, target = data
        inputs, target = inputs.to(device), target.to(device)  # GPU--------------
        optimizer.zero_grad()

        # forward + backward + update
        outputs = model(inputs)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

        running_loss += loss
        if batch_idx % 200 == 199:
            print('第%d轮,训练次数:%d,loss:%.3f' % (epoch + 1, batch_idx + 1, running_loss / 200))
            running_loss = 0.0


accuracy_ls = []


def test():
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, label = data
            images, label = images.to(device), label.to(device)  # GPU--------------
            outputs = model(images)
            _, predicted = torch.max(outputs.data, dim=1)
            total += label.size(0)
            correct += (predicted == label).sum().item()
        accuracy_ls.append(correct / total)
        print('Accuracy on test set: %d %%' % (100 * correct / total))


if __name__ == '__main__':
    choice = input('训练请输入0,调用当前模型来测试请输入1:')
    if choice == '0':
        for epoch in range(10):
            train(epoch)
            test()

        epoch_ls = list(np.arange(1, len(accuracy_ls) + 1))
        plt.plot(epoch_ls, accuracy_ls, c='r')
        plt.xlabel('Epoch')
        plt.ylabel('accuracy')
        plt.grid()
        plt.savefig('study11_inception_MNIST.png', dpi=300)
        plt.show()

        torch.save(model, 'study11_model.pth')

    elif choice == '1':
        model = torch.load('study11_model.pth')
        test()

    else:
        print('您的输入不正确')
