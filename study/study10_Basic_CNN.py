# CNN
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


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 10, kernel_size=5)  # 10,24,24  pool: 10,12,12
        self.conv2 = torch.nn.Conv2d(10, 20, kernel_size=5)  # 20,8,8   pool: 20,4,4
        self.pooling = torch.nn.MaxPool2d(2)
        self.fc = torch.nn.Linear(320, 10)  # full_connected_neural_network

    def forward(self, x):
        # Flatten data from (n, 1, 28, 28) to (n, 784)
        batch_size = x.size(0)
        x = torch.relu(self.pooling(self.conv1(x)))
        x = torch.relu(self.pooling(self.conv2(x)))
        x = x.view(batch_size, -1)
        x = self.fc(x)  # 因为用交叉熵损失，最后一层不作激活
        return x


model = Net()

# Move model to GPU
# Define device as the first visible cuda device if we have CUDA available.
device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')  # GPU--------------
print(torch.cuda.is_available())
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
        if batch_idx % 300 == 299:
            print('第%d轮,训练次数:%d,loss:%.3f' % (epoch + 1, batch_idx + 1, running_loss / 300))
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
    for epoch in range(10):
        train(epoch)
        test()

    epoch_ls = list(np.arange(1, len(accuracy_ls)+1))
    plt.plot(epoch_ls, accuracy_ls, c='r')
    plt.xlabel('Epoch')
    plt.ylabel('accuracy')
    plt.savefig('study10_basicCNN_MNIST.png', dpi=300)
    plt.show()
