# 多分类问题
import numpy as np
import torch
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F

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

pass
print('done')
