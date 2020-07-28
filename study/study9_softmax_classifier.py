# 多分类问题
import numpy as np
import torch
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # 均值 & 标准差, MNIST的经验值
])

train_dataset = datasets.MNIST(root='../dataset/mnist/',
                               train=True,
                               download=True,
                               transform=transform)

pass

test_dataset = datasets.MNIST(root='../dataset/mnist/',
                              train=True,
                              download=True,
                              transform=transform)

pass
