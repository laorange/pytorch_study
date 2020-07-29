import torch

inputs = [3, 4, 5, 6, 7,
          2, 4, 6, 8, 2,
          1, 6, 7, 8, 4,
          9, 7, 4, 6, 2,
          3, 7, 5, 4, 1]

inputs = torch.Tensor(inputs).view(1, 1, 5, 5)

conv_layer = torch.nn.Conv2d(in_channels=1,
                             out_channels=1,
                             kernel_size=3,  # 3→3x3, (5,3)...一般用奇数
                             padding=1,  # 向外填充0
                             stride=2,  # 步长
                             bias=False)

kernel = torch.Tensor([1, 2, 3, 4, 5, 6, 7, 8, 9]).view(1, 1, 3, 3)

conv_layer.weight.data = kernel.data

output = conv_layer(inputs)
print(output)

maxpooling_layer = torch.nn.MaxPool2d(kernel_size=2)  # 最大池化, 默认的stride也会=2
output = maxpooling_layer(inputs)
print(output)

# ----------------------------------------------------------------------- #

# in_channels, out_channels = 5, 10
# width, height = 100, 100  # 图像大小
# kernel_size = 3  # 卷积核大小
# batch_size = 1
#
# inputs = torch.randn(batch_size, in_channels, width, height)  # 从正态分布进行采样的随机数 random normal
#
# conv_layer = torch.nn.Conv2d(in_channels=in_channels,
#                              out_channels=out_channels,
#                              kernel_size=kernel_size)  # 3→3x3, (5,3)...一般用奇数
#
# outputs = conv_layer(inputs)
#
# print(inputs.shape)
# print(outputs.shape)
# print(conv_layer.weight.shape)
