import torch
from torch.utils.tensorboard import SummaryWriter

from model_cnn_res import ResidualBlock, CnnWithResidualBlocks
import numpy as np

writer = SummaryWriter()
output_channels = [16, 16, 16, 16]
kernel_sizes = [3 for i in range(len(output_channels))]
paddings = [0, 1, 1, 1]
strides = [2, 1, 1, 1]
shortcut_connection_flags = [0, 1]*2

res_block = ResidualBlock(1, output_channels, kernel_sizes, strides,
                    paddings, shortcut_connection_flags, True)

x = np.ones(shape=(10, 187, 1))
x = torch.tensor(data=x, dtype=torch.float32)
x = torch.transpose(x, 1, 2) # channel first

y = res_block(x)

resnet = CnnWithResidualBlocks(config=None)
y2 = resnet(x)

writer.add_graph(res_block, x)
writer.add_graph(resnet, x)
writer.close()