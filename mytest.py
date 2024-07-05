import torch
import numpy as np

input1 = torch.randn(100, 128)
input2 = torch.randn(100, 128)

output = input1 @ input2.T
print(output.shape)
