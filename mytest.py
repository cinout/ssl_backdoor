import torch
import numpy as np

neighbors = torch.tensor(
    [[20, 1, 20, 1, 20, 1, 2], [20, 3, 20, 1, 20, 1, 2], [20, 5, 20, 1, 10, 1, 4]]
)
x = torch.zeros([neighbors.shape[0]])

for i in range(len(neighbors)):
    x[i] = len(neighbors[i].unique())
print(x)
