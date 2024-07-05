import torch
import numpy as np

neighbors = torch.tensor(
    [[20, 1, 20, 1, 20, 1, 2], [20, 3, 20, 1, 20, 1, 2], [20, 5, 20, 1, 10, 1, 4]]
)
print(neighbors.shape)
unique_values = torch.unique(neighbors, dim=-1)
print(unique_values)
