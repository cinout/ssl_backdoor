import torch
import numpy as np

batch_size = 4
num_views = 8

neighbors = torch.randint(0, 100, size=(batch_size * num_views,))

print(neighbors)


neighbors = neighbors.reshape(num_views, -1)
print(neighbors)
