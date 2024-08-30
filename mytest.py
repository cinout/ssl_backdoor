import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from moco.moco.loader import NCropsTransform
from moco.moco.loader import GaussianBlur
import random
from functools import partial
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt
from collections import Counter
import time

# max_indices_at_channel = np.random.randint(0, 20, size=(4, 10))
# this_bs, num_views = max_indices_at_channel.shape
# print(max_indices_at_channel)

# entropies = []

# for votes in max_indices_at_channel:
#     votes_counter = Counter(votes).most_common()
#     counts = np.array([c for (name, c) in votes_counter])
#     print(counts)
#     p = counts / counts.sum()
#     h = -np.sum(p * np.log(p))
#     entropy = np.exp(h)
#     entropies.append(entropy)

# print(f">>>> entropies at channel are: {[round(e,2) for e in entropies]}")
# entropies = np.array(entropies)
# min_index = np.argmin(entropies)
# print(min_index)

a = [2, 3, 5, 1, 2]
a = np.unique(np.array(a))
print(a)
