import torchvision.datasets as datasets
from PIL import Image
import torchvision.transforms as transforms
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import copy
from collections import Counter

clean_file = "dataset_imagenet100_HTBA_train_clean_votes.npy"
poison_file = "dataset_imagenet100_HTBA_train_poison_votes.npy"

# clean_file = "dataset_imagenet100_train_clean_votes.npy"
# poison_file = "dataset_imagenet100_train_poison_votes.npy"
# # [(7, 2), (8, 11), (10, 42), (13, 8), (14, 17), (15, 20), (17, 6), (18, 47), (22, 38), (23, 24), (25, 1), (27, 45), (28, 23), (29, 12), (36, 46), (38, 13), (40, 33), (47, 41)]

# # [(7, 2), (8, 11), (13, 8), (14, 17), (15, 20), (17, 6)]


# clean_file = "dataset_cifar10_train_clean_votes.npy"
# poison_file = "dataset_cifar10_train_poison_votes.npy"
# # [(0, 12), (1, 14), (2, 4), (4, 0), (5, 15), (6, 1), (7, 29), (8, 10), (9, 16), (10, 37), (11, 2), (12, 11), (14, 5), (15, 34), (25, 17), (27, 6), (30, 8), (31, 13), (32, 45), (40, 7), (42, 22), (44, 9), (49, 18)]

# # [(0, 12), (1, 14), (2, 4), (4, 0), (5, 15), (6, 1), (8, 10), (9, 16), (11, 2), (12, 11), (14, 5)]

# clean_file = "dataset_cifar100_train_clean_votes.npy"
# poison_file = "dataset_cifar100_train_poison_votes.npy"
# # [(1, 49), (4, 37), (5, 18), (8, 5), (9, 1), (11, 40), (12, 0), (14, 7), (17, 11), (29, 8), (30, 9), (31, 43), (33, 2), (35, 6), (36, 15), (38, 26), (39, 4), (40, 19), (42, 25), (43, 12), (45, 24), (46, 13), (48, 17), (49, 35)]

# # [(5, 18), (8, 5), (9, 1), (12, 0), (14, 7), (17, 11)]

with open(clean_file, "rb") as f:
    cifar100_clean_votes = np.load(f)
with open(poison_file, "rb") as f:
    cifar100_poison_votes = np.load(f)


topk = 50

clean_top_channels = Counter(cifar100_clean_votes.flatten()).most_common(topk)
poison_top_channels = Counter(cifar100_poison_votes.flatten()).most_common(topk)


clean_top_channels = [c for (c, count) in clean_top_channels]
poison_top_channels = [c for (c, count) in poison_top_channels]

position_in_clean = []  # each pair is (idx_in_poison, idx_in_clean)
for i, channel in enumerate(poison_top_channels):
    if channel in clean_top_channels:
        position_in_clean.append((i, clean_top_channels.index(channel)))

print(position_in_clean)
