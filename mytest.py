import torch
import numpy as np

data = np.load(
    "moco/HTBA_trigger_10_targeted_n02106550/0002/mocom0.999_contr1tau0.2_mlp_aug+_cos_b256_lr0.06_e120,160,200/linear_0.01_PRE/checkpoint_0199.pth.tar/conf_matrix_poisoned.npy"
)

print(data)
print(data.shape)
