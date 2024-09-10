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


a = np.array([2, 0, 9, 0])
print(a.sum(), a.shape[0])
# with open("zz_clean.npy", "rb") as f:
#     clean_voted_channels = np.load(f)
# with open("zz_poison.npy", "rb") as f:
#     poison_voted_channels = np.load(f)


# fig, ax = plt.subplots(1, 2, figsize=(8, 4))
# # fig.suptitle("Channel Voting Distribution")

# ax[0].set(
#     xlabel="Channel Index",
#     ylabel="Votes",
# )
# ax[1].set(
#     xlabel="Channel Index",
#     # ylabel="Votes",
# )

# # draw histogram (BOTH)
# n_bins = 200
# ax[0].hist(
#     clean_voted_channels,
#     bins=n_bins,
#     color="cornflowerblue",
#     label="clean",
# )
# ax[1].hist(
#     poison_voted_channels,
#     bins=n_bins,
#     color="tomato",
#     label="poison",
# )

# legend = ax[0].legend(loc="upper right", shadow=True)
# legend = ax[1].legend(loc="upper right", shadow=True)
# legend.get_frame()

# plt.savefig(f"zz.png")
# plt.close()
