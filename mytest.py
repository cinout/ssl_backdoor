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

with open("zz_clean.npy", "rb") as f:
    clean_voted_channels = np.load(f)
with open("zz_poison.npy", "rb") as f:
    poison_voted_channels = np.load(f)
print(clean_voted_channels.shape)
