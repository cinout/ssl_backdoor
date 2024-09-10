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

print(a * (-1))

# plt.close()
