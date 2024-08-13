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


a = torch.randn((10, 10))
b = torch.tensor([2, 5, 3])
a[:, b] = 0.0
print("start waiting")
time.sleep(90)
print(a)
