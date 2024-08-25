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

target_class = 26
labels = torch.tensor([2, 5, 6])
labels = torch.ones_like(labels) * target_class
print(labels)
