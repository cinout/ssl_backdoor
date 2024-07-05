# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Copyright (c) 2020 Tongzhou Wang
from PIL import ImageFilter, Image
import random
import torchvision.transforms.functional as F
import torchvision.transforms as transforms


class NCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform, n):
        self.base_transform = base_transform
        self.n = n

    def __call__(self, x):
        aug_image = []

        for _ in range(self.n):
            aug_image.append(self.base_transform(x))

        return aug_image


class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x
