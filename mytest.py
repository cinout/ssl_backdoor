import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from moco.moco.loader import NCropsTransform
from moco.moco.loader import GaussianBlur
import random
from functools import partial


"""
VISUALIZE AUGMENTATION
"""
basic_augmentation = [
    transforms.RandomResizedCrop(224, scale=(0.3, 0.95), ratio=(0.2, 5)),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5),
]


augmentation = basic_augmentation + [transforms.ElasticTransform(alpha=[0.0, 120.0])]


num_views = 4
sample_backdoored_imgs = [
    "/Users/haitianh/Downloads/Code/ssl_backdoor/dataset/SSL-Backdoor/train/loc_random_loc-min_0.25_loc-max_0.75_alpha_0.00_width_50_rate_0.50_targeted_True/n02106550/n02106550_11441.jpg",
    "/Users/haitianh/Downloads/Code/ssl_backdoor/dataset/SSL-Backdoor/train/loc_random_loc-min_0.25_loc-max_0.75_alpha_0.00_width_50_rate_0.50_targeted_True/n02106550/n02106550_10993.jpg",
    "/Users/haitianh/Downloads/Code/ssl_backdoor/dataset/SSL-Backdoor/train/loc_random_loc-min_0.25_loc-max_0.75_alpha_0.00_width_50_rate_0.50_targeted_True/n02106550/n02106550_213.jpg",
    "/Users/haitianh/Downloads/Code/ssl_backdoor/dataset/SSL-Backdoor/train/loc_random_loc-min_0.25_loc-max_0.75_alpha_0.00_width_50_rate_0.50_targeted_True/n02106550/n02106550_7608.jpg",
    "/Users/haitianh/Downloads/Code/ssl_backdoor/dataset/SSL-Backdoor/train/loc_random_loc-min_0.25_loc-max_0.75_alpha_0.00_width_50_rate_0.50_targeted_True/n02106550/n02106550_11867.jpg",
    "/Users/haitianh/Downloads/Code/ssl_backdoor/dataset/SSL-Backdoor/train/loc_random_loc-min_0.25_loc-max_0.75_alpha_0.00_width_50_rate_0.50_targeted_True/n02106550/n02106550_398.jpg",
    "/Users/haitianh/Downloads/Code/ssl_backdoor/dataset/SSL-Backdoor/train/loc_random_loc-min_0.25_loc-max_0.75_alpha_0.00_width_50_rate_0.50_targeted_True/n02106550/n02106550_4802.jpg",
    "/Users/haitianh/Downloads/Code/ssl_backdoor/dataset/SSL-Backdoor/train/loc_random_loc-min_0.25_loc-max_0.75_alpha_0.00_width_50_rate_0.50_targeted_True/n02106550/n02106550_6064.jpg",
    "/Users/haitianh/Downloads/Code/ssl_backdoor/dataset/SSL-Backdoor/train/loc_random_loc-min_0.25_loc-max_0.75_alpha_0.00_width_50_rate_0.50_targeted_True/n02106550/n02106550_4191.jpg",
    "/Users/haitianh/Downloads/Code/ssl_backdoor/dataset/SSL-Backdoor/train/loc_random_loc-min_0.25_loc-max_0.75_alpha_0.00_width_50_rate_0.50_targeted_True/n02106550/n02106550_3500.jpg",
    "/Users/haitianh/Downloads/Code/ssl_backdoor/dataset/SSL-Backdoor/train/loc_random_loc-min_0.25_loc-max_0.75_alpha_0.00_width_50_rate_0.50_targeted_True/n02106550/n02106550_8618.jpg",
]

transform = NCropsTransform(transforms.Compose(augmentation), num_views)

for img_path in sample_backdoored_imgs:
    img = Image.open(img_path).convert("RGB")
    img = transform(img)  # a list of size "num_views"

    file_name = img_path.split("/")[-1].split(".")[0]
    print(file_name)
    for idx, view in enumerate(img):
        view.save(f"{file_name}_view{idx}.jpg", "JPEG")
