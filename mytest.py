import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from moco.moco.loader import NCropsTransform
from moco.moco.loader import GaussianBlur
import random
from functools import partial
import torch.nn.functional as F


import numpy as np
import torch
from pyod.models.mad import MAD


def min_max_normalization(x):
    x_min = torch.min(x)
    x_max = torch.max(x)
    norm = (x - x_min) / (x_max - x_min)
    return norm


def get_ss_score(full_cov):
    """
    https://github.com/MadryLab/backdoor_data_poisoning/blob/master/compute_corr.py
    """
    full_mean = np.mean(full_cov, axis=0, keepdims=True)
    centered_cov = full_cov - full_mean
    u, s, v = np.linalg.svd(centered_cov, full_matrices=False)
    eigs = v[0:1]
    # shape num_top, num_active_indices
    corrs = np.matmul(eigs, np.transpose(full_cov))  # full_cov or centered_cov??
    scores = np.linalg.norm(corrs, axis=0)  # 2-norm by default
    return scores


class SSAnalysis:
    def __init__(self):
        """
        Note that we assume the backdoor target label is unknown,
        this may impacts performance of SS
        """
        return

    def train(self, data, targets, cls_idx):
        # Iterating over all labels
        cls_scores = []
        for idx in cls_idx:
            if len(idx) == 0:
                cls_scores.append([])
                continue
            temp_feats = data[idx]
            scores = get_ss_score(
                temp_feats.flatten(start_dim=1).detach().cpu().numpy()
            )
            cls_scores.append(scores)

        # extract score back to original sequence
        scores = []
        for i in range(data.shape[0]):
            c = targets[i]
            c_i = np.where(cls_idx[c] == i)
            s = cls_scores[c][c_i][0]
            scores.append(s)
        scores = np.array(scores)
        self.mean = np.mean(scores)
        self.std = np.std(scores)
        return

    def predict(self, data, targets, cls_idx, t=1):
        # Iterating over all labels
        cls_scores = []
        for idx in cls_idx:
            temp_feats = data[idx]
            scores = get_ss_score(
                temp_feats.flatten(start_dim=1).detach().cpu().numpy()
            )
            cls_scores.append(scores)

        # extract score back to original sequence
        scores = []
        for i in range(data.shape[0]):
            c = targets[i]
            c_i = np.where(cls_idx[c] == i)
            s = cls_scores[c][c_i][0]
            scores.append(s)
        scores = np.array(scores)
        p = np.abs((self.mean - scores)) / self.std
        p = np.where((p > t), 1, 0)
        return p

    def analysis(self, data, targets, cls_idx):
        """
        data (torch.tensor) b,c,h,w: data is the extracted feature from the model
        """

        # Iterating over all labels
        cls_scores = []
        for idx in cls_idx:
            temp_feats = data[idx]
            scores = get_ss_score(
                temp_feats.flatten(start_dim=1).detach().cpu().numpy()
            )
            cls_scores.append(scores)

        # extract score back to original sequence
        scores = []
        for i in range(data.shape[0]):
            c = targets[i]
            c_i = np.where(cls_idx[c] == i)
            s = cls_scores[c][c_i][0]
            scores.append(s)
        scores = np.array(scores).reshape(-1, 1)
        clf = MAD()  # This improves SS performance
        clf.fit(scores)
        return clf.decision_scores_


exit()
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
