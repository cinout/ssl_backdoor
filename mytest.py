import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from moco.moco.loader import NCropsTransform
from moco.moco.loader import GaussianBlur
import random
from functools import partial
import torch.nn.functional as F


class STRIP_Detection(torch.nn.Module):
    def __init__(self, data, alpha=1.0, beta=1.0, n=100):
        super(STRIP_Detection, self).__init__()
        self.data = data
        self.alpha = alpha
        self.beta = beta
        self.n = n

    # def _superimpose(self, background, overlay):
    #     # cv2.addWeighted(background, 1, overlay, 1, 0)
    #     imgs = self.alpha * background + self.beta * overlay
    #     imgs = torch.clamp(imgs, 0, 1)
    #     return imgs

    def forward(self, model, imgs, labels=None):
        # Return Entropy H
        idx = np.random.randint(0, self.data.shape[0], size=self.n)
        H = []
        for img in imgs:
            x = torch.stack([img] * self.n).to(imgs.device)
            for i in range(self.n):
                x_0 = x[i]
                x_1 = self.data[idx[i]].to(imgs.device)
                x_2 = self._superimpose(x_0, x_1)
                x[i] = x_2
            logits = model(x)
            p = F.softmax(logits.detach(), dim=1)
            H_i = -torch.sum(p * torch.log(p), dim=1)
            H.append(H_i.mean().item())
        return torch.tensor(H).detach().cpu()


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
