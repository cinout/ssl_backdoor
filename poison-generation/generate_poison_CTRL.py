"""
This script generates poisoned data.

Author: Aniruddha Saha
"""

import os
import re
import sys
import glob

import random
import numpy as np
import logging

import configparser
from PIL import Image
from tqdm import tqdm
import kornia

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.transforms as transforms


config = configparser.ConfigParser()
config.read(sys.argv[1])

experimentID = config["experiment"]["ID"]

seed = 42
options = config["poison_generation"]
data_root = options["data_root"]
poison_savedir = options["poison_savedir"].format(experimentID)
splits = [split for split in options["splits"].split(",")]
poison_injection_rate = float(options["poison_injection_rate"])
targeted = options.getboolean("targeted")
target_wnid = options["target_wnid"]
logfile = options["logfile"].format(experimentID)

window_size = int(options["window_size"])
magnitude = float(options["magnitude"])

val_transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224)])
channel_list = [1, 2]
# pos_list = [
#     (int(window_size / 2) - 1, int(window_size / 2) - 1),
#     (window_size - 1, window_size - 1),
# ]
pos_list = [(15, 15), (31, 31)]


os.makedirs(poison_savedir, exist_ok=True)
os.makedirs("data/{}".format(experimentID), exist_ok=True)

# logging
os.makedirs(os.path.dirname(logfile), exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    handlers=[logging.FileHandler(logfile, "w"), logging.StreamHandler()],
)


def main():
    with open("scripts/imagenet100_classes.txt", "r") as f:
        class_list = [l.strip() for l in f.readlines()]

    # # Comment lines above and uncomment if you are using Full ImageNet and provide path to train/val folder of ImageNet.
    # class_list = os.listdir('/datasets/imagenet/train')

    logging.info("Experiment ID: {}".format(experimentID))

    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)
        cudnn.deterministic = True

    # FIXME: uncomment
    generate_poison(class_list, data_root, poison_savedir, splits=splits)

    # # # Debug: If you want to run for one image.
    # # file = f"{data_root}/imagenet100/val/n01558993/ILSVRC2012_val_00001598.JPEG"
    # file = f"{data_root}/val/n01558993/ILSVRC2012_val_00029627.jpg"
    # poisoned_image = add_watermark(
    #     file,
    #     val=True,
    # )
    # poisoned_image.save("test.png")


def dct_fft_impl(v):
    return torch.view_as_real(torch.fft.fft(v, dim=1))


def idct_irfft_impl(V):
    return torch.fft.irfft(torch.view_as_complex(V), n=V.shape[1], dim=1)


def dct(x, norm=None):
    """
    Discrete Cosine Transform, Type II (a.k.a. the DCT)
    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last dimension
    """
    x_shape = x.shape
    N = x_shape[-1]
    x = x.contiguous().view(-1, N)

    v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)

    Vc = dct_fft_impl(v)

    k = -torch.arange(N, dtype=x.dtype, device=x.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V = Vc[:, :, 0] * W_r - Vc[:, :, 1] * W_i

    if norm == "ortho":
        V[:, 0] /= np.sqrt(N) * 2
        V[:, 1:] /= np.sqrt(N / 2) * 2

    V = 2 * V.view(*x_shape)

    return V


def dct_2d(x, norm=None):
    """
    2-dimentional Discrete Cosine Transform, Type II (a.k.a. the DCT)
    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last 2 dimensions
    """
    X1 = dct(x, norm=norm)
    X2 = dct(X1.transpose(-1, -2), norm=norm)
    return X2.transpose(-1, -2)


def idct(X, norm=None):
    """
    The inverse to DCT-II, which is a scaled Discrete Cosine Transform, Type III
    Our definition of idct is that idct(dct(x)) == x
    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
    :param X: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the inverse DCT-II of the signal over the last dimension
    """

    x_shape = X.shape
    N = x_shape[-1]

    X_v = X.contiguous().view(-1, x_shape[-1]) / 2

    if norm == "ortho":
        X_v[:, 0] *= np.sqrt(N) * 2
        X_v[:, 1:] *= np.sqrt(N / 2) * 2

    k = (
        torch.arange(x_shape[-1], dtype=X.dtype, device=X.device)[None, :]
        * np.pi
        / (2 * N)
    )
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V_t_r = X_v
    V_t_i = torch.cat([X_v[:, :1] * 0, -X_v.flip([1])[:, :-1]], dim=1)

    V_r = V_t_r * W_r - V_t_i * W_i
    V_i = V_t_r * W_i + V_t_i * W_r

    V = torch.cat([V_r.unsqueeze(2), V_i.unsqueeze(2)], dim=2)

    v = idct_irfft_impl(V)
    x = v.new_zeros(v.shape)
    x[:, ::2] += v[:, : N - (N // 2)]
    x[:, 1::2] += v.flip([1])[:, : N // 2]

    return x.view(*x_shape)


def idct_2d(X, norm=None):
    """
    The inverse to 2D DCT-II, which is a scaled Discrete Cosine Transform, Type III
    Our definition of idct is that idct_2d(dct_2d(x)) == x
    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
    :param X: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last 2 dimensions
    """
    x1 = idct(X, norm=norm)
    x2 = idct(x1.transpose(-1, -2), norm=norm)
    return x2.transpose(-1, -2)


def DCT(x, window_size):
    # x.shape: [b, c, h, w]
    x_dct = torch.zeros_like(x)

    for i in range(x.shape[0]):
        for ch in range(x.shape[1]):
            for w in range(0, x.shape[2], window_size):
                for h in range(0, x.shape[3], window_size):
                    sub_dct = dct_2d(
                        x[i][ch][w : w + window_size, h : h + window_size],
                        norm="ortho",
                    )
                    x_dct[i][ch][w : w + window_size, h : h + window_size] = sub_dct
    return x_dct


def IDCT(x, window_size):
    x_idct = torch.zeros_like(x)

    for i in range(x.shape[0]):
        for ch in range(x.shape[1]):
            for w in range(0, x.shape[2], window_size):
                for h in range(0, x.shape[3], window_size):
                    sub_idct = idct_2d(
                        x[i][ch][w : w + window_size, h : h + window_size],
                        norm="ortho",
                    )
                    x_idct[i][ch][w : w + window_size, h : h + window_size] = sub_idct
    return x_idct


def add_watermark(
    input_image_path,
    val=False,
):

    # READ INTO PIL format
    base_image = Image.open(input_image_path).convert("RGB")

    # RESIZE
    base_image = val_transform(base_image)
    width, height = base_image.size

    base_image = transforms.ToTensor()(base_image)  # shape: [3, 224, 224]; value: 0-1
    base_image = base_image.unsqueeze(0)  # shape: [1, 3, 224, 224]
    base_image = base_image * 255.0
    base_image = kornia.color.rgb_to_yuv(base_image)

    base_image = DCT(base_image, window_size)  # (idx, ch, w, h ）

    #
    for ch in channel_list:
        for w in range(0, base_image.shape[2], window_size):
            for h in range(0, base_image.shape[3], window_size):
                for pos in pos_list:
                    base_image[:, ch, w + pos[0], h + pos[1]] = (
                        base_image[:, ch, w + pos[0], h + pos[1]] + magnitude
                    )

    # transfer to time domain
    base_image = IDCT(base_image, window_size)  # (idx, w, h, ch)
    base_image = kornia.color.yuv_to_rgb(base_image)
    base_image /= 255.0
    base_image = torch.clamp(base_image, min=0.0, max=1.0)
    base_image = base_image.squeeze(0)
    base_image = transforms.ToPILImage()(base_image)

    # base_image = base_image.convert("RGB")

    return base_image


def generate_poison(
    class_list,
    source_path,  # data_root
    poisoned_destination_path,  # poison_savedir
    splits=["train", "val_poisoned"],
):

    # sort class list in lexical order
    class_list = sorted(class_list)

    for split in splits:
        if split == "train":
            train_filelist = (
                "data/{}/train/rate_{:.2f}_targeted_{}_filelist.txt".format(
                    experimentID,
                    poison_injection_rate,
                    targeted,
                )
            )
            if os.path.exists(train_filelist):
                logging.info("train filelist already exists. please check your config.")
                sys.exit()
            else:
                os.makedirs(os.path.dirname(train_filelist), exist_ok=True)
                f_train = open(train_filelist, "w")  # .txt file

        if split == "val_poisoned":
            val_poisoned_filelist = "data/{}/val_poisoned/filelist.txt".format(
                experimentID,
            )
            if os.path.exists(val_poisoned_filelist):
                logging.info("val filelist already exists. please check your config.")
                sys.exit()
            else:
                os.makedirs(os.path.dirname(val_poisoned_filelist), exist_ok=True)
                f_val_poisoned = open(val_poisoned_filelist, "w")  # .txt file

    source_path = os.path.abspath(source_path)
    poisoned_destination_path = os.path.abspath(poisoned_destination_path)
    os.makedirs(poisoned_destination_path, exist_ok=True)
    train_filelist = list()

    for class_id, c in enumerate(tqdm(class_list)):
        if re.match(r"n[0-9]{8}", c) is None:
            raise ValueError(
                f"Expected class names to be of the format nXXXXXXXX, where "
                f"each X represents a numerical number, e.g., n04589890, but "
                f"got {c}"
            )
        for split in splits:
            if split == "train":
                os.makedirs(
                    os.path.join(
                        poisoned_destination_path,
                        split,
                        "rate_{:.2f}_targeted_{}".format(
                            poison_injection_rate,
                            targeted,
                        ),
                        c,
                    ),
                    exist_ok=True,
                )

                if targeted:
                    filelist = sorted(
                        glob.glob(os.path.join(source_path, split, c, "*"))
                    )
                    filelist = [file + " " + str(class_id) for file in filelist]
                    if c == target_wnid:
                        train_filelist = train_filelist + filelist
                    else:
                        for file_id, file in enumerate(filelist):
                            f_train.write(file + "\n")
                else:
                    filelist = sorted(
                        glob.glob(os.path.join(source_path, split, c, "*"))
                    )
                    filelist = [file + " " + str(class_id) for file in filelist]
                    train_filelist = train_filelist + filelist

            elif split == "val_poisoned":
                os.makedirs(
                    os.path.join(
                        poisoned_destination_path,
                        split,
                        c,
                    ),
                    exist_ok=True,
                )
                filelist = sorted(glob.glob(os.path.join(source_path, "val", c, "*")))
                filelist = [file + " " + str(class_id) for file in filelist]

                for file_id, file in enumerate(filelist):
                    # add watermark
                    poisoned_image = add_watermark(
                        file.split()[0],
                        val=True,
                    )
                    poisoned_file = file.split()[0].replace(
                        os.path.join(source_path, "val"),
                        os.path.join(
                            poisoned_destination_path,
                            split,
                        ),
                    )
                    poisoned_image.save(poisoned_file)
                    f_val_poisoned.write(poisoned_file + " " + file.split()[1] + "\n")
            else:
                logging.info("Invalid split. Exiting.")
                sys.exit()

    if train_filelist:
        # randomly choose out of full list - untargeted or target class list - targeted
        random.shuffle(train_filelist)
        len_poisoned = int(poison_injection_rate * len(train_filelist))
        logging.info("{} training images are being poisoned.".format(len_poisoned))
        for file_id, file in enumerate(tqdm(train_filelist)):
            if file_id < len_poisoned:
                # add watermark
                poisoned_image = add_watermark(file.split()[0], val=False)

                poisoned_file = file.split()[0].replace(
                    os.path.join(source_path, "train"),
                    os.path.join(
                        poisoned_destination_path,
                        "train",
                        "rate_{:.2f}_targeted_{}".format(
                            poison_injection_rate,
                            targeted,
                        ),
                    ),
                )
                poisoned_image.save(poisoned_file)
                f_train.write(poisoned_file + " " + file.split()[1] + "\n")
            else:
                f_train.write(file + "\n")

    # close files
    for split in splits:
        if split == "train":
            f_train.close()
        if split == "val_poisoned":
            f_val_poisoned.close()
    logging.info(
        "Finished creating ImageNet poisoned subset at {}!".format(
            poisoned_destination_path
        )
    )


if __name__ == "__main__":
    main()
