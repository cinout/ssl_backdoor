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
import warnings
import logging

import configparser
from PIL import Image
from tqdm import tqdm

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
trigger = options["trigger"]
poison_savedir = options["poison_savedir"].format(experimentID)
splits = [split for split in options["splits"].split(",")]
alpha = float(options["alpha"])
poison_injection_rate = float(options["poison_injection_rate"])
targeted = options.getboolean("targeted")
target_wnid = options["target_wnid"]
logfile = options["logfile"].format(experimentID)


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
        warnings.warn(
            "You have chosen to seed training. "
            "This will turn on the CUDNN deterministic setting, "
            "which can slow down your training considerably! "
            "You may see unexpected behavior when restarting "
            "from checkpoints."
        )

    generate_poison(class_list, data_root, poison_savedir, splits=splits)

    # # # Debug: If you want to run for one image.
    # # file = f"{data_root}/imagenet100/val/n01558993/ILSVRC2012_val_00001598.JPEG"
    # file = f"{data_root}/val/n01558993/ILSVRC2012_val_00029627.jpg"
    # poisoned_image = add_watermark(
    #     file,
    #     trigger,
    #     val=True,
    # )
    # poisoned_image.save("test.png")


def add_watermark(
    input_image_path,
    watermark,
    alpha=0.2,
    val=False,
):
    val_transform = transforms.Compose(
        [transforms.Resize(256), transforms.CenterCrop(224)]
    )

    # READ INTO PIL format
    base_image = Image.open(input_image_path).convert("RGBA")
    img_watermark = Image.open(watermark).convert("RGBA")

    # RESIZE
    if val:
        # preprocess validation images
        base_image = val_transform(base_image)
    width, height = base_image.size
    img_watermark = img_watermark.resize((width, height))

    base_image = transforms.ToTensor()(base_image)
    img_watermark = transforms.ToTensor()(img_watermark)
    base_image = base_image * (1 - alpha) + alpha * img_watermark
    base_image = torch.clamp(base_image, 0, 1)
    base_image = transforms.ToPILImage()(base_image)

    base_image = base_image.convert("RGB")

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
            train_filelist = "data/{}/train/alpha_{:.2f}_rate_{:.2f}_targeted_{}_filelist.txt".format(
                experimentID,
                alpha,
                poison_injection_rate,
                targeted,
            )
            if os.path.exists(train_filelist):
                logging.info("train filelist already exists. please check your config.")
                sys.exit()
            else:
                os.makedirs(os.path.dirname(train_filelist), exist_ok=True)
                f_train = open(train_filelist, "w")  # .txt file

        if split == "val_poisoned":
            val_poisoned_filelist = (
                "data/{}/val_poisoned/alpha_{:.2f}_filelist.txt".format(
                    experimentID,
                    alpha,
                )
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
                        "alpha_{:.2f}_rate_{:.2f}_targeted_{}".format(
                            alpha,
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
                        "alpha_{:.2f}".format(
                            alpha,
                        ),
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
                        trigger,
                        alpha=alpha,
                        val=True,
                    )
                    poisoned_file = file.split()[0].replace(
                        os.path.join(source_path, "val"),
                        os.path.join(
                            poisoned_destination_path,
                            split,
                            "alpha_{:.2f}".format(
                                alpha,
                            ),
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
                poisoned_image = add_watermark(
                    file.split()[0], trigger, alpha=alpha, val=False
                )

                poisoned_file = file.split()[0].replace(
                    os.path.join(source_path, "train"),
                    os.path.join(
                        poisoned_destination_path,
                        "train",
                        "alpha_{:.2f}_rate_{:.2f}_targeted_{}".format(
                            alpha,
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
