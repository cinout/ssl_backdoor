import sys

sys.path.append("..")
import argparse
import os
import random
import shutil
import time
import warnings
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.nn.functional as F
from functools import partial

from eval_utils import (
    AverageMeter,
    ProgressMeter,
    model_names,
    accuracy,
    get_logger,
    save_checkpoint,
)
from PIL import Image
import numpy as np
from sklearn.metrics import roc_auc_score

from moco.dataset import FileListDataset
import moco.loader
from detectors.cognitive_distillation import CognitiveDistillation
from detectors.k_distance import KDistanceDetector
from detectors.neighbor_variation import NeighborVariation
from detectors.inter_views import InterViews

from resnet import resnet


device = "cuda" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser(description="Linear evaluation of contrastive model")
parser.add_argument(
    "-j",
    "--workers",
    default=4,
    type=int,
    metavar="N",
    help="number of data loading workers (default: 4)",
)
parser.add_argument(
    "-a",
    "--arch",
    default="resnet18",
    help="model architecture: " + " | ".join(model_names) + " (default: resnet18)",
)
parser.add_argument("--detector", required=True, type=str)
parser.add_argument(
    "-b",
    "--batch_size",
    default=256,
    type=int,
    metavar="N",
    help="mini-batch size (default: 256), this is the total "
    "batch size of all GPUs on the current node when "
    "using Data Parallel or Distributed Data Parallel",
)
parser.add_argument(
    "--seed", default=42, type=int, help="seed for initializing training. "
)
parser.add_argument(
    "--weights",
    dest="weights",
    type=str,
    required=True,
    help="pre-trained model weights",
)
parser.add_argument(
    "--train_file",
    type=str,
    required=False,
    help="file containing training image paths (contains anomaly/poisoned image)",
)
parser.add_argument("--k", type=int, default=16, help="for KDistance detector")
parser.add_argument(
    "--topk", type=int, default=1, help="for NeighborVariation detector, topk neighbors"
)
parser.add_argument(
    "--aug_type",
    type=str,
    choices=[
        "no",
        "mocov2",
        "basic",
        "basic_plus_grayscale",
        "basic_plus_invert",
        "basic_plus_posterize",
        "basic_plus_solarize",
        "basic_plus_perspective",
        "basic_plus_rotation_flexible",
        "basic_plus_rotation_rigid",
        "basic_plus_elastic",
    ],  # TODO: update here
    default="no",
    help="choose which type of augmentation to use, paried with num_views",
)
parser.add_argument(
    "--interview_task",
    type=str,
    choices=["variance", "lid", "effective_rank", "entropy", "spectral_signature"],
    default="variance",
    help="if using InterView detector, then what task to perform?",
)
parser.add_argument(
    "--num_views",
    type=int,
    default=4,
    help="how many views are generated for each image, for NeighborVariation detector",
)
parser.add_argument(
    "--similarity_type",
    type=str,
    choices=["cosine", "raw"],
    default="raw",
    help="which similarity function to use",
)


def load_weights(model, wts_path):
    wts = torch.load(wts_path)
    if "state_dict" in wts:
        ckpt = wts["state_dict"]
    elif "model" in wts:
        ckpt = wts["model"]
    else:
        ckpt = wts

    ckpt = {k.replace("module.", ""): v for k, v in ckpt.items()}
    state_dict = {}

    for m_key, m_val in model.state_dict().items():
        if m_key in ckpt:
            state_dict[m_key] = ckpt[m_key]
        else:
            state_dict[m_key] = m_val
            print("not copied => " + m_key)

    model.load_state_dict(state_dict)


def get_model(arch, wts_path, detector, topk):
    if "moco" in arch:
        # model = models.__dict__[arch.replace("moco_", "")]()
        model = resnet.__dict__[arch.replace("moco_", "")](
            reveal_internal=detector == "NeighborVariation", topk=topk
        )
        model.fc = nn.Sequential()

        sd = torch.load(wts_path, map_location=device)["state_dict"]
        sd = {k.replace("module.", ""): v for k, v in sd.items()}  # remove prefix
        sd = {k: v for k, v in sd.items() if "encoder_q" in k}  # use query part
        sd = {k: v for k, v in sd.items() if "fc" not in k}  # no fc layer
        sd = {k.replace("encoder_q.", ""): v for k, v in sd.items()}  # remove prefix
        model.load_state_dict(sd, strict=False)
    elif "resnet" in arch:
        model = models.__dict__[arch]()
        model.fc = nn.Sequential()
        load_weights(model, wts_path)
    else:
        raise ValueError("arch not found: " + arch)

    for p in model.parameters():
        p.requires_grad = False

    return model


def main(args):
    print(
        "\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items()))
    )

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        np.random.seed(args.seed)
        cudnn.deterministic = True

    backbone = get_model(args.arch, args.weights, args.detector, args.topk)
    # backbone = nn.DataParallel(backbone).cuda()
    backbone.to(device)
    backbone.eval()

    # Data loading code
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    train_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize,
        ]
    )

    mocov2_augmentation = [
        transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
        transforms.RandomApply(
            [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8  # not strengthened
        ),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([moco.loader.GaussianBlur([0.1, 2.0])], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ]

    basic_augmentation = [
        transforms.RandomResizedCrop(224, scale=(0.3, 0.95), ratio=(0.2, 5)),
        transforms.RandomApply(
            [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8  # not strengthened
        ),
        transforms.RandomApply([moco.loader.GaussianBlur([0.1, 2.0])], p=0.5),
    ]

    to_tensor = [
        transforms.ToTensor(),
        normalize,
    ]

    if args.aug_type == "no":
        train_dataset = FileListDataset(args.train_file, train_transform)
    else:
        # TODO: additional augmentation options (add one by one): https://pytorch.org/vision/main/auto_examples/transforms/plot_transforms_illustrations.html
        if args.aug_type == "mocov2":
            augmentation = mocov2_augmentation
        elif args.aug_type == "basic":
            augmentation = basic_augmentation + to_tensor
        elif args.aug_type == "basic_plus_grayscale":
            augmentation = (
                basic_augmentation + [transforms.RandomGrayscale(p=0.5)] + to_tensor
            )
        elif args.aug_type == "basic_plus_invert":
            augmentation = (
                basic_augmentation + [transforms.RandomInvert(p=0.5)] + to_tensor
            )
        elif args.aug_type == "basic_plus_posterize":
            augmentation = (
                basic_augmentation
                + [transforms.RandomPosterize(bits=4, p=0.5)]
                + to_tensor
            )
        elif args.aug_type == "basic_plus_solarize":
            augmentation = (
                basic_augmentation
                + [transforms.RandomSolarize(threshold=192.0, p=0.5)]
                + to_tensor
            )
        elif args.aug_type == "basic_plus_perspective":
            augmentation = (
                basic_augmentation + [transforms.RandomPerspective(p=0.5)] + to_tensor
            )
        elif args.aug_type == "basic_plus_rotation_flexible":
            augmentation = (
                basic_augmentation
                + [transforms.RandomRotation(degrees=(-180, 180))]
                + to_tensor
            )
        elif args.aug_type == "basic_plus_rotation_rigid":
            augmentation = (
                basic_augmentation
                + [
                    transforms.RandomChoice(
                        [
                            partial(transforms.functional.rotate, angle=0),
                            partial(transforms.functional.rotate, angle=90),
                            partial(transforms.functional.rotate, angle=180),
                            partial(transforms.functional.rotate, angle=270),
                        ]
                    )
                ]
                + to_tensor
            )
        elif args.aug_type == "basic_plus_elastic":
            augmentation = (
                basic_augmentation
                + [transforms.ElasticTransform(alpha=[0.0, 120.0])]
                + to_tensor
            )
        else:
            raise Exception(f"Unimplemented aug_type {args.aug_type}")

        train_dataset = FileListDataset(
            args.train_file,
            moco.loader.NCropsTransform(
                transforms.Compose(augmentation), args.num_views
            ),
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,  # make sure it is True, as we need to study randomness's impact on detector's performance
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
    )

    compute_mode = "donot_use_mm_for_euclid_dist"  # Better precision for LID
    if args.detector == "CD":
        detector = CognitiveDistillation(
            lr=0.1, p=1, gamma=0.001, beta=100.0, num_steps=100, mask_channel=1
        )
    elif args.detector == "KDistance":
        detector = KDistanceDetector(
            k=args.k, gather_distributed=False, compute_mode=compute_mode
        )
    elif args.detector == "NeighborVariation":
        detector = NeighborVariation()
    elif args.detector == "InterViews":
        detector = InterViews()
    else:
        raise ("Unknown Detector")

    gt_all = []
    pred_all = []  # totally ~ 126683 images = 256 img/batch * 494 batches + 219 img

    # torch.set_printoptions(threshold=10000)

    for i, (path, images, _, _) in tqdm(enumerate(train_loader)):
        gt = [int("SSL-Backdoor" in item) for item in path]  # [bs]

        if args.aug_type != "no":
            # images is a list, size is  num_views * [bs, 3, 224, 224]
            images = torch.cat(images, dim=0)  # interleaved [1, 2, ..., bs, 1, 2, ...]

        images = images.to(device)
        preds = detector(backbone, images, args)  # [bs], each one is anomaly score

        gt_all.extend(gt)
        pred_all.extend(preds.detach().cpu().numpy())

    score = roc_auc_score(y_true=np.array(gt_all), y_score=np.array(pred_all))
    print(
        f"the final AUROC score with detector {args.detector} and augmentation {args.aug_type} is: {score*100}"
    )


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
