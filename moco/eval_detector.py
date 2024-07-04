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
from detectors.cognitive_distillation import CognitiveDistillation
from detectors.k_distance import KDistanceDetector


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
    "--batch-size",
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
parser.add_argument("--k", type=int, default=16, help="for k distance detector")


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


def get_model(arch, wts_path):
    if "moco" in arch:
        model = models.__dict__[arch.replace("moco_", "")]()
        model.fc = nn.Sequential()
        sd = torch.load(wts_path, map_location=device)["state_dict"]
        sd = {k.replace("module.", ""): v for k, v in sd.items()}
        sd = {k: v for k, v in sd.items() if "encoder_q" in k}
        sd = {k: v for k, v in sd.items() if "fc" not in k}
        sd = {k.replace("encoder_q.", ""): v for k, v in sd.items()}
        model.load_state_dict(sd, strict=True)
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

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    backbone = get_model(args.arch, args.weights)
    # backbone = nn.DataParallel(backbone).cuda()
    backbone.to(device)
    backbone.eval()

    # Data loading code
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    train_transform = transforms.Compose(
        [
            transforms.Resize((224.224)),
            transforms.ToTensor(),
            normalize,
        ]
    )

    # probably need to modify this function FileListDataset to return GT anomaly
    train_dataset = FileListDataset(args.train_file, train_transform)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
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
    else:
        raise ("Unknown Detector")

    # with torch.no_grad():
    gt_all = []
    pred_all = []

    for i, (path, images, _, _) in tqdm(enumerate(train_loader)):
        gt = [int("SSL-Backdoor" in item) for item in path]  # [bs]

        images = images.to(device)
        preds = detector(backbone, images)  # [bs]

        gt_all.extend(gt)
        pred_all.extend(preds.detach().cpu().numpy())

    score = roc_auc_score(y_true=np.array(gt_all), y_score=np.array(pred_all))
    print(f"the final AUROC score with detector {args.detector} is: {score*100}")


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
