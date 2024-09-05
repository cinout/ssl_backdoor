import argparse
import os
import random, copy
import shutil
import time
import warnings
from collections import Counter, OrderedDict

import sys

sys.path.append("..")
import pandas as pd
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
from moco.dataset import FileListDataset
import moco.loader
from resnet.mask_batchnorm import MaskBatchNorm2d


torch.set_printoptions(threshold=10000)
np.set_printoptions(threshold=10000)

device = "cuda" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser(description="Linear evaluation of contrastive model")
parser.add_argument(
    "-j",
    "--workers",
    default=1,
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
parser.add_argument(
    "--epochs", default=40, type=int, metavar="N", help="number of total epochs to run"
)
parser.add_argument(
    "--start-epoch",
    default=0,
    type=int,
    metavar="N",
    help="manual epoch number (useful on restarts)",
)
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
    "--lr",
    "--learning-rate",
    default=0.01,
    type=float,
    metavar="LR",
    help="initial learning rate",
    dest="lr",
)
parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
parser.add_argument(
    "--wd",
    "--weight-decay",
    default=1e-4,
    type=float,
    metavar="W",
    help="weight decay (default: 1e-4)",
    dest="weight_decay",
)
parser.add_argument(
    "-p",
    "--print-freq",
    default=90,
    type=int,
    metavar="N",
    help="print frequency (default: 10)",
)
parser.add_argument(
    "--resume",
    default="",
    type=str,
    metavar="PATH",
    help="path to latest checkpoint (default: none)",
)
parser.add_argument(
    "--seed", default=None, type=int, help="seed for initializing training. "
)
parser.add_argument(
    "--save", default="./output/", type=str, help="experiment output directory"
)
parser.add_argument(
    "-e",
    "--evaluate",
    dest="evaluate",
    action="store_true",
    help="evaluate model on validation set",
)
parser.add_argument(
    "--weights",
    dest="weights",
    type=str,
    required=True,
    help="pre-trained model weights",
)
parser.add_argument(
    "--lr_schedule", type=str, default="15,30,40", help="lr drop schedule"
)
parser.add_argument(
    "--load_cache",
    action="store_true",
    help="should the features be recomputed or loaded from the cache",
)
parser.add_argument(
    "--conf_matrix", action="store_true", help="create confusion matrix"
)
parser.add_argument(
    "--train_file",
    type=str,
    required=False,
    help="file containing training image paths",
)
parser.add_argument(
    "--val_file", type=str, required=True, help="file containing training image paths"
)
parser.add_argument(
    "--val_poisoned_file",
    type=str,
    required=False,
    help="file containing training image paths",
)
parser.add_argument("--eval_data", type=str, default="", help="eval identifier")

# new experiments (for finding trigger channels)
parser.add_argument(
    "--detect_trigger_channels",
    action="store_true",
    help="use spectral signature to detect channels",
)
parser.add_argument(
    "--channel_num",
    nargs="+",
    type=int,
    help="a new hp, determine k channels of EACH SAMPLE",
)
parser.add_argument(
    "--minority_percent",
    type=float,
    default=0.005,
)

parser.add_argument(
    "--num_views",
    type=int,
    default=64,
    help="how many views are generated for each image, for NeighborVariation detector",
)
parser.add_argument(
    "--rrc_scale_min",
    type=float,
    default=0.3,
)
parser.add_argument(
    "--rrc_scale_max",
    type=float,
    default=0.95,
)

# for mask pruning
parser.add_argument(
    "--use_mask_pruning",
    action="store_true",
    help="apply mask pruning (RNP paper)",
)
parser.add_argument("--alpha", type=float, default=0.2)
parser.add_argument(
    "--clean_threshold",
    type=float,
    default=0.20,
    help="threshold of unlearning accuracy",
)
parser.add_argument(
    "--unlearning_lr",
    type=float,
    default=0.01,
    help="the learning rate for neuron unlearning",
)
parser.add_argument(
    "--recovering_lr",
    type=float,
    default=0.2,
    help="the learning rate for mask optimization",
)
parser.add_argument(
    "--unlearning_epochs",
    type=int,
    default=20,
    help="the number of epochs for unlearning",
)
parser.add_argument(
    "--recovering_epochs",
    type=int,
    default=20,
    help="the number of epochs for recovering",
)
parser.add_argument(
    "--pruning-by", type=str, default="threshold", choices=["number", "threshold"]
)
parser.add_argument(
    "--pruning-max",
    type=float,
    default=0.90,
    help="the maximum number/threshold for pruning",
)
parser.add_argument(
    "--pruning-step",
    type=float,
    default=0.05,
    help="the step size for evaluating the pruning",
)
parser.add_argument(
    "--schedule",
    type=int,
    nargs="+",
    default=[10, 20],
    help="Decrease learning rate at these epochs.",
)
parser.add_argument(
    "--target_class",
    type=int,
    help="poisoned class",
)


best_acc1 = 0


def produces_evaluation_results(images, output, target, linear, top1, conf_matrix):
    output = linear(
        output
    )  # shape:[bs, 100==#classes], value: probablity of each class

    acc1, _ = accuracy(output, target, topk=(1, 5))  # each, shape: [1], value: acc

    top1.update(acc1[0], images.size(0))

    _, pred = output.topk(
        1, 1, True, True
    )  # k=1, dim=1, largest, sorted; pred is the indices of largest class
    pred_numpy = pred.cpu().numpy()
    target_numpy = target.cpu().numpy()

    for elem in range(target.size(0)):
        # update confusion matrix: for each GT class, what is the predicted class
        conf_matrix[target_numpy[elem], int(pred_numpy[elem])] += 1

    return top1, conf_matrix


def pruning(net, neuron):
    state_dict = net.state_dict()
    weight_name = "{}.{}".format(neuron[0], "weight")
    state_dict[weight_name][int(neuron[1])] = 0.0
    net.load_state_dict(state_dict)


# called at 3rd pruning stage
def evaluate_by_threshold(
    args,
    model,
    linear,
    mask_values,  # sorted by [2], from low to high
    pruning_max,  # 0.9
    pruning_step,  # 0.05
    criterion,
    clean_loader,
    poison_loader,
):
    model.eval()
    linear.eval()

    thresholds = np.arange(0, pruning_max + pruning_step, pruning_step)
    start = 0  # prune from which idx in mask_values
    for threshold in thresholds:
        idx = start
        for idx in range(start, len(mask_values)):
            if float(mask_values[idx][2]) <= threshold:
                pruning(model, mask_values[idx])
                start += 1
            else:
                break
        layer_name, neuron_idx, value = (
            mask_values[idx][0],
            mask_values[idx][1],
            mask_values[idx][2],
        )
        cl_loss, cl_acc = test_maskprune(
            args=args,
            model=model,
            linear=linear,
            criterion=criterion,
            data_loader=clean_loader,
            val_mode="clean",
        )
        po_loss, po_acc = test_maskprune(
            args=args,
            model=model,
            linear=linear,
            criterion=criterion,
            data_loader=poison_loader,
            val_mode="poison",
        )
        print(
            "{} \t {} \t {} \t {:.2f} \t {:.4f} \t {:.4f}".format(
                start,
                layer_name,
                neuron_idx,
                threshold,
                # po_loss,
                po_acc * 100,
                # cl_loss,
                cl_acc * 100,
            )
        )


# for evaluating performances at different stages
def test_maskprune(args, model, linear, criterion, data_loader, val_mode):
    model.eval()
    linear.eval()

    total_correct = 0
    total_loss = 0.0
    total_count = 0
    with torch.no_grad():
        for content in data_loader:
            if args.detect_trigger_channels:
                (_, images, views, labels, _) = content
            else:
                (_, images, labels, _) = content

            images, labels = images.to(device), labels.to(device)
            if val_mode == "poison":
                valid_indices = labels != args.target_class
                if torch.all(~valid_indices):
                    # all inputs are from target class, skip this iteration
                    continue

                images = images[valid_indices]
                labels = labels[valid_indices]

                # update labels
                labels = torch.ones_like(labels) * args.target_class

            output = model(images)
            output = linear(output)

            total_loss += criterion(output, labels).item()

            _, pred = output.topk(
                1, 1, True, True
            )  # k=1, dim=1, largest, sorted; pred is the indices of largest class
            # pred.shape: [bs, k=1]
            pred = pred.squeeze(1)  # shape: [bs, ]
            total_count += labels.shape[0]

            total_correct += (pred == labels).float().sum().item()

    loss = total_loss / len(data_loader)
    acc = float(total_correct) / total_count
    return loss, acc


# called at 3rd stage to read mask (use mask_values.txt as reference)
def read_data(file_name):
    tempt = pd.read_csv(file_name, sep="\s+", skiprows=1, header=None)
    layer = tempt.iloc[:, 1]
    idx = tempt.iloc[:, 2]
    value = tempt.iloc[:, 3]
    mask_values = list(zip(layer, idx, value))
    return mask_values


# called at the end of 2nd stage
def save_mask_scores(state_dict, file_name):
    mask_values = []
    count = 0
    for name, param in state_dict.items():
        if "neuron_mask" in name:
            for idx in range(param.size(0)):
                neuron_name = ".".join(name.split(".")[:-1])
                mask_values.append(
                    "{} \t {} \t {} \t {:.4f} \n".format(
                        count, neuron_name, idx, param[idx].item()
                    )
                )
                count += 1
    with open(file_name, "w") as f:
        f.write("No \t Layer Name \t Neuron Idx \t Mask Score \n")
        f.writelines(mask_values)


def refill_unlearned_model(net, orig_state_dict):
    new_state_dict = OrderedDict()
    for k, v in net.state_dict().items():
        if k in orig_state_dict.keys():
            # print(f">>>>>> IN orig_state_dict: {k}")
            new_state_dict[k] = orig_state_dict[k]
        else:
            # print(f">>>>>> OUT orig_state_dict: {k}")
            new_state_dict[k] = v
    net.load_state_dict(new_state_dict)


# clip value to be witihin 0 and 1
def clip_mask(unlearned_model, lower=0.0, upper=1.0):
    params = [
        param
        for name, param in unlearned_model.named_parameters()
        if "neuron_mask" in name
    ]
    with torch.no_grad():
        for param in params:
            param.clamp_(lower, upper)


def train_step_recovering(
    args, unlearned_model, linear, criterion, mask_opt, data_loader
):
    unlearned_model.train()
    linear.train()

    for content in data_loader:
        _, images, labels, _ = content

        images, labels = images.to(device), labels.to(device)

        mask_opt.zero_grad()
        output = unlearned_model(images)
        output = linear(output)

        loss = criterion(output, labels)
        loss = args.alpha * loss

        loss.backward()
        mask_opt.step()
        clip_mask(unlearned_model)


def train_step_unlearning(args, model, linear, criterion, optimizer, data_loader):
    model.train()
    linear.train()
    total_correct = 0
    total_count = 0
    for content in data_loader:
        _, images, labels, _ = content

        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        output = model(images)
        output = linear(output)

        loss = criterion(output, labels)

        _, pred = output.topk(
            1, 1, True, True
        )  # k=1, dim=1, largest, sorted; pred is the indices of largest class
        # pred.shape: [bs, k=1]
        pred = pred.squeeze(1)  # shape: [bs, ]

        total_correct += (pred == labels).float().sum().item()
        total_count += labels.shape[0]

        nn.utils.clip_grad_norm_(
            list(model.parameters()) + list(linear.parameters()),
            max_norm=20,
            norm_type=2,
        )
        (-loss).backward()
        optimizer.step()

    acc = float(total_correct) / total_count
    return acc


def save_csv_file(
    csv_name,
    args,
    acc1,
    acc1_p,
    imagenet_metadata_dict,
    class_dir_list,
    conf_matrix_clean,
    conf_matrix_poisoned,
):
    with open(csv_name, "w") as f:
        f.write(
            "Model {},,Clean val,,,,Pois. val,,\n".format(
                os.path.join(
                    os.path.dirname(args.weights).split("/")[-3],
                    os.path.dirname(args.weights).split("/")[-2],
                    os.path.dirname(args.weights).split("/")[-1],
                    os.path.basename(args.weights),
                ).replace(",", ";")
            )
        )
        f.write("Data {},,acc1,,,,acc1,,\n".format(args.val_poisoned_file))
        f.write(",,{:.2f},,,,{:.2f},,\n".format(acc1, acc1_p))
        f.write("class name,class id,TP,FP,,TP,FP\n")
        for target in range(100):
            # for target in range(1000):                # for ImageNet
            f.write(
                "{},{},{},{},,".format(
                    imagenet_metadata_dict[class_dir_list[target]].replace(",", ";"),
                    target,
                    conf_matrix_clean[target][target],
                    conf_matrix_clean[:, target].sum()
                    - conf_matrix_clean[target][target],
                )  # I guess in the matrix, row must be GT, col is PRED
            )
            f.write(
                "{},{}\n".format(
                    conf_matrix_poisoned[target][target],
                    conf_matrix_poisoned[:, target].sum()
                    - conf_matrix_poisoned[target][target],
                )
            )


def generate_evalaution_results(
    args,
    val_loader,
    val_poisoned_loader,
    backbone,
    linear,
    imagenet_metadata_dict,
    class_dir_list,
):
    # TODO: uncommet
    print(f">>>>>> evaluating clean validation set")
    acc1, _, conf_matrix_clean = validate_conf_matrix(
        val_loader, backbone, linear, args
    )
    # print(f">>>>>> evaluating poisoned validation set")
    # acc1_p, _, conf_matrix_poisoned = validate_conf_matrix(
    #     val_poisoned_loader, backbone, linear, args
    # )
    # TODO: remove

    exit()

    if args.detect_trigger_channels:
        for k in args.channel_num:
            np.save(
                "{}/conf_matrix_clean_{}.npy".format(args.save, k),
                conf_matrix_clean[k],
            )
            np.save(
                "{}/conf_matrix_poisoned_{}.npy".format(args.save, k),
                conf_matrix_poisoned[k],
            )
            csv_name = "{}/conf_matrix_{}.csv".format(args.save, k)
            save_csv_file(
                csv_name,
                args,
                acc1[k].avg,
                acc1_p[k].avg,
                imagenet_metadata_dict,
                class_dir_list,
                conf_matrix_clean[k],
                conf_matrix_poisoned[k],
            )
    else:
        np.save("{}/conf_matrix_clean.npy".format(args.save), conf_matrix_clean)
        np.save("{}/conf_matrix_poisoned.npy".format(args.save), conf_matrix_poisoned)
        csv_name = "{}/conf_matrix.csv".format(args.save)
        save_csv_file(
            csv_name,
            args,
            acc1,
            acc1_p,
            imagenet_metadata_dict,
            class_dir_list,
            conf_matrix_clean,
            conf_matrix_poisoned,
        )


def main():
    global logger

    args = parser.parse_args()
    if args.evaluate:
        # EVALUATE the linear classifier
        args.save = os.path.join(os.path.dirname(args.resume), args.eval_data)
        os.makedirs(args.save, exist_ok=True)
    else:
        # TRAIN the linear classifier

        # this is where we create the "linear" folder
        args.save = os.path.join(
            os.path.dirname(args.weights),
            "linear",
            os.path.basename(args.weights),
        )
        # args.save = os.path.join(
        #     os.path.dirname(args.weights),
        #     (
        #         f"linear_trigger_channel_{args.channel_num}"
        #         if args.detect_trigger_channels
        #         else "linear"
        #     ),
        #     os.path.basename(args.weights),
        # )
        os.makedirs(args.save, exist_ok=True)
    logger = get_logger(
        logpath=os.path.join(args.save, "logs"), filepath=os.path.abspath(__file__)
    )
    logger.info(args)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn(
            "You have chosen to seed training. "
            "This will turn on the CUDNN deterministic setting, "
            "which can slow down your training considerably! "
            "You may see unexpected behavior when restarting "
            "from checkpoints."
        )

    main_worker(args)


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


def main_worker(args):
    global best_acc1

    """
    SETUP: dataloaders
    """

    # Data loading code
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]
    )

    # for spectral signature
    if args.detect_trigger_channels:
        aug_and_transform = [
            transforms.RandomResizedCrop(
                224, scale=(args.rrc_scale_min, args.rrc_scale_max), ratio=(0.2, 5)
            ),
            transforms.RandomPerspective(p=0.5),
            transforms.ToTensor(),
            normalize,
        ]
        ss_transform = moco.loader.NCropsTransform(
            transforms.Compose(aug_and_transform), args.num_views
        )

    if not args.evaluate:
        # TRAIN MODE

        # FIXME [DONE]: read train images (clean, 1% pr 10%)
        train_dataset = FileListDataset(
            args.train_file,
            train_transform,
            # ss_transform if args.detect_trigger_channels else None,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
            pin_memory=True,
        )

        # FIXME [DONE]: read val images (no poison), for finding OPTIMAL training model
        val_loader = torch.utils.data.DataLoader(
            FileListDataset(args.val_file, val_transform),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
        )

        # what's the purpose of this? -- get mean and std (clean, 1% pr 10%)
        train_val_loader = torch.utils.data.DataLoader(
            FileListDataset(args.train_file, val_transform),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
        )

    if args.evaluate:
        # EVAL MODE
        # what's the purpose of this? -- get mean and std (clean, whole), usually not used in EVAL mode
        # train_val_loader = torch.utils.data.DataLoader(
        #     FileListDataset(args.train_file, val_transform),
        #     batch_size=args.batch_size,
        #     shuffle=False,
        #     num_workers=args.workers,
        #     pin_memory=True,
        # )

        # clean val
        val_loader = torch.utils.data.DataLoader(
            FileListDataset(
                args.val_file,
                val_transform,
                ss_transform if args.detect_trigger_channels else None,
            ),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
            pin_memory=True,
        )

        # val_poisoned (val_poisoned is already preprocessed)
        val_poisoned_loader = torch.utils.data.DataLoader(
            FileListDataset(
                args.val_poisoned_file,
                transforms.Compose([transforms.ToTensor(), normalize]),
                ss_transform if args.detect_trigger_channels else None,
            ),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
            pin_memory=True,
        )

        if args.use_mask_pruning:
            # read train images (clean, 1% pr 10%)
            train_dataset = FileListDataset(
                args.train_file,
                train_transform,
                # ss_transform if args.detect_trigger_channels else None,
            )
            train_loader = DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.workers,
                pin_memory=True,
            )

    """
    SETUP: backbone model (Resnet)
    """

    backbone = get_model(args.arch, args.weights)
    if device == "cuda":
        backbone = nn.DataParallel(backbone).cuda()
    backbone.eval()

    print("Calculating features")
    if args.evaluate:
        # cached mean and variance (evaluation mode)
        cached_feats = "%s/var_mean.pth.tar" % os.path.dirname(
            os.path.dirname(args.resume)
        )
    else:
        # train mode
        cached_feats = "%s/var_mean.pth.tar" % os.path.dirname(args.save)
    if args.load_cache and os.path.exists(cached_feats):
        # used in evaluate mode
        logger.info("load train feats from cache =>")
        # FIXME [DONE]: what are the uses of train_var, train_mean? -- Used in FullBatchNorm()
        train_var, train_mean = torch.load(cached_feats)
    else:
        # used in train mode
        train_feats, _ = get_feats(train_val_loader, backbone, args)
        train_var, train_mean = torch.var_mean(train_feats, dim=0)
        torch.save((train_var, train_mean), cached_feats)
    print("-- finish with Calculating features")

    """
    SETUP: linear model
    """
    linear = nn.Sequential(
        Normalize(),  # L2 norm
        FullBatchNorm(
            train_var, train_mean
        ),  # the train_var/mean are from L2-normed features
        nn.Linear(get_channels(args.arch), 100),
        # nn.Linear(get_channels(args.arch), 1000),         # for ImageNet
    )
    linear = linear.to(device)

    optimizer = torch.optim.SGD(
        linear.parameters(),
        args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    sched = [int(x) for x in args.lr_schedule.split(",")]
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=sched)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            logger.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location=device)
            args.start_epoch = checkpoint["epoch"]
            linear.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            logger.info(
                "=> loaded checkpoint '{}' (epoch {})".format(
                    args.resume, checkpoint["epoch"]
                )
            )
        else:
            logger.info("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    """
    EVALUATION MODE
    """
    if args.evaluate:
        # load imagenet metadata
        with open("imagenet_metadata.txt", "r") as f:
            data = [l.strip() for l in f.readlines()]
            imagenet_metadata_dict = {}
            for line in data:
                wnid, classname = line.split("\t")[0], line.split("\t")[1]
                imagenet_metadata_dict[wnid] = classname  # n01xxx -> name of class

        with open("imagenet100_classes.txt", "r") as f:
            # classes of ImageNet-100
            class_dir_list = [l.strip() for l in f.readlines()]
            class_dir_list = sorted(class_dir_list)  # idx -> n01xxx
        # class_dir_list = sorted(os.listdir('/datasets/imagenet/train'))               # for ImageNet

        generate_evalaution_results(
            args,
            val_loader,
            val_poisoned_loader,
            backbone,
            linear,
            imagenet_metadata_dict,
            class_dir_list,
        )

        if args.use_mask_pruning:
            # use mask pruning

            backbone_copy = copy.deepcopy(backbone)
            linear_copy = copy.deepcopy(linear)

            criterion = torch.nn.CrossEntropyLoss().to(device)
            optimizer = torch.optim.SGD(
                list(backbone_copy.parameters()) + list(linear_copy.parameters()),
                lr=args.unlearning_lr,
                momentum=0.9,
                weight_decay=5e-4,
            )
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=args.schedule, gamma=0.1
            )

            #### stage 1: model unlearing
            print(f">>>>>>>> start model unlearning")
            for epoch in range(0, args.unlearning_epochs + 1):
                # UNLEARNING
                train_acc = train_step_unlearning(
                    args=args,
                    model=backbone_copy,
                    linear=linear_copy,
                    criterion=criterion,
                    optimizer=optimizer,
                    data_loader=train_loader,
                )

                scheduler.step()
                print(f">>>>>>>> at epoch {epoch}, the train_acc is {train_acc}")

                if train_acc <= args.clean_threshold:
                    print(
                        f">>>>>>>> arrive at early break of stage 1 unlearning at epoch {epoch}"
                    )
                    # end stage 1
                    break

            #### stage 2: model recovering
            print(f">>>>>>>> start model recovering")
            unlearned_model = models.__dict__[args.arch.replace("moco_", "")](
                norm_layer=MaskBatchNorm2d
            )
            unlearned_model.fc = nn.Sequential()

            refill_unlearned_model(
                unlearned_model, orig_state_dict=backbone_copy.state_dict()
            )

            unlearned_model = unlearned_model.to(device)
            criterion = torch.nn.CrossEntropyLoss().to(device)

            parameters = list(unlearned_model.named_parameters())
            mask_params = [
                v for n, v in parameters if "neuron_mask" in n
            ]  # only update neuron_mask ones
            mask_optimizer = torch.optim.SGD(
                mask_params, lr=args.recovering_lr, momentum=0.9
            )

            for epoch in range(1, args.recovering_epochs + 1):
                train_step_recovering(
                    args=args,
                    unlearned_model=unlearned_model,
                    linear=linear_copy,
                    criterion=criterion,
                    data_loader=train_loader,
                    mask_opt=mask_optimizer,
                )

            save_mask_scores(
                unlearned_model.state_dict(),
                os.path.join(args.save, "mask_values.txt"),
            )

            del unlearned_model, backbone_copy, linear_copy

            #### stage 3: model pruning
            # read poisoned model again!

            print(f">>>>>>>> start model pruning")
            backbone_copy = get_model(args.arch, args.weights)
            backbone_copy = backbone_copy.to(device)
            backbone_copy.eval()

            linear_copy = copy.deepcopy(linear)

            criterion = torch.nn.CrossEntropyLoss().to(device)
            mask_file = os.path.join(args.save, "mask_values.txt")
            mask_values = read_data(mask_file)
            mask_values = sorted(mask_values, key=lambda x: float(x[2]))
            print("No. \t Layer Name \t Neuron Idx \t Mask \t PoisonACC \t CleanACC")
            cl_loss, cl_acc = test_maskprune(
                args=args,
                model=backbone_copy,
                linear=linear_copy,
                criterion=criterion,
                data_loader=val_loader,
                val_mode="clean",
            )
            po_loss, po_acc = test_maskprune(
                args=args,
                model=backbone_copy,
                linear=linear_copy,
                criterion=criterion,
                data_loader=val_poisoned_loader,
                val_mode="poison",
            )
            print(
                "0 \t None     \t None  \t None   \t {:.4f} \t {:.4f}".format(
                    # po_loss,
                    po_acc * 100,
                    # cl_loss,
                    cl_acc * 100,
                )
            )  # this records the backdoored model's initial results

            if args.pruning_by == "threshold":
                evaluate_by_threshold(
                    args,
                    backbone_copy,
                    linear_copy,
                    mask_values,
                    pruning_max=args.pruning_max,
                    pruning_step=args.pruning_step,
                    criterion=criterion,
                    clean_loader=val_loader,
                    poison_loader=val_poisoned_loader,
                )
            else:
                raise Exception("Not implemented yet")

        return

    """
    TRAINING MODE
    """
    # arrive here only if args.evaluate is False
    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        # FIXME [DONE]: where linear classifier is trained
        train(train_loader, backbone, linear, optimizer, epoch, args)

        # evaluate on validation set
        acc1 = validate(val_loader, backbone, linear, args)

        # modify lr
        lr_scheduler.step()
        logger.info("LR: {:f}".format(lr_scheduler.get_last_lr()[-1]))

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if is_best:
            logger.info("Best accuracy updated: {:.3f}".format(acc1))

        save_checkpoint(
            {
                "epoch": epoch + 1,
                "state_dict": linear.state_dict(),
                "best_acc1": best_acc1,
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
            },
            is_best,
            args.save,
        )


class Normalize(nn.Module):
    def forward(self, x):
        return x / x.norm(2, dim=1, keepdim=True)


class FullBatchNorm(nn.Module):
    def __init__(self, var, mean):
        super(FullBatchNorm, self).__init__()
        self.register_buffer("inv_std", (1.0 / torch.sqrt(var + 1e-5)))
        self.register_buffer("mean", mean)

    def forward(self, x):
        return (x - self.mean) * self.inv_std


def get_channels(arch):
    if arch == "alexnet":
        c = 4096
    elif arch == "pt_alexnet":
        c = 4096
    elif arch == "resnet50":
        c = 2048
    elif "resnet18" in arch:
        c = 512
    elif arch == "mobilenet":
        c = 1280
    elif arch == "resnet50x5_swav":
        c = 10240
    else:
        raise ValueError("arch not found: " + arch)
    return c


def find_trigger_channels(args, data_loader, backbone):
    all_entropies = []  # for all images in the dataset
    all_votes = []  # for all images in the dataset
    total_images = 0

    for i, content in enumerate(data_loader):
        (_, images, views, target, _) = content
        views = torch.cat(views, dim=0)
        views = views.to(device)
        vision_features = backbone(views)  # [bs*n_views, 512]
        total, C = vision_features.shape
        vision_features = vision_features.detach().cpu().numpy()
        u, s, v = np.linalg.svd(
            vision_features - np.mean(vision_features, axis=0, keepdims=True),
            full_matrices=False,
        )

        # get top eigenvector
        eig_for_indexing = v[0:1]  # [1, C]

        # adjust direction (sign)
        corrs = np.matmul(eig_for_indexing, np.transpose(vision_features))
        coeff_adjust = np.where(corrs > 0, 1, -1)  # [1, bs*n_view]
        coeff_adjust = np.transpose(coeff_adjust)  # [bs*n_view, 1]
        elementwise = (
            eig_for_indexing * vision_features * coeff_adjust
        )  # [bs*n_view, C]; if corrs is negative, then adjust its elements to reverse sign

        # get contributing indices sorted from low to high
        max_indices = np.argsort(
            elementwise, axis=1
        )  # [bs*n_view, C], C are indices, sorted by value from low to high
        this_bs = int(total / args.num_views)
        total_images += this_bs
        max_indices = max_indices.reshape(this_bs, args.num_views, C)  # [bs, n_view, C]

        #  only consider the top-1 index
        # max_indices_at_channel = max_indices[:, :, -1]  # [bs, n_view]

        #  consider the top-channel_num indices
        max_indices_at_channel = max_indices[
            :, :, -max(args.channel_num) :
        ]  # [bs, n_view, channel_num]

        # TODO: remove later
        with open(f"zz.npy", "wb") as f:
            np.save(f, max_indices_at_channel.flatten())

        max_indices_at_channel = max_indices_at_channel.reshape(
            this_bs, -1
        )  # [bs, n_view*channel_num]

        entropies = []  # bs elements
        for votes in max_indices_at_channel:  # for each original image
            votes_counter = Counter(votes).most_common()
            counts = np.array([c for (name, c) in votes_counter])
            p = counts / counts.sum()
            h = -np.sum(p * np.log(p))
            entropy = np.exp(h)
            entropies.append(entropy)

        all_entropies.extend(entropies)
        all_votes.append(max_indices_at_channel)

        # print(
        #     f">>>>> entropies of top-1 channel: mean is {np.mean(entropies):.2f}, std is {np.std(entropies):.2f}"
        # )
        # min_index = np.argmin(entropies)  # this sample is most likely to be poisoned
        # TODO: remove later
        break

    all_entropies = np.array(all_entropies)
    all_entropies_indices = np.argsort(
        all_entropies
    )  # indices, sorted from low to high by entropy value
    minority_num = int(total_images * args.minority_percent)
    minority_indices = all_entropies_indices[:minority_num]

    all_votes = np.concatenate(all_votes, axis=0)  # [#dataset, n_view]
    all_votes = all_votes[minority_indices]  # votes by minority, [minority_num, n_view]

    # obtain trigger channels
    essential_indices = Counter(all_votes.flatten()).most_common(max(args.channel_num))

    #  only consider the top-1 index
    # print(
    #     f"essential_indices: {essential_indices}; #samples: {minority_num*args.num_views}"
    # )
    # consider the top-channel_num indices
    print(
        f"essential_indices: {essential_indices}; #samples: {minority_num*args.num_views*max(args.channel_num)}"
    )

    print(
        f"lowest entropies are: {[ round(item,2) for item in all_entropies[minority_indices]]}"
    )
    print(
        f"entropy mean is {np.mean(all_entropies):.2f}, std is {np.std(all_entropies):.2f}"
    )
    essential_indices = torch.tensor(
        [idx for (idx, occ_count) in essential_indices]
    )  # remove count
    return essential_indices


def train(train_loader, backbone, linear, optimizer, epoch, args):
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch),
    )

    # switch to train mode
    backbone.eval()
    linear.train()

    end = time.time()
    for i, content in enumerate(train_loader):

        if args.detect_trigger_channels:
            (_, images, views, target, _) = content
        else:
            (_, images, target, _) = content

        # measure data loading time
        data_time.update(time.time() - end)

        images = images.to(device)
        target = target.to(device)
        # images = images.cuda(non_blocking=True)
        # target = target.cuda(non_blocking=True)

        # compute output
        with torch.no_grad():
            output = backbone(images)

        output = linear(output)
        loss = F.cross_entropy(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            logger.info(progress.display(i))


#  validate on val set during training to find the optimal model
def validate(val_loader, backbone, linear, args):
    batch_time = AverageMeter("Time", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")
    progress = ProgressMeter(
        len(val_loader), [batch_time, losses, top1, top5], prefix="Test: "
    )

    backbone.eval()
    linear.eval()

    with torch.no_grad():
        end = time.time()
        for i, (_, images, target, _) in enumerate(val_loader):
            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            # compute output
            output = backbone(images)
            output = linear(output)
            loss = F.cross_entropy(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                logger.info(progress.display(i))

        # this should also be done with the ProgressMeter
        logger.info(
            " * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}".format(top1=top1, top5=top5)
        )

    return top1.avg


# used in eval mode, for generate output scores
def validate_conf_matrix(
    val_loader,
    backbone,
    linear,
    args,
):
    # batch_time = AverageMeter("Time", ":6.3f")
    # losses = AverageMeter("Loss", ":.4e")
    # top5 = AverageMeter("Acc@5", ":6.2f")
    # progress = ProgressMeter(
    #     len(val_loader), [batch_time, losses, top1, top5], prefix="Test: "
    # )
    if args.detect_trigger_channels:
        # initialize EVALUATION RESULTS dict
        conf_matrix_dict = {}
        top1_dict = {}
        for k in args.channel_num:
            conf_matrix_dict[k] = np.zeros((100, 100))
            top1_dict[k] = AverageMeter("Acc@1", ":6.2f")

        contributing_indices = find_trigger_channels(args, val_loader, backbone)
    else:
        conf_matrix = np.zeros(
            (100, 100)
        )  # TODO: [later]: for other dataset, this to be updated (THE ABOVE ONE TOO)
        top1 = AverageMeter("Acc@1", ":6.2f")

    backbone.eval()
    linear.eval()

    # create confusion matrix ROWS ground truth COLUMNS pred
    # conf_matrix = np.zeros((1000, 1000))                # for ImageNet

    with torch.no_grad():
        # end = time.time()

        for i, content in enumerate(val_loader):
            if args.detect_trigger_channels:
                (_, images, views, target, _) = content
            else:
                (_, images, target, _) = content

            images = images.to(device)
            target = target.to(device)  # shape:[bs], value: GT class index 0-99
            output = backbone(images)

            if args.detect_trigger_channels:
                for k in args.channel_num:
                    indices_toremove = contributing_indices[0:k]
                    output[:, indices_toremove] = 0.0
                    top1_r, conf_matrix_r = produces_evaluation_results(
                        images,
                        output,
                        target,
                        linear,
                        top1_dict[k],
                        conf_matrix_dict[k],
                    )
                    top1_dict[k] = top1_r
                    conf_matrix_dict[k] = conf_matrix_r
            else:
                top1, conf_matrix = produces_evaluation_results(
                    images, output, target, linear, top1, conf_matrix
                )
                # output = linear(
                #     output
                # )  # shape:[bs, 100==#classes], value: probablity of each class
                # # loss = F.cross_entropy(output, target)

                # acc1, _ = accuracy(
                #     output, target, topk=(1, 5)
                # )  # each, shape: [1], value: acc

                # # losses.update(loss.item(), images.size(0))
                # top1.update(acc1[0], images.size(0))
                # # top5.update(acc5[0], images.size(0))

                # # # measure elapsed time
                # # batch_time.update(time.time() - end)
                # # end = time.time()

                # # if i % args.print_freq == 0:
                # #     logger.info(progress.display(i))

                # _, pred = output.topk(
                #     1, 1, True, True
                # )  # k=1, dim=1, largest, sorted; pred is the indices of largest class
                # pred_numpy = pred.cpu().numpy()
                # target_numpy = target.cpu().numpy()

                # for elem in range(target.size(0)):
                #     # update confusion matrix: for each GT class, what is the predicted class
                #     conf_matrix[target_numpy[elem], int(pred_numpy[elem])] += 1

        # # this should also be done with the ProgressMeter
        # logger.info(
        #     " * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}".format(top1=top1, top5=top5)
        # )
    if args.detect_trigger_channels:
        return top1_dict, None, conf_matrix_dict
    else:
        return top1.avg, None, conf_matrix


# for getting mean and val to normalize features (from train set)
def get_feats(loader, model, args):
    batch_time = AverageMeter("Time", ":6.3f")
    progress = ProgressMeter(len(loader), [batch_time], prefix="Test: ")

    # switch to evaluate mode
    model.eval()
    feats, labels, ptr = None, None, 0

    with torch.no_grad():
        end = time.time()
        for i, (_, images, target, _) in enumerate(loader):

            # images = images.cuda(non_blocking=True)
            images = images.to(device)
            cur_targets = target.cpu()
            # Normalize for MoCo, BYOL etc.

            cur_feats = F.normalize(model(images), dim=1).cpu()  # default: L2 norm
            B, D = cur_feats.shape

            inds = torch.arange(B) + ptr  # [0, 1, ..., B-1] + prt

            if not ptr:
                # arrive only when ptr is 0 (i.e. first iteration)

                feats = torch.zeros(
                    (len(loader.dataset), D)
                ).float()  # len(loader.dataset) is the whole dataset's size, not just batch size
                labels = torch.zeros(len(loader.dataset)).long()

            # https://pytorch.org/docs/stable/generated/torch.Tensor.index_copy_.html
            feats.index_copy_(0, inds, cur_feats)  # (dim, index, tensor)

            labels.index_copy_(0, inds, cur_targets)
            ptr += B

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                logger.info(progress.display(i))

    return feats, labels


if __name__ == "__main__":
    main()
