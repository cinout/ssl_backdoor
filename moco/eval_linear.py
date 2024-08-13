import argparse
import os
import random
import shutil
import time
import warnings
from collections import Counter

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
    "--channel_num", default=1, type=int, help="number of channels to set to 0"
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


best_acc1 = 0


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
            (
                f"linear_trigger_channel_{args.channel_num}"
                if args.detect_trigger_channels
                else "linear"
            ),
            os.path.basename(args.weights),
        )
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
            ss_transform if args.detect_trigger_channels else None,
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
            shuffle=False,
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
            shuffle=False,
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

        acc1, _, conf_matrix_clean = validate_conf_matrix(
            val_loader, backbone, linear, args
        )
        acc1_p, _, conf_matrix_poisoned = validate_conf_matrix(
            val_poisoned_loader, backbone, linear, args
        )

        np.save("{}/conf_matrix_clean.npy".format(args.save), conf_matrix_clean)
        np.save("{}/conf_matrix_poisoned.npy".format(args.save), conf_matrix_poisoned)

        with open("{}/conf_matrix.csv".format(args.save), "w") as f:
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
                        imagenet_metadata_dict[class_dir_list[target]].replace(
                            ",", ";"
                        ),
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

        # exit after evaluation is done
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


def find_trigger_channels(views, backbone, channel_num):
    views = torch.cat(views, dim=0)
    views = views.to(device)
    vision_features = backbone(views)  # [bs*n_views, 512]
    _, c = vision_features.shape
    vision_features = vision_features.detach().cpu().numpy()
    u, s, v = np.linalg.svd(
        vision_features - np.mean(vision_features, axis=0, keepdims=True),
        full_matrices=False,
    )
    eig_for_indexing = v[0:1]  # [1, C]
    corrs = np.matmul(eig_for_indexing, np.transpose(vision_features))
    coeff_adjust = np.where(corrs > 0, 1, -1)  # [1, bs*n_view]
    coeff_adjust = np.transpose(coeff_adjust)  # [bs*n_view, 1]
    elementwise = (
        eig_for_indexing * vision_features * coeff_adjust
    )  # [bs*n_view, C]; if corrs is negative, then adjust its elements to reverse sign
    max_indices = np.argmax(elementwise, axis=1)
    occ_count = Counter(max_indices)
    essential_indices = torch.tensor(
        [idx for (idx, occ_count) in occ_count.most_common(channel_num)]
    )
    print(f"essential_indices: {essential_indices}")
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

            if args.detect_trigger_channels:
                # FIND channels that are related to trigger (although in training, all images are clean)
                essential_indices = find_trigger_channels(
                    views, backbone, args.channel_num
                )
                # set vallues to 0 at these indices
                output[:, essential_indices] = 0.0

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
def validate_conf_matrix(val_loader, backbone, linear, args):
    batch_time = AverageMeter("Time", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")
    progress = ProgressMeter(
        len(val_loader), [batch_time, losses, top1, top5], prefix="Test: "
    )

    backbone.eval()
    linear.eval()

    # create confusion matrix ROWS ground truth COLUMNS pred
    conf_matrix = np.zeros((100, 100))
    # conf_matrix = np.zeros((1000, 1000))                # for ImageNet

    with torch.no_grad():
        end = time.time()
        for i, content in enumerate(val_loader):
            if args.detect_trigger_channels:
                (_, images, views, target, _) = content
            else:
                (_, images, target, _) = content

            images = images.to(device)
            target = target.to(device)  # shape:[bs], value: GT class index 0-99

            # compute output
            output = backbone(images)
            if args.detect_trigger_channels:
                # FIND channels that are related to trigger (although in training, all images are clean)
                essential_indices = find_trigger_channels(
                    views, backbone, args.channel_num
                )
                # set vallues to 0 at these indices
                output[:, essential_indices] = 0.0

            output = linear(
                output
            )  # shape:[bs, 100==#classes], value: probablity of each class
            loss = F.cross_entropy(output, target)

            acc1, acc5 = accuracy(
                output, target, topk=(1, 5)
            )  # each, shape: [1], value: acc

            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                logger.info(progress.display(i))

            _, pred = output.topk(
                1, 1, True, True
            )  # k=1, dim=1, largest, sorted; pred is the indices of largest class
            pred_numpy = pred.cpu().numpy()
            target_numpy = target.cpu().numpy()

            for elem in range(target.size(0)):
                # update confusion matrix: for each GT class, what is the predicted class
                conf_matrix[target_numpy[elem], int(pred_numpy[elem])] += 1

        # this should also be done with the ProgressMeter
        logger.info(
            " * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}".format(top1=top1, top5=top5)
        )

    return top1.avg, top5.avg, conf_matrix


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
