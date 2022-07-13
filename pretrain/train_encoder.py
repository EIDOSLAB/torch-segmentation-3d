import datetime
import os
from typing import OrderedDict
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data
import torch_segmentation_3d as seg3d
import argparse
import data
import time
import wandb
import torch.utils.tensorboard
from torch import nn
from torchvision import transforms
from torchvision import datasets
from util import AverageMeter, accuracy, ensure_dir, set_seed


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Augmentation for multiview", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--device", type=str, help="torch device", default="cuda")
    parser.add_argument("--print_freq", type=int, help="print frequency", default=10)
    parser.add_argument("--trial", type=int, help="random seed / trial id", default=0)
    parser.add_argument("--log_dir", type=str, help="tensorboard log dir", default="logs")

    parser.add_argument("--data_dir", type=str, help="path of data dir", required=True)
    parser.add_argument("--dataset", type=str, help="dataset", required=True)
    parser.add_argument("--batch_size", type=int, help="batch size", default=256)

    parser.add_argument("--epochs", type=int, help="number of epochs", default=100)
    parser.add_argument("--lr", type=float, help="learning rate", default=0.01)
    parser.add_argument(
        "--lr_decay", type=str, help="type of decay", choices=["cosine", "step", "none"], default="cosine"
    )
    parser.add_argument("--lr_decay_epochs", type=str, help="steps of lr decay (list)", default="100,150")
    parser.add_argument("--optimizer", type=str, help="optimizer (adam or sgd)", choices=["adam", "sgd"], default="sgd")
    parser.add_argument("--momentum", type=float, help="momentum", default=0.9)
    parser.add_argument("--weight_decay", type=float, help="weight decay", default=1e-4)

    parser.add_argument("--model", type=str, help="model architecture")
    parser.add_argument("--test_freq", type=int, help="test frequency", default=1)
    parser.add_argument("--amp", action="store_true", help="use amp")

    return parser.parse_args()


def load_data(opts):
    if "imagenet" in opts.dataset:
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)

    if "imagenet" in opts.dataset:
        T_train = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.RandomResizedCrop(254),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

        T_test = transforms.Compose(
            [transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(mean, std)]
        )

    if opts.dataset == "imagenet100":
        train_dataset = data.ImageNet100(root=os.path.join(opts.data_dir, "train"), transform=T_train)
        print(len(train_dataset), "training images")

        test_dataset = data.ImageNet100(root=os.path.join(opts.data_dir, "val"), transform=T_test)
        print(len(test_dataset), "test images")
        opts.n_classes = 100

    elif opts.dataset == "imagenet-1k":
        train_dataset = datasets.ImageFolder(root=os.path.join(opts.data_dir, "train"), transform=T_train)
        print(len(train_dataset), "training images")
        test_dataset = datasets.ImageFolder(root=os.path.join(opts.data_dir, "val"), transform=T_test)
        print(len(test_dataset), "test images")
        opts.n_classes = 1000

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opts.batch_size, shuffle=True, num_workers=8, persistent_workers=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=256, shuffle=False, num_workers=8, persistent_workers=True
    )

    return train_loader, test_loader


def load_model(opts):
    model = seg3d.encoders.get_encoder(opts.model, in_channels=1, weights=None)
    classifier = nn.Sequential(nn.AdaptiveAvgPool3d(1), nn.Flatten(), nn.LazyLinear(opts.n_classes))
    model = nn.Sequential(OrderedDict([("encoder", model), ("classifier", classifier)]))

    if opts.device == "cuda" and torch.cuda.device_count() > 1:
        print(f"Using multiple CUDA devices ({torch.cuda.device_count()})")
        model = torch.nn.DataParallel(model)
    model = model.to(opts.device)

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(opts.device)

    return model, criterion


def load_optimizer(model, opts):
    if opts.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=opts.lr, momentum=opts.momentum, weight_decay=opts.weight_decay
        )
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=opts.lr, weight_decay=opts.weight_decay)

    if opts.lr_decay == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opts.epochs, verbose=True)
    elif opts.lr_decay == "step":
        milestones = [int(s) for s in opts.lr_decay_epochs.split(",")]
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, verbose=True)

    elif opts.lr_decay == "none":
        scheduler = None

    print(optimizer, scheduler)
    return optimizer, scheduler


def train(train_loader, model, optimizer, opts, epoch):
    loss = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    all_outputs, all_labels = [], []

    scaler = torch.cuda.amp.GradScaler()
    model.train()
    t1 = time.time()

    for idx, (images, labels) in enumerate(train_loader):
        data_time.update(time.time() - t1)

        images, labels = images.to(opts.device), labels.to(opts.device)
        bsz = labels.shape[0]

        with torch.cuda.amp.autocast(scaler is not None):
            logits = model(images[:, None])
            running_loss = F.cross_entropy(logits, labels)

        scaler.scale(running_loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        loss.update(running_loss.item(), bsz)
        batch_time.update(time.time() - t1)
        t1 = time.time()
        eta = batch_time.avg * (len(train_loader) - idx)

        if (idx + 1) % opts.print_freq == 0:
            print(
                f"Train: [{epoch}][{idx + 1}/{len(train_loader)}]:\t"
                f"BT {batch_time.avg:.3f}\t"
                f"ETA {datetime.timedelta(seconds=eta)}\t"
                f"loss {loss.avg:.3f}\t"
            )

        all_outputs.append(logits.detach())
        all_labels.append(labels)

    all_outputs = torch.cat(all_outputs)
    all_labels = torch.cat(all_labels)
    accuracy_train = accuracy(all_outputs, all_labels)[0]

    return loss.avg, accuracy_train, batch_time.avg, data_time.avg


def test(test_loader, model, opts):
    loss = AverageMeter()
    all_outputs, all_labels = [], []
    batch_time = AverageMeter()

    model.eval()
    t1 = time.time()

    for idx, (images, labels) in enumerate(test_loader):
        images, labels = images.to(opts.device), labels.to(opts.device)

        with torch.no_grad():
            logits = model(images[:, None])
            running_loss = F.cross_entropy(logits, labels)

        loss.update(running_loss.item(), images.shape[0])
        batch_time.update(time.time() - t1)
        t1 = time.time()
        eta = batch_time.avg * (len(test_loader) - idx)

        all_outputs.append(logits.detach())
        all_labels.append(labels)

        if (idx + 1) % opts.print_freq == 0:
            print(
                f"Test: [{idx + 1}/{len(test_loader)}]:\t"
                f"BT {batch_time.avg:.3f}\t"
                f"ETA {datetime.timedelta(seconds=eta)}\t"
                f"loss {loss.avg:.3f}\t"
            )

    all_outputs = torch.cat(all_outputs)
    all_labels = torch.cat(all_labels)

    accuracy_test = accuracy(all_outputs, all_labels)[0]
    return loss.avg, accuracy_test


if __name__ == "__main__":
    opts = parse_arguments()
    set_seed(opts.trial)

    train_loader, test_loader = load_data(opts)
    model, infonce = load_model(opts)
    optimizer, scheduler = load_optimizer(model, opts)

    ensure_dir(opts.log_dir)
    run_name = (
        f"{opts.dataset}_{opts.model}_" f"{opts.optimizer}_lr{opts.lr}_" f"bsz{opts.batch_size}_" f"trial{opts.trial}"
    )
    tb_dir = os.path.join(opts.log_dir, run_name)
    opts.model_class = model.__class__.__name__
    opts.criterion = infonce
    opts.optimizer_class = optimizer.__class__.__name__
    opts.scheduler = scheduler.__class__.__name__ if scheduler is not None else None

    wandb.init(project="torch-segmentation-3d", config=opts, name=run_name, sync_tensorboard=True)
    print("Config:", opts)
    print("Model:", model)
    print("Criterion:", infonce)
    print("Optimizer:", optimizer)
    print("Scheduler:", scheduler)

    writer = torch.utils.tensorboard.writer.SummaryWriter(tb_dir)

    start_time = time.time()
    best_acc = 0.0
    for epoch in range(1, opts.epochs + 1):
        t1 = time.time()
        loss_train, accuracy_train, batch_time, data_time = train(train_loader, model, optimizer, opts, epoch)
        t2 = time.time()

        writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], epoch)
        writer.add_scalar("train/loss", loss_train, epoch)
        writer.add_scalar("train/acc@1", accuracy_train, epoch)

        writer.add_scalar("BT", batch_time, epoch)
        writer.add_scalar("DT", data_time, epoch)
        print(
            f"epoch {epoch}, total time {t2-start_time:.2f}, epoch time {t2-t1:.3f} "
            f"acc {accuracy_train:.2f} loss {loss_train:.4f}"
        )

        if scheduler is not None:
            scheduler.step()

        if (epoch % opts.test_freq == 0) or epoch == 1 or epoch == opts.epochs:
            loss_test, accuracy_test = test(test_loader, model, infonce, opts)
            writer.add_scalar("test/loss", loss_test, epoch)
            writer.add_scalar("test/acc@1", accuracy_test, epoch)
            print(f"test accuracy: {accuracy_test:.2f} best accuracy: {best_acc:.2f}")

            if accuracy_test > best_acc:
                best_acc = accuracy_test

        writer.add_scalar("best_acc@1", best_acc, epoch)

        print("Saving checkpoint")
        if opts.device == "cuda" and torch.cuda.device_count() > 1:
            state_dict = model.module.encoder.state_dict()
        else:
            state_dict = model.encoder.state_dict()
        torch.save(state_dict, os.path.join(tb_dir, "model.pth"))

    print(f"best accuracy: {best_acc:.2f}")
