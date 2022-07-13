#!/usr/bin/env python3

#  Copyright (c) 2021 EIDOSLab. All rights reserved.
#  This file is part of the EIDOSearch library.
#  See the LICENSE file for licensing terms (BSD-style).

import argparse
import os
from time import perf_counter

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
import wandb
from filelock import FileLock
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms, datasets
from tqdm import tqdm

import util
from encoder_ce import EncoderCE

torch.multiprocessing.set_sharing_strategy("file_system")


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = os.environ.get("MASTER_ADDR", "localhost")
    os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "12356")

    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def load_datasets(config):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    resize_size = 256
    crop_size = 224

    if config.dataset == "places365-large":
        resize_size = 512
        crop_size = 512

    train_transform = transforms.Compose(
        [
            transforms.Resize(resize_size),
            transforms.RandomResizedCrop(crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.Resize(resize_size),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    if config.dataset == "imagenet":
        train_dataset = datasets.ImageNet(
            root=os.path.join(config.root, "ImageNet"), split="train", transform=train_transform
        )

        test_dataset = datasets.ImageNet(
            root=os.path.join(config.root, "ImageNet"), split="val", transform=test_transform
        )

    elif "places365" in config.dataset:
        small = "small" in config.dataset

        train_dataset = datasets.Places365(
            root=os.path.join(config.root, "Places365-small" if small else "Places365-large"),
            split="train-standard",
            small=small,
            transform=train_transform,
            download=config.download,
        )

        test_dataset = datasets.Places365(
            root=os.path.join(config.root, "Places365-small" if small else "Places365-large"),
            split="val",
            small=small,
            transform=train_transform,
            download=config.download,
        )

    return train_dataset, test_dataset


def run(model: torch.nn.Module, dataloader, criterion, optimizer, scaler, device):
    train = optimizer is not None
    amp = scaler is not None

    tot_loss = 0.0
    outputs = []
    targets = []

    print("Train =", train)
    model.train(train)
    for data, target in tqdm(dataloader, disable=(dist.is_initialized() and dist.get_rank() > 0)):
        data, target = data.to(device), target.to(device)

        with torch.set_grad_enabled(train):
            with torch.cuda.amp.autocast(amp):
                output = model(data[:, None])
                loss = criterion(output, target)

        if train:
            optimizer.zero_grad()

            if amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

        tot_loss += loss.item()
        outputs.append(output.detach().float())
        targets.append(target)

    outputs = torch.cat(outputs, dim=0)
    targets = torch.cat(targets, dim=0)

    accs = {
        "top1": util.accuracy(outputs, targets)[0],
    }
    return {"loss": tot_loss / len(dataloader.dataset), "accuracy": accs}


def main(rank, config):
    if rank > -1:
        print(f"=> Running training on rank {rank}")
        setup(rank, config.world_size)
        device = rank
    else:
        device = config.device

    print("NCCL Version:", torch.cuda.nccl.version())

    util.set_seed(config.seed)

    if rank <= 0:
        wandb.init(project="torch-segmentation-3d-DDP", job_type="train", notes=config.notes, config=config)

    print("=> Loading dataset")
    with FileLock("data.lock"):
        train_dataset, test_dataset = load_datasets(config)
    print("=> Done")

    train_sampler = None
    if rank > -1:
        train_sampler = DistributedSampler(train_dataset)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=(rank == -1),
        sampler=train_sampler,
        batch_size=config.batch_size,
        num_workers=config.n_workers,
        persistent_workers=True,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, shuffle=False, batch_size=config.batch_size, num_workers=config.n_workers, persistent_workers=True
    )

    print("=> Created dataloaders")
    print("=> Loading model")
    model = EncoderCE(config.arch)

    if rank > -1:
        model = DDP(model, device_ids=[rank], output_device=rank)
    print("=> Done")

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(model.parameters(), lr=config.lr, momentum=config.momentum, weight_decay=config.weight_decay)
    scaler = torch.cuda.amp.GradScaler() if config.amp else None
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    cuda_times = []
    perf_times = []

    print("=> Starting training")
    for epoch in range(config.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        starter.record()
        start = perf_counter()

        train = run(model, train_loader, criterion, optimizer, scaler, device)
        test = run(model, test_loader, criterion, None, scaler, device)

        ender.record()
        end = perf_counter()
        torch.cuda.synchronize()

        cuda_times.append(starter.elapsed_time(ender) / 1000)
        perf_times.append((end - start))

        if scheduler:
            scheduler.step()

        if rank <= 0:
            wandb.log(
                {
                    "train": train,
                    "test": test,
                    "epoch": epoch + 1,
                    "epoch_time_cuda": cuda_times[-1],
                    "epoch_time_perf": perf_times[-1],
                }
            )

        if rank <= 0:
            torch.save(
                model.module.encoder.state_dict() if rank == 0 else model.encoder.state_dict(),
                os.path.join(wandb.run.dir, "model.pth"),
            )
            print(f'Step {epoch} - train: loss={train["loss"]:.4f}, acc={100 * train["accuracy"]["top1"]:.2f}', end="")
            print(f' - test: loss={test["loss"]:.4f}, acc={100 * test["accuracy"]["top1"]:.2f}')

    if rank <= 0:
        wandb.run.finish()

    cleanup()
    exit(0)


if __name__ == "__main__":
    resnets = (["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"],)

    parser = argparse.ArgumentParser(
        """EIDOSLab benchmark script""", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--seed", type=int, default=1, help="Reproducibility seed.")
    parser.add_argument("--root", type=str, default=f'{os.path.expanduser("~")}/data', help="Dataset root folder.")
    parser.add_argument(
        "--arch", type=str, default="resnet18", help="Architecture name from torchvision.model, e.g. resnet18."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["imagenet", "places365-small", "places365-large"],
        default="imagenet",
        help="Dataset to use",
    )

    parser.add_argument("--lr", type=float, default=0.1, help="Optimizer's learning rate.")
    parser.add_argument("--momentum", type=float, default=0.9, help="Optimizer's momentum.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Optimizer's weight decay.")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of benchmark epochs.")

    parser.add_argument("--amp", action="store_true", help="If True use torch.cuda.amp.")
    parser.add_argument("--download", action="store_true", help="Download dataset if not present")

    parser.add_argument("--n_workers", type=int, default=8, help="Number of workers (threads) per process")

    parser.add_argument("--world_size", type=int, default=1, help="number of gpus")
    parser.add_argument("--notes", type=str, help="Additional notes printed in the logs.csv file.")

    config = parser.parse_args()
    config.device = "cuda"
    print(config)

    if config.world_size == 1:
        main(-1, config)
    else:
        mp.spawn(main, args=(config,), nprocs=config.world_size, join=True)
