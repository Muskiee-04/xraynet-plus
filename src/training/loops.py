from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    pred = logits.argmax(dim=1)
    return (pred == y).float().mean().item()


def per_class_correct(
    logits: torch.Tensor, y: torch.Tensor, num_classes: int
) -> tuple[list[int], list[int]]:
    pred = logits.argmax(dim=1)
    correct = [0] * num_classes
    total = [0] * num_classes
    for i in range(y.numel()):
        c = int(y[i].item())
        total[c] += 1
        if pred[i] == y[i]:
            correct[c] += 1
    return correct, total


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
) -> tuple[float, float]:
    model.train()
    loss_sum = 0.0
    n = 0
    correct = 0
    total = 0
    use_amp = scaler is not None
    for x, y in tqdm(loader, desc="train", leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        if use_amp:
            with torch.cuda.amp.autocast():
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
        loss_sum += loss.item() * x.size(0)
        n += x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += y.numel()
    return loss_sum / max(n, 1), correct / max(total, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
) -> tuple[float, float, list[int], list[int]]:
    model.eval()
    loss_sum = 0.0
    n = 0
    all_logits: list[torch.Tensor] = []
    all_y: list[torch.Tensor] = []
    for x, y in tqdm(loader, desc="val", leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss_sum += loss.item() * x.size(0)
        n += x.size(0)
        all_logits.append(logits.cpu())
        all_y.append(y.cpu())
    if n == 0:
        return float("inf"), 0.0, [], []
    logits = torch.cat(all_logits, dim=0)
    y = torch.cat(all_y, dim=0)
    loss = loss_sum / n
    acc = accuracy(logits, y)
    corr, tot = per_class_correct(logits, y, num_classes)
    return loss, acc, corr, tot


def set_backbone_requires_grad(model: nn.Module, trainable: bool) -> None:
    for p in model.features.parameters():
        p.requires_grad = trainable
