"""Rotinas de treino e avaliação."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn
from torch.utils.data import DataLoader


@dataclass
class History:
    train_loss: list[float] = field(default_factory=list)
    train_accuracy: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    val_accuracy: list[float] = field(default_factory=list)


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    correct = 0
    total = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = criterion(logits, targets)
            if training:
                loss.backward()
                optimizer.step()
            total_loss += float(loss.item()) * targets.size(0)
            correct += int((logits.argmax(dim=1) == targets).sum().item())
            total += targets.size(0)
    return total_loss / max(1, total), correct / max(1, total)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    learning_rate: float,
    device: torch.device,
) -> History:
    """Treina com CrossEntropyLoss e Adam, retendo o melhor estado."""
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history = History()
    best_state: dict[str, torch.Tensor] | None = None
    best_val_loss = float("inf")
    model.to(device)

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = _run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_acc = _run_epoch(model, val_loader, criterion, device)
        history.train_loss.append(train_loss)
        history.train_accuracy.append(train_acc)
        history.val_loss.append(val_loss)
        history.val_accuracy.append(val_acc)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
        print(
            f"Epoca {epoch:02d}/{epochs}: "
            f"treino loss={train_loss:.4f} acc={train_acc:.3f} | "
            f"validacao loss={val_loss:.4f} acc={val_acc:.3f}"
        )
    if best_state is not None:
        model.load_state_dict(best_state)
    return history


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int = 6,
) -> tuple[float, torch.Tensor]:
    """Retorna acurácia e matriz de confusão (linhas=reais, colunas=preditas)."""
    model.eval()
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    with torch.no_grad():
        for inputs, targets in loader:
            predictions = model(inputs.to(device)).argmax(dim=1).cpu()
            for target, prediction in zip(targets.view(-1), predictions.view(-1)):
                confusion[int(target), int(prediction)] += 1
    accuracy = float(confusion.diag().sum().item()) / max(1, int(confusion.sum().item()))
    return accuracy, confusion

