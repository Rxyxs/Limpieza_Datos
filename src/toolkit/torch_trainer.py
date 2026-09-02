"""Loop de entrenamiento genérico para los modelos PyTorch de los 4 dominios.

Un único trainer reusable (en vez de reescribir el loop en cada dominio) que:
entrena un mínimo de `min_epochs` épocas (piso explícito, no un valor que el
early stopping pueda saltarse antes de tener chance de converger), aplica
early stopping con paciencia sobre la pérdida de validación, y se queda con
los pesos de la mejor época (no los de la última) -- la última época no es
necesariamente la mejor una vez que el modelo empieza a sobreajustar.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn


@dataclass
class TrainResult:
    model: nn.Module
    train_losses: list[float] = field(default_factory=list)
    val_losses: list[float] = field(default_factory=list)
    best_epoch: int = 0
    epochs_run: int = 0
    early_stopped: bool = False


def train_with_early_stopping(
    model: nn.Module,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    loss_fn: nn.Module,
    max_epochs: int = 300,
    min_epochs: int = 100,
    patience: int = 20,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    verbose_every: int = 0,
) -> TrainResult:
    """Entrena `model` con Adam, guardando el mejor estado según `val_loss`.

    `min_epochs=100` por defecto: el proyecto exige al menos 100 épocas de
    entrenamiento por modelo, así que el early stopping solo puede activarse
    *después* de esa marca, nunca antes.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    result = TrainResult(model=model)

    best_val = float("inf")
    best_state = None
    bad_epochs = 0

    for epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad()
        train_pred = model(X_train)
        train_loss = loss_fn(train_pred, y_train)
        train_loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            val_loss = loss_fn(val_pred, y_val)

        result.train_losses.append(float(train_loss.item()))
        result.val_losses.append(float(val_loss.item()))

        if val_loss.item() < best_val:
            best_val = val_loss.item()
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            result.best_epoch = epoch + 1
            bad_epochs = 0
        else:
            bad_epochs += 1

        if verbose_every and (epoch + 1) % verbose_every == 0:
            print(f"  epoch {epoch + 1:4d}/{max_epochs}  train={train_loss.item():.5f}  val={val_loss.item():.5f}")

        if epoch + 1 >= min_epochs and bad_epochs >= patience:
            result.early_stopped = True
            break

    result.epochs_run = len(result.train_losses)
    if best_state is not None:
        model.load_state_dict(best_state)
    return result
