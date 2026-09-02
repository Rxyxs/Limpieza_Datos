"""Pruebas unitarias para src/toolkit/torch_trainer.py -- entrenamiento real
(no mockeado) sobre un problema sintético trivial, para verificar el
comportamiento del loop (piso de min_epochs, early stopping, mejor checkpoint)
sin depender de ningún dominio real."""
import torch
from torch import nn

from src.toolkit.torch_trainer import train_with_early_stopping


def _toy_data():
    torch.manual_seed(0)
    X = torch.randn(200, 3)
    true_w = torch.tensor([[2.0], [-1.0], [0.5]])
    y = X @ true_w + 0.1 * torch.randn(200, 1)
    return X[:150], y[:150], X[150:], y[150:]


def test_respects_min_epochs_floor():
    X_train, y_train, X_val, y_val = _toy_data()
    model = nn.Linear(3, 1)
    result = train_with_early_stopping(
        model, X_train, y_train, X_val, y_val, loss_fn=nn.MSELoss(),
        max_epochs=500, min_epochs=100, patience=3,
    )
    assert result.epochs_run >= 100


def test_converges_and_tracks_best_epoch_on_easy_linear_problem():
    X_train, y_train, X_val, y_val = _toy_data()
    model = nn.Linear(3, 1)
    result = train_with_early_stopping(
        model, X_train, y_train, X_val, y_val, loss_fn=nn.MSELoss(),
        max_epochs=500, min_epochs=100, patience=15, lr=0.05,
    )
    assert result.val_losses[-1] < result.val_losses[0]
    assert 1 <= result.best_epoch <= result.epochs_run
    assert len(result.train_losses) == result.epochs_run
