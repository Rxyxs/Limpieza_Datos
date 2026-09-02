"""Modelado: predice el retorno logarítmico del dólar al día siguiente.

Comparación honesta de 3 enfoques sobre el mismo split cronológico (nunca
aleatorio -- barajar filas de una serie de tiempo filtraría información del
futuro al pasado):
  1. baseline_media: predice siempre la media del retorno visto en train.
  2. mlp_pytorch: red densa entrenada >=100 épocas con early stopping.
  3. xgboost: gradient boosting sobre las mismas features.

Un R² bajo (incluso cercano a 0) en el retorno diario del dólar no es un bug:
es el resultado esperado si el mercado es razonablemente eficiente. El punto
de comparar contra el baseline es precisamente cuantificar cuánta señal real
hay más allá de "no sé nada, predigo el promedio".
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch import nn

from src.domains.financial_bcch.features import FEATURE_COLUMNS, TARGET_COLUMN
from src.toolkit.encoding import zscore_scale
from src.toolkit.torch_trainer import train_with_early_stopping

ROOT = Path(__file__).resolve().parents[3]
PROCESSED_DIR = ROOT / "data" / "processed" / "financial"
REPORTS_DIR = ROOT / "outputs" / "financial"


class ReturnMLP(nn.Module):
    """LeakyReLU, no ReLU: con solo 14 features de señal débil (retorno cambiario
    diario, cerca de ruido puro por eficiencia de mercado) y entrenamiento
    full-batch por cientos de épocas, una red con ReLU estándar colapsa por
    "dying ReLU" -- todas las neuronas quedan con pre-activación negativa y
    gradiente cero, y el modelo termina prediciendo una constante ajena a la
    escala real del target (encontrado empíricamente: R² por debajo de -80 con
    capas chicas). LeakyReLU(0.1) deja un gradiente pequeño pero no nulo en la
    zona negativa, evitando el colapso.
    """

    def __init__(self, n_features: int, hidden: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden), nn.LeakyReLU(0.1),
            nn.Linear(hidden, hidden // 2), nn.LeakyReLU(0.1),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x):
        return self.net(x)


def chronological_split(df: pd.DataFrame, train_frac: float = 0.70, val_frac: float = 0.15):
    n = len(df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    return df.iloc[:train_end], df.iloc[train_end:val_end], df.iloc[val_end:]


def _metrics(y_true, y_pred) -> dict:
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


def train_all_models(features_df: pd.DataFrame) -> dict:
    train_df, val_df, test_df = chronological_split(features_df)

    X_train_raw, X_val_raw, X_test_raw = train_df[FEATURE_COLUMNS], val_df[FEATURE_COLUMNS], test_df[FEATURE_COLUMNS]
    y_train, y_val, y_test = train_df[TARGET_COLUMN].values, val_df[TARGET_COLUMN].values, test_df[TARGET_COLUMN].values

    X_train_scaled, scale_stats = zscore_scale(X_train_raw.reset_index(drop=True), FEATURE_COLUMNS)
    X_val_scaled = X_val_raw.reset_index(drop=True).copy()
    X_test_scaled = X_test_raw.reset_index(drop=True).copy()
    for col in FEATURE_COLUMNS:
        mean, std = scale_stats[col]
        X_val_scaled[col] = (X_val_scaled[col] - mean) / std
        X_test_scaled[col] = (X_test_scaled[col] - mean) / std

    results: dict = {}

    # 1. Baseline: predice la media de train para toda fila de test.
    baseline_pred = np.full_like(y_test, fill_value=y_train.mean())
    results["baseline_media"] = _metrics(y_test, baseline_pred)

    # 2. PyTorch MLP, >=100 épocas con early stopping y mejor checkpoint.
    torch.manual_seed(42)
    X_train_t = torch.tensor(X_train_scaled.values, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_val_t = torch.tensor(X_val_scaled.values, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)
    X_test_t = torch.tensor(X_test_scaled.values, dtype=torch.float32)

    model = ReturnMLP(n_features=len(FEATURE_COLUMNS), hidden=16)
    train_result = train_with_early_stopping(
        model, X_train_t, y_train_t, X_val_t, y_val_t, loss_fn=nn.MSELoss(),
        max_epochs=400, min_epochs=100, patience=25, lr=1e-3, weight_decay=1e-2,
    )
    model.eval()
    with torch.no_grad():
        mlp_pred = model(X_test_t).numpy().ravel()
    results["mlp_pytorch"] = _metrics(y_test, mlp_pred)
    results["mlp_pytorch"]["epochs_run"] = train_result.epochs_run
    results["mlp_pytorch"]["best_epoch"] = train_result.best_epoch
    results["mlp_pytorch"]["early_stopped"] = train_result.early_stopped

    # 3. XGBoost, mismas features (sin escalar -- los árboles no lo necesitan).
    xgb_model = xgb.XGBRegressor(
        n_estimators=500, max_depth=4, learning_rate=0.02, subsample=0.8, colsample_bytree=0.8,
        random_state=42, early_stopping_rounds=30, eval_metric="rmse",
    )
    xgb_model.fit(X_train_raw, y_train, eval_set=[(X_val_raw, y_val)], verbose=False)
    xgb_pred = xgb_model.predict(X_test_raw)
    results["xgboost"] = _metrics(y_test, xgb_pred)
    results["xgboost"]["best_iteration"] = int(xgb_model.best_iteration)

    return {
        "results": results,
        "train_losses": train_result.train_losses,
        "val_losses": train_result.val_losses,
        "best_epoch": train_result.best_epoch,
        "y_test": y_test.tolist(),
        "mlp_pred": mlp_pred.tolist(),
        "xgb_pred": xgb_pred.tolist(),
        "baseline_pred": baseline_pred.tolist(),
        "n_train": len(train_df), "n_val": len(val_df), "n_test": len(test_df),
    }


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    features_df = pd.read_csv(PROCESSED_DIR / "financial_features.csv", parse_dates=["fecha"])
    output = train_all_models(features_df)

    metrics_path = REPORTS_DIR / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"train={output['n_train']} val={output['n_val']} test={output['n_test']}")
    for name, m in output["results"].items():
        extra = {k: v for k, v in m.items() if k not in {"r2", "rmse", "mae"}}
        print(f"  {name}: R2={m['r2']:.4f}  RMSE={m['rmse']:.5f}  MAE={m['mae']:.5f}  {extra}")
    print(f"metrics -> {metrics_path}")


if __name__ == "__main__":
    main()
