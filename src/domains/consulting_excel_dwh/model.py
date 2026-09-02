"""Modelado: predice la esperanza de vida al nacer del año siguiente, a partir
de indicadores socioeconómicos reales del año actual (gasto en salud, acceso a
agua/saneamiento/electricidad, PIB per cápita, mortalidad infantil, Gini,
desempleo, crecimiento poblacional) + sus rezagos de 1 año, para 163 países
reales, 1961-2024.

Split cronológico por año (nunca aleatorio): entrenar con años tempranos y
testear en los más recientes es la única forma honesta de validar un panel de
series de tiempo -- barajar filas mezclaría información del futuro al pasado
a través de países.
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

from src.domains.consulting_excel_dwh.features import FEATURE_COLUMNS, TARGET_COLUMN
from src.toolkit.encoding import inverse_zscore, zscore_scale
from src.toolkit.torch_trainer import train_with_early_stopping

ROOT = Path(__file__).resolve().parents[3]
PROCESSED_DIR = ROOT / "data" / "processed" / "consulting"
REPORTS_DIR = ROOT / "outputs" / "consulting"


class LifeExpectancyMLP(nn.Module):
    """LeakyReLU, no ReLU: la misma corrección aplicada en `financial_bcch/model.py`
    -- con pocas features y entrenamiento full-batch por cientos de épocas, ReLU
    estándar puede colapsar por "dying ReLU" (gradiente cero en toda la red,
    predicción constante ajena a la escala real del target). Acá el target
    (años de esperanza de vida) SÍ tiene señal fuerte y real en las features
    (gasto en salud, agua potable, PIB), así que se espera un R² alto -- pero
    la corrección se aplica desde el inicio, no reactivamente.
    """

    def __init__(self, n_features: int, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden), nn.LeakyReLU(0.1),
            nn.Linear(hidden, hidden // 2), nn.LeakyReLU(0.1),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x):
        return self.net(x)


def chronological_split(df: pd.DataFrame, train_end_year: int = 2010, val_end_year: int = 2018):
    train_df = df[df["anio"] <= train_end_year]
    val_df = df[(df["anio"] > train_end_year) & (df["anio"] <= val_end_year)]
    test_df = df[df["anio"] > val_end_year]
    return train_df, val_df, test_df


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

    baseline_pred = np.full_like(y_test, fill_value=y_train.mean())
    results["baseline_media"] = _metrics(y_test, baseline_pred)

    # El target (años de esperanza de vida, media ~70) también se escala --
    # sin esto, la pérdida MSE arranca en ~5000 (70^2) y la red necesita
    # cientos de épocas solo para que el sesgo de la última capa converja
    # cerca de la escala real, produciendo predicciones desde 7.7 hasta 92.8
    # años (encontrado empíricamente) mientras el rango real es 18.8-84.6.
    # Estandarizar X e Y es la práctica estándar para redes neuronales, no
    # solo X.
    y_mean, y_std = y_train.mean(), y_train.std()
    y_train_scaled = (y_train - y_mean) / y_std
    y_val_scaled = (y_val - y_mean) / y_std

    torch.manual_seed(42)
    X_train_t = torch.tensor(X_train_scaled.values, dtype=torch.float32)
    y_train_t = torch.tensor(y_train_scaled, dtype=torch.float32).view(-1, 1)
    X_val_t = torch.tensor(X_val_scaled.values, dtype=torch.float32)
    y_val_t = torch.tensor(y_val_scaled, dtype=torch.float32).view(-1, 1)
    X_test_t = torch.tensor(X_test_scaled.values, dtype=torch.float32)

    model = LifeExpectancyMLP(n_features=len(FEATURE_COLUMNS), hidden=32)
    train_result = train_with_early_stopping(
        model, X_train_t, y_train_t, X_val_t, y_val_t, loss_fn=nn.MSELoss(),
        max_epochs=400, min_epochs=100, patience=25, lr=1e-3, weight_decay=1e-3,
    )
    model.eval()
    with torch.no_grad():
        mlp_pred_scaled = model(X_test_t).numpy().ravel()
    mlp_pred = inverse_zscore(mlp_pred_scaled, y_mean, y_std)
    results["mlp_pytorch"] = _metrics(y_test, mlp_pred)
    results["mlp_pytorch"]["epochs_run"] = train_result.epochs_run
    results["mlp_pytorch"]["best_epoch"] = train_result.best_epoch
    results["mlp_pytorch"]["early_stopped"] = train_result.early_stopped

    xgb_model = xgb.XGBRegressor(
        n_estimators=500, max_depth=5, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8,
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
    features_df = pd.read_csv(PROCESSED_DIR / "consulting_features.csv")
    output = train_all_models(features_df)

    metrics_path = REPORTS_DIR / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"train={output['n_train']} val={output['n_val']} test={output['n_test']}")
    for name, m in output["results"].items():
        extra = {k: v for k, v in m.items() if k not in {"r2", "rmse", "mae"}}
        print(f"  {name}: R2={m['r2']:.4f}  RMSE={m['rmse']:.4f}  MAE={m['mae']:.4f}  {extra}")
    print(f"metrics -> {metrics_path}")


if __name__ == "__main__":
    main()
