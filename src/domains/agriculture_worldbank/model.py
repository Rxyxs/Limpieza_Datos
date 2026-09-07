"""Modelado: predice el rendimiento de cereales (`cereal_yield_kg_ha`) del
año-país a partir de indicadores agrícolas contemporáneos + lags/rolling del
propio rendimiento y fertilizante pasados.

Split CRONOLÓGICO por año (nunca aleatorio -- barajar país-años mezclaría
información del futuro al pasado dentro de la misma serie de un país):
  - train: 1993-2015 (23 años x 9 países)
  - val:   2016-2019 (4 años x 9 países)
  - test:  2020-2025 (6 años x 9 países)

Comparación de 3 enfoques sobre el mismo split:
  1. baseline_media_pais: predice la media de train del PROPIO país (más
     fuerte y más honesto que la media global, porque el rendimiento de
     cereales varía estructuralmente por país -- Uruguay/Argentina vs.
     Bolivia no son comparables sin ese contexto).
  2. mlp_pytorch: red densa entrenada >=100 épocas con early stopping.
  3. xgboost: gradient boosting sobre las mismas features.

A diferencia del dominio financiero (retorno cambiario, ruido casi puro por
eficiencia de mercado), acá hay drivers reales y bien documentados en la
literatura agro-económica (fertilizante, irrigación, tierra arable), así que
un R² claramente positivo es un resultado esperado -- pero se reporta lo que
realmente se mida, no un número forzado.

Gotcha real ya encontrado y corregido en el dominio financiero de referencia:
con `nn.ReLU()` plano, una MLP chica entrenada full-batch sobre un dataset
tabular modesto puede colapsar por "dying ReLU" (todas las neuronas quedan
con pre-activación negativa y gradiente cero, la red converge a predecir una
constante ajena a la escala real del target -- R² tan malo como -8746 medido
empíricamente ahí). Se usa `nn.LeakyReLU(0.1)` acá desde el inicio por la
misma razón.
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

from src.domains.agriculture_worldbank.features import COUNTRY_COLUMNS, FEATURE_COLUMNS, TARGET_COLUMN
from src.toolkit.encoding import zscore_scale
from src.toolkit.model_zoo import build_sequences, fit_elasticnet, fit_lstm, fit_random_forest
from src.toolkit.torch_trainer import train_with_early_stopping

ROOT = Path(__file__).resolve().parents[3]
PROCESSED_DIR = ROOT / "data" / "processed" / "agriculture"
REPORTS_DIR = ROOT / "outputs" / "agriculture"

# Ventana de la LSTM: 5 años, por país. Es el único hiperparámetro
# de forma que no se elige por validación, porque cambia el dataset (y no
# solo el modelo) y haría incomparables los tamaños de train entre corridas.
LSTM_WINDOW = 5

TRAIN_END_YEAR = 2015
VAL_END_YEAR = 2019


class YieldMLP(nn.Module):
    """LeakyReLU, no ReLU: mismo colapso por "dying ReLU" documentado en
    `financial_bcch/model.py` puede aparecer acá con solo ~200 filas de
    entrenamiento y entrenamiento full-batch por cientos de épocas.
    LeakyReLU(0.1) deja un gradiente pequeño pero no nulo en la zona negativa,
    evitando que toda la capa oculta muera.
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


def chronological_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Sin `reset_index`: el índice original identifica la posición de cada fila
    # dentro del panel completo, que es lo que permite cortar las secuencias de
    # la LSTM con exactamente el mismo split que usan los otros cinco modelos.
    train_df = df[df["year"] <= TRAIN_END_YEAR]
    val_df = df[(df["year"] > TRAIN_END_YEAR) & (df["year"] <= VAL_END_YEAR)]
    test_df = df[df["year"] > VAL_END_YEAR]
    return train_df, val_df, test_df


def _country_of(df: pd.DataFrame) -> pd.Series:
    """Reconstruye el código de país desde las columnas one-hot (el panel de
    features ya no trae `country_iso3` como texto tras `encode_categorical_onehot`)."""
    return df[COUNTRY_COLUMNS].idxmax(axis=1).str.replace("country_iso3_", "", regex=False)


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

    # El target también se escala para entrenar la red (rendimiento en
    # kg/hectárea tiene una escala de miles, lo que ralentiza la convergencia
    # de una MLP inicializada cerca de 0) -- se revierte a unidades reales
    # (kg/hectárea) antes de calcular cualquier métrica, así que R²/RMSE/MAE
    # siguen siendo directamente interpretables.
    y_mean, y_std = float(y_train.mean()), float(y_train.std())
    y_train_scaled = (y_train - y_mean) / y_std
    y_val_scaled = (y_val - y_mean) / y_std

    results: dict = {}

    # 1. Baseline: media de train POR PAÍS (más honesto que la media global,
    # el rendimiento de cereales varía estructuralmente entre países).
    train_country = _country_of(train_df)
    test_country = _country_of(test_df)
    country_means = pd.Series(y_train, index=train_country).groupby(level=0).mean()
    global_train_mean = float(y_train.mean())
    baseline_pred = test_country.map(country_means).fillna(global_train_mean).to_numpy(dtype=float)
    results["baseline_media_pais"] = _metrics(y_test, baseline_pred)

    # 2. PyTorch MLP, >=100 épocas con early stopping y mejor checkpoint.
    torch.manual_seed(42)
    X_train_t = torch.tensor(X_train_scaled.values, dtype=torch.float32)
    y_train_t = torch.tensor(y_train_scaled, dtype=torch.float32).view(-1, 1)
    X_val_t = torch.tensor(X_val_scaled.values, dtype=torch.float32)
    y_val_t = torch.tensor(y_val_scaled, dtype=torch.float32).view(-1, 1)
    X_test_t = torch.tensor(X_test_scaled.values, dtype=torch.float32)

    model = YieldMLP(n_features=len(FEATURE_COLUMNS), hidden=32)
    train_result = train_with_early_stopping(
        model, X_train_t, y_train_t, X_val_t, y_val_t, loss_fn=nn.MSELoss(),
        max_epochs=400, min_epochs=100, patience=25, lr=1e-3, weight_decay=5e-3,
    )
    model.eval()
    with torch.no_grad():
        mlp_pred_scaled = model(X_test_t).numpy().ravel()
    mlp_pred = mlp_pred_scaled * y_std + y_mean
    results["mlp_pytorch"] = _metrics(y_test, mlp_pred)
    results["mlp_pytorch"]["epochs_run"] = train_result.epochs_run
    results["mlp_pytorch"]["best_epoch"] = train_result.best_epoch
    results["mlp_pytorch"]["early_stopped"] = train_result.early_stopped

    # 3. XGBoost, mismas features (sin escalar -- los árboles no lo necesitan),
    # target en unidades reales.
    xgb_model = xgb.XGBRegressor(
        n_estimators=500, max_depth=4, learning_rate=0.02, subsample=0.8, colsample_bytree=0.8,
        random_state=42, early_stopping_rounds=30, eval_metric="rmse",
    )
    xgb_model.fit(X_train_raw, y_train, eval_set=[(X_val_raw, y_val)], verbose=False)
    xgb_pred = xgb_model.predict(X_test_raw)
    results["xgboost"] = _metrics(y_test, xgb_pred)
    results["xgboost"]["best_iteration"] = int(xgb_model.best_iteration)

    # 4. Lineal regularizado (Ridge / ElasticNet / Lasso) sobre las features ya
    # escaladas -- la penalización compara magnitudes de coeficientes, así que
    # sin escalar elegiría por unidad de medida antes que por información real.
    enet = fit_elasticnet(X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled)
    results["elasticnet"] = {**_metrics(y_test, enet.predictions), **enet.metadata}

    # 5. Random Forest: exactamente las mismas features crudas que XGBoost, con
    # la estrategia opuesta (promediar árboles independientes y profundos en vez
    # de encadenar árboles débiles que corrigen al anterior).
    rf = fit_random_forest(
        X_train_raw, y_train, X_val_raw, y_val, X_test_raw, feature_names=list(FEATURE_COLUMNS),
    )
    results["random_forest"] = {**_metrics(y_test, rf.predictions), **rf.metadata}

    # 6. LSTM: lee las últimas LSTM_WINDOW observaciones como SECUENCIA, no como
    # columnas de lag aplanadas. Se entrena solo con las filas de historia
    # completa, pero predice sobre TODAS las filas de test, para que el conjunto
    # de evaluación siga siendo idéntico al de los otros cinco modelos.
    scaled_all = features_df.copy()
    for col in FEATURE_COLUMNS:
        mean, std = scale_stats[col]
        scaled_all[col] = (scaled_all[col] - mean) / std
    sequences, complete = build_sequences(
        scaled_all, list(FEATURE_COLUMNS), window=LSTM_WINDOW,
        time_column="year", group_column="country_name",
    )
    train_pos = train_df.index.to_numpy()
    val_pos = val_df.index.to_numpy()
    test_pos = test_df.index.to_numpy()
    lstm_fit = fit_lstm(
        sequences[train_pos][complete[train_pos]], y_train[complete[train_pos]],
        sequences[val_pos][complete[val_pos]], y_val[complete[val_pos]],
        sequences[test_pos],
    )
    lstm_pred = lstm_fit.predictions
    results["lstm"] = {**_metrics(y_test, lstm_pred), **lstm_fit.metadata}
    results["lstm"]["n_test_con_relleno"] = int((~complete[test_pos]).sum())

    return {
        "results": results,
        "train_losses": train_result.train_losses,
        "val_losses": train_result.val_losses,
        "best_epoch": train_result.best_epoch,
        "y_test": y_test.tolist(),
        "mlp_pred": mlp_pred.tolist(),
        "xgb_pred": xgb_pred.tolist(),
        "elasticnet_pred": enet.predictions.tolist(),
        "rf_pred": rf.predictions.tolist(),
        "lstm_pred": lstm_pred.tolist(),
        "baseline_pred": baseline_pred.tolist(),
        "n_train": len(train_df), "n_val": len(val_df), "n_test": len(test_df),
    }


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    features_df = pd.read_csv(PROCESSED_DIR / "agriculture_features.csv")
    output = train_all_models(features_df)

    metrics_path = REPORTS_DIR / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"train={output['n_train']} val={output['n_val']} test={output['n_test']}")
    for name, m in output["results"].items():
        extra = {k: v for k, v in m.items() if k not in {"r2", "rmse", "mae"}}
        print(f"  {name}: R2={m['r2']:.4f}  RMSE={m['rmse']:.2f}  MAE={m['mae']:.2f}  {extra}")
    print(f"metrics -> {metrics_path}")


if __name__ == "__main__":
    main()
