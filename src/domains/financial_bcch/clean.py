"""Limpieza del panel financiero: 5 indicadores reales de mindicador.cl con
frecuencias distintas (dólar/UF/TPM publican diario en días hábiles, IPC/IMACEC
publican mensual) que hay que alinear a una única grilla temporal antes de
poder modelar. Además de filas que no existen (fin de semana, feriado, mes sin
publicar todavía), se encontró un valor de nivel realmente corrupto en la
propia API: la UF del 2014-12-29 llega como 608.15 en vez de ~24.626 (un
salto de +3.954% seguido de -97,5% al día siguiente) -- un error de captura
real en la fuente pública, detectado y corregido acá vía
`fix_implausible_level_jumps`, no descartado ni ignorado.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.toolkit.datetime_cleaning import reindex_to_full_calendar
from src.toolkit.duplicates import exact_duplicate_report
from src.toolkit.missing_data import missingness_report
from src.toolkit.outliers import fix_implausible_level_jumps, winsorize_column

ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = ROOT / "data" / "raw" / "financial"
PROCESSED_DIR = ROOT / "data" / "processed" / "financial"

DAILY_INDICATORS = ["dolar", "uf", "tpm"]
MONTHLY_INDICATORS = ["ipc", "imacec"]


def _load_raw(codigo: str) -> pd.DataFrame:
    df = pd.read_csv(RAW_DIR / f"{codigo}.csv", parse_dates=["fecha"])
    df["fecha"] = df["fecha"].dt.tz_localize(None).dt.normalize()
    return df[["fecha", "valor"]].rename(columns={"valor": codigo})


def build_clean_panel() -> tuple[pd.DataFrame, dict]:
    """Devuelve `(panel_limpio, reporte)` con el panel diario alineado y limpio."""
    report: dict = {}

    daily_frames = []
    for codigo in DAILY_INDICATORS:
        raw = _load_raw(codigo)
        raw, n_dupes = exact_duplicate_report(raw, subset=["fecha"])
        filled = reindex_to_full_calendar(raw, date_column="fecha", freq="D", fill_method="ffill")
        report[f"{codigo}_dupes_exactas"] = n_dupes
        report[f"{codigo}_filas_creadas_por_calendario"] = len(filled) - len(raw)

        # dólar y UF tienen tendencia real -> un salto/reversión implausible en
        # el nivel es un error de captura, no volatilidad (TPM se deja fuera:
        # es una tasa de política, un salto real de varios puntos es un evento
        # de política monetaria genuino, no un error de fuente).
        if codigo in {"dolar", "uf"}:
            filled, n_level_errors = fix_implausible_level_jumps(filled, codigo, sort_by="fecha", threshold=0.5)
            report[f"{codigo}_errores_de_nivel_corregidos"] = n_level_errors

        daily_frames.append(filled)

    panel = daily_frames[0]
    for frame in daily_frames[1:]:
        panel = panel.merge(frame, on="fecha", how="outer")

    # IPC/IMACEC publican una vez al mes; se asignan a la grilla diaria vía
    # forward-fill (el dato "vigente" de un mes rige hasta la próxima
    # publicación) -- alineación de grano mensual->diario, la técnica estándar
    # para combinar indicadores macro de baja frecuencia con series diarias.
    for codigo in MONTHLY_INDICATORS:
        raw = _load_raw(codigo)
        panel = panel.merge(raw, on="fecha", how="left")
        panel[codigo] = panel[codigo].ffill()

    panel = panel.dropna().sort_values("fecha").reset_index(drop=True)

    before_missing = missingness_report(panel).set_index("columna")["pct_nulos"]

    # El nivel del dólar tiene tendencia real (no es ruido) -- winsorizar el
    # NIVEL recortaría movimientos genuinos del mercado. Se calcula el retorno
    # logarítmico diario primero y se winsoriza ESE, que es donde un valor
    # extremo sí es estadísticamente atípico frente al resto de los retornos.
    price_ratio = panel["dolar"] / panel["dolar"].shift(1)
    panel["dolar_log_return"] = np.where(price_ratio > 0, np.log(price_ratio), np.nan)
    panel = panel.dropna(subset=["dolar_log_return"]).reset_index(drop=True)
    panel, n_return_outliers = winsorize_column(panel, "dolar_log_return", k=4.0)
    report["dolar_log_return_outliers_winsorizados"] = n_return_outliers

    after_missing = missingness_report(panel).set_index("columna")["pct_nulos"]
    report["missingness_antes"] = before_missing
    report["missingness_despues"] = after_missing
    report["n_filas_finales"] = len(panel)

    return panel, report


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    panel, report = build_clean_panel()
    out_path = PROCESSED_DIR / "financial_panel_clean.csv"
    panel.to_csv(out_path, index=False)

    print(f"Panel limpio: {len(panel):,} filas ({panel['fecha'].min().date()} a {panel['fecha'].max().date()})")
    print(f"  -> {out_path}")
    for key, value in report.items():
        if isinstance(value, pd.Series):
            continue
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
