"""Limpieza del panel agrícola: 8 indicadores reales del Banco Mundial para 9
países sudamericanos (1990-2025) que hay que reshapear de formato largo
(un archivo por indicador) a un único panel ancho `(country, year) -> 8
columnas` antes de poder modelar.

La missingness real de este dominio tiene dos caras muy distintas:
  1. La mayoría de los indicadores están casi completos por país (34-36 de 36
     años reales) -- los huecos que sí existen son sobre todo el AÑO MÁS
     RECIENTE aún no reportado por la agencia estadística de cada país, un
     borde temporal (trailing gap), no un hueco interno.
  2. `AG.LND.IRIG.AG.ZS` (tierra agrícola irrigada, % del total) es la
     excepción real y seria: solo 50 de 324 celdas país-año tienen dato, y
     Perú no tiene NI UN SOLO valor real en los 36 años -- un país con la
     serie completa ausente, no un hueco que `interpolate_within_group`
     pueda resolver (no hay ningún punto de anclaje dentro del propio país).
     Esto no se inventa ni se maquilla: se documenta con los números reales
     abajo y se resuelve con imputación por media entre países (fallback
     explícito, no interpolación disfrazada de imputación).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.domains.agriculture_worldbank.fetch import INDICATORS, RAW_DIR
from src.toolkit.missing_data import impute_numeric_by_category, interpolate_within_group, missingness_report

ROOT = Path(__file__).resolve().parents[3]
PROCESSED_DIR = ROOT / "data" / "processed" / "agriculture"

INDICATOR_SLUGS = list(INDICATORS.values())


def _load_wide_panel() -> pd.DataFrame:
    """Carga los 8 CSV crudos (formato largo, uno por indicador) y los junta
    en un único panel ancho indexado por (country_iso3, year)."""
    frames = []
    for code, slug in INDICATORS.items():
        raw = pd.read_csv(RAW_DIR / f"{slug}.csv")
        raw = raw.rename(columns={"value": slug})
        frames.append(raw[["country_iso3", "country_name", "year", slug]])

    panel = frames[0]
    for frame in frames[1:]:
        panel = panel.merge(frame, on=["country_iso3", "country_name", "year"], how="outer")

    return panel.sort_values(["country_iso3", "year"]).reset_index(drop=True)


def build_clean_panel() -> tuple[pd.DataFrame, dict]:
    """Devuelve `(panel_limpio, reporte)` con el panel país-año limpio."""
    report: dict = {}
    panel = _load_wide_panel()

    before_missing = missingness_report(panel)
    report["missingness_antes"] = before_missing.set_index("columna")["pct_nulos"]
    print("Missingness ANTES de limpiar:")
    print(before_missing.to_string(index=False))

    # Paso 1: interpolar cada indicador DENTRO de cada país (nunca a través de
    # una frontera de país, eso mezclaría series nacionales sin relación real).
    # `limit_direction="both"` también resuelve los bordes: un año final aún no
    # reportado (ej. 2025) se rellena con el último valor real conocido, no se
    # inventa una tendencia -- una elección honesta, no una interpolación real.
    for slug in INDICATOR_SLUGS:
        panel = interpolate_within_group(panel, column=slug, group_column="country_iso3", sort_by="year")

    interpolated_missing = missingness_report(panel).set_index("columna")["pct_nulos"]
    report["missingness_tras_interpolar"] = interpolated_missing

    # Paso 2: lo que sigue nulo tras el paso 1 es, por construcción, un país
    # SIN NINGÚN valor real de ese indicador en toda la ventana -- interpolar
    # no puede resolverlo porque no hay ningún ancla dentro de la propia serie.
    # Se documenta el caso real antes de decidir: Perú e irrigación.
    still_missing = panel[INDICATOR_SLUGS].isna().sum()
    still_missing = still_missing[still_missing > 0]
    fallback_detail: dict[str, dict] = {}
    for slug, n_missing in still_missing.items():
        countries_fully_missing = (
            panel.loc[panel[slug].isna(), "country_iso3"].unique().tolist()
        )
        fallback_detail[slug] = {
            "celdas_sin_ancla_dentro_del_pais": int(n_missing),
            "paises_con_la_serie_completa_ausente": countries_fully_missing,
        }
        # Fallback: media condicional por país. Si un país no tiene NINGÚN
        # valor real (ej. Perú en irrigación), `impute_numeric_by_category`
        # cae a la media global entre los 9 países -- un valor honesto pero
        # deliberadamente burdo, documentado acá, no una interpolación
        # disfrazada.
        panel = impute_numeric_by_category(panel, value_column=slug, category_column="country_iso3")
    report["fallback_media_por_pais"] = fallback_detail

    after_missing = missingness_report(panel)
    report["missingness_despues"] = after_missing.set_index("columna")["pct_nulos"]
    print("\nMissingness DESPUÉS de limpiar:")
    print(after_missing.to_string(index=False))

    report["n_filas_finales"] = len(panel)
    report["n_paises"] = panel["country_iso3"].nunique()
    report["rango_anios"] = (int(panel["year"].min()), int(panel["year"].max()))

    return panel, report


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    panel, report = build_clean_panel()
    out_path = PROCESSED_DIR / "agriculture_panel_clean.csv"
    panel.to_csv(out_path, index=False)

    print(f"\nPanel limpio: {len(panel):,} filas, {report['n_paises']} países, años {report['rango_anios']}")
    print(f"  -> {out_path}")
    print("\nFallback de media entre países (celdas que ninguna interpolación interna podía resolver):")
    for slug, detail in report["fallback_media_por_pais"].items():
        print(f"  {slug}: {detail['celdas_sin_ancla_dentro_del_pais']} celdas -- "
              f"países con la serie completa ausente: {detail['paises_con_la_serie_completa_ausente']}")


if __name__ == "__main__":
    main()
