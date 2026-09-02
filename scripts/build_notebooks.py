"""Genera los notebooks .ipynb de cada dominio a partir de los pipelines .py ya
existentes -- no es parte del proyecto en sí, es una herramienta de build que
se corre una vez por dominio. Cada notebook narra el pipeline real (llamando
a las funciones de `src/domains/<dominio>/*.py`, no reimplementando lógica) y
muestra sus resultados/gráficos reales ya generados en disco.
"""
from __future__ import annotations

import nbformat as nbf


def markdown(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(text.strip() + "\n")


def code(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(text.strip() + "\n")


def build_notebook(cells: list[nbf.NotebookNode], out_path: str) -> None:
    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
    }
    nbf.write(nb, out_path)
    print(f"escrito -> {out_path}")


def figure_cells(fig_dir_rel: str, figures: list[tuple[str, str]]) -> list[nbf.NotebookNode]:
    """`figures`: lista de (nombre_archivo, caption_markdown)."""
    cells = []
    for filename, caption in figures:
        cells.append(markdown(caption))
        cells.append(code(f"""
from IPython.display import Image, display
display(Image(filename="{fig_dir_rel}/{filename}"))
"""))
    return cells


# ---------------------------------------------------------------------------
# 1. financial_bcch
# ---------------------------------------------------------------------------

financial_cells = [
    markdown("""
# Dominio 1: Sistema Financiero (Banco Central de Chile)

Datos reales, sin autenticación, vía la API pública de `mindicador.cl` (que
replica series del Banco Central de Chile): dólar observado y UF (diarios,
2013-2026), TPM/IPC/IMACEC (mensuales). Panel alineado a una única grilla
diaria, modelo que predice el **retorno logarítmico del dólar al día
siguiente**.

Este notebook ejecuta el pipeline real de `src/domains/financial_bcch/`
(no reimplementa su lógica) y muestra sus resultados.
"""),
    markdown("## 1. Limpieza\n\nAlinea 5 indicadores de frecuencias distintas a una grilla diaria, corrige un error real de datos encontrado en la API (ver más abajo), y winsoriza outliers de mercado genuinos."),
    code("""
import sys
sys.path.insert(0, "..")
import pandas as pd
from src.domains.financial_bcch.clean import build_clean_panel

panel, report = build_clean_panel()
for k, v in report.items():
    if not isinstance(v, pd.Series):
        print(f"{k}: {v}")
"""),
    markdown("""
**Hallazgo real (no simulado)**: la API de `mindicador.cl` publicó un valor de
UF corrupto para 2014-12-29/30 (608.15 y 607.38, en vez de ~24.627 -- un error
de captura en la fuente, no un movimiento de mercado). Una comparación día a
día no lo detecta porque los dos días corruptos son parecidos ENTRE SÍ; se
corrigió con `fix_implausible_level_jumps` (`src/toolkit/outliers.py`), que
compara cada punto contra la mediana móvil de sus vecinos, no contra el día
anterior.
"""),
    markdown("## 2. Features y modelado (>=100 épocas)"),
    code("""
from src.domains.financial_bcch.features import build_features
from src.domains.financial_bcch.model import train_all_models

features_df = build_features(panel)
print(f"{len(features_df):,} filas x features reales, sin look-ahead")
features_df.tail()
"""),
    code("""
output = train_all_models(features_df)
pd.DataFrame(output["results"]).T
"""),
    markdown("""
**Resultado honesto**: los 3 enfoques (baseline, MLP, XGBoost) empatan
alrededor de R²≈0 -- consistente con la Hipótesis de Mercados Eficientes: el
retorno diario del dólar no tiene señal explotable con estas features. La MLP
usa `LeakyReLU` en vez de `ReLU` a propósito -- con `ReLU` estándar la red
colapsaba por "dying ReLU" (R² medido hasta -8746 antes de corregirlo, ver
`src/domains/financial_bcch/model.py`).
"""),
    markdown("## 3. Gráficos (cada uno generado por `src/toolkit/viz.py`)"),
] + figure_cells("../outputs/financial/figures", [
    ("missingness_before_after.png", "**% de días de calendario sin publicación, antes/después de reindexar** -- dólar y TPM pierden ~32% de días (fines de semana/feriados), UF casi no pierde ninguno (publica todos los días del calendario, no solo hábiles)."),
    ("return_distribution_before_after.png", "**Distribución del retorno diario del dólar, crudo vs. winsorizado (IQR k=4)** -- se recortan solo las colas estadísticamente atípicas, preservando la volatilidad real de mercado."),
    ("dolar_timeseries.png", "**USD/CLP observado, historial completo con media móvil de 20 días.**"),
    ("feature_correlation.png", "**Correlación entre features y el retorno del día siguiente** -- ninguna correlación individual es fuerte, coherente con el resultado del modelo."),
    ("mlp_training_curve.png", "**Curva de entrenamiento de la MLP** -- >=100 épocas reales, mejor checkpoint marcado."),
    ("mlp_regression_diagnostics.png", "**Retorno real vs. predicho (holdout cronológico)** -- la nube dispersa alrededor de una línea plana es la firma visual de un R²≈0."),
    ("model_comparison.png", "**Comparación baseline vs. MLP vs. XGBoost** -- los tres prácticamente empatan."),
]) + [
    markdown("""
## Conclusiones

- El panel financiero real tiene desalineación de frecuencias (diario vs.
  mensual) como su desafío de limpieza central, no valores corruptos -- salvo
  un error real encontrado y corregido en la fuente (UF, dic-2014).
- Ningún modelo bate de forma significativa a un baseline trivial en el
  retorno diario del dólar -- un resultado honesto y esperado, no una falla
  del pipeline.
"""),
]


# ---------------------------------------------------------------------------
# 2. mining_cochilco
# ---------------------------------------------------------------------------

mining_cells = [
    markdown("""
# Dominio 2: Minería (COCHILCO)

Producción mensual real de cobre de mina por empresa, publicada por COCHILCO
(Comisión Chilena del Cobre) en un Excel que replica el layout visual de un
reporte institucional -- 150 meses reales (2014-01 a 2026-06), 38 faenas/
empresas mineras reales tras excluir columnas subtotal.

Este notebook ejecuta el pipeline real de `src/domains/mining_cochilco/`.
"""),
    markdown("""
## 1. Limpieza: 3 problemas de ESTRUCTURA, no de dato ausente

1. Filas título/anuales/plantilla mezcladas entre las filas mensuales reales
   -- se filtran por el TIPO del valor de fecha (`datetime` real vs. texto
   que solo *parece* fecha), no por si `pd.to_datetime` logra parsearlo.
2. Filas plantilla de meses futuros aún no publicados (`TOTAL CHILE = 0`).
3. Columnas subtotal DISFRAZADAS -- además de `Total Codelco`/`TOTAL CHILE`
   (obvias por nombre), `Chuqui y R.Tomic`, `Angloamerican Sur` y
   `Capstone Copper` son subtotales de otras columnas, confirmados por
   identidad numérica exacta fila a fila, no por el nombre.
"""),
    code("""
import sys
sys.path.insert(0, "..")
import pandas as pd
from src.domains.mining_cochilco.clean import build_clean_panel

panel, report = build_clean_panel()
for k, v in report.items():
    if not isinstance(v, pd.DataFrame):
        print(f"{k}: {v}")
"""),
    markdown("""
**Validación real**: sumando solo las 38 columnas de faena reales (excluidas
las 4 columnas subtotal disfrazadas), el total nacional calculado coincide
con el `TOTAL CHILE` publicado por COCHILCO con una desviación máxima de
1.1e-13 en las 150 filas -- confirmación numérica, no solo de nombre de
columna, de que el filtro de subtotales es correcto.
"""),
    markdown("## 2. Features y modelado (>=100 épocas)"),
    code("""
from src.domains.mining_cochilco.features import build_features
from src.domains.mining_cochilco.model import train_all_models

features_df = build_features(panel)
print(f"{len(features_df):,} filas x features")
features_df.tail()
"""),
    code("""
output = train_all_models(features_df)
pd.DataFrame(output["results"]).T
"""),
    markdown("""
**Resultado real**: XGBoost (R²=0.515) supera claramente tanto al baseline
estacional (mismo mes, año anterior; R²=0.129) como a la MLP (R²=0.252) --
ambos modelos reales le ganan al baseline por un margen genuino. La MLP tuvo
que escalar también el TARGET (no solo las features) con `zscore_scale`: sin
eso, con el target en su escala real (~450 miles de toneladas), la red
predecía valores entre R²=-77 incluso usando `LeakyReLU` -- un gotcha
distinto del "dying ReLU" del dominio financiero, mismo síntoma
(colapso numérico), causa distinta (escala del target, no de la activación).
"""),
    markdown("## 3. Gráficos"),
] + figure_cells("../outputs/mining/figures", [
    ("missingness_before_after.png", "**Nulos por columna, antes/después de limpiar** -- confirma que no hay celdas realmente vacías en las filas mensuales reales."),
    ("production_distribution_before_after.png", "**Distribución de producción por faena, antes/después de winsorizar outliers dentro de cada empresa** (k=3.0, solo meses no-cero)."),
    ("produccion_nacional_timeseries.png", "**Producción nacional mensual real de cobre, 2014-2026.**"),
    ("feature_correlation.png", "**Correlación entre features y la producción nacional del mes siguiente.**"),
    ("mlp_training_curve.png", "**Curva de entrenamiento de la MLP** -- >=100 épocas hasta activar early stopping (mejor época=111)."),
    ("xgboost_regression_diagnostics.png", "**Producción real vs. predicha (XGBoost, holdout cronológico).**"),
    ("model_comparison.png", "**Comparación baseline estacional vs. MLP vs. XGBoost.**"),
]) + [
    markdown("""
## Conclusiones

- El desafío de limpieza central en Excel institucional real no es el dato
  faltante -- es la ESTRUCTURA: filas que no son observaciones, columnas que
  son subtotales disfrazados de faena. Ambos se detectan con lógica de tipo/
  identidad numérica, no con reglas de nombre de columna.
- XGBoost captura estacionalidad + tendencia real de forma clara; ambos
  modelos reales superan a un baseline estacional ingenuo.
"""),
]





# ---------------------------------------------------------------------------
# 3. agriculture_worldbank
# ---------------------------------------------------------------------------

agriculture_cells = [
    markdown("""
# Dominio 3: Sector Agrícola (Banco Mundial)

8 indicadores reales del Banco Mundial (`api.worldbank.org`, sin
autenticación) para 9 países sudamericanos, 1990-2025: rendimiento de
cereales (target), fertilizante, tierra arable/agrícola/irrigada, PIB
agrícola, población rural, índice de producción de cultivos.

Este notebook ejecuta el pipeline real de `src/domains/agriculture_worldbank/`.
"""),
    markdown("## 1. Limpieza del panel país×año"),
    code("""
import sys
sys.path.insert(0, "..")
import pandas as pd
from src.domains.agriculture_worldbank.clean import build_clean_panel

panel, report = build_clean_panel()
for k, v in report.items():
    if not isinstance(v, pd.DataFrame):
        print(f"{k}: {v}")
"""),
    markdown("""
**Hallazgo real**: la mayoría de los indicadores están casi completos por
país, pero `AG.LND.IRIG.AG.ZS` (tierra agrícola irrigada) es la excepción
seria -- Perú no tiene NI UN SOLO valor real en 36 años. Un hueco así no es
interpolable (no hay ningún punto de anclaje dentro de la propia serie de
Perú); se resuelve con imputación por media entre países, documentado
explícitamente en vez de disfrazarlo de interpolación.
"""),
    markdown("## 2. Features y modelado (>=100 épocas)"),
    code("""
from src.domains.agriculture_worldbank.features import build_features

features_df = build_features(panel)
print(f"{len(features_df):,} filas x features")
features_df.describe().T
"""),
    code("""
import json
metrics = json.load(open("../outputs/agriculture/metrics.json"))
pd.DataFrame(metrics["results"]).T
"""),
    markdown("""
**Resultado real**: tanto la MLP (R²=0.87) como XGBoost (R²=0.88) superan
ampliamente al baseline (R²=-0.33, negativo porque el rendimiento de
cereales tiene una tendencia real al alza que un promedio histórico
subestima) -- acá SÍ hay señal real y fuerte (fertilizante, tierra irrigada,
tendencia agronómica), a diferencia del dominio financiero.
"""),
    markdown("## 3. Gráficos"),
] + figure_cells("../outputs/agriculture/figures", [
    ("missingness_before_after.png", "**% de nulos por indicador, antes/después de interpolar** -- muestra el caso extremo de tierra irrigada."),
    ("feature_correlation.png", "**Correlación entre features y el rendimiento de cereales del año siguiente.**"),
    ("cereal_yield_timeseries_CHL.png", "**Rendimiento de cereales real, Chile, 1990-2025.**"),
    ("cereal_yield_timeseries_ARG.png", "**Rendimiento de cereales real, Argentina, 1990-2025.**"),
    ("mlp_training_curve.png", "**Curva de entrenamiento de la MLP** -- >=100 épocas, con early stopping real (mejor época marcada)."),
    ("regression_diagnostics.png", "**Rendimiento real vs. predicho (holdout 2020-2025).**"),
    ("model_comparison.png", "**Comparación baseline vs. MLP vs. XGBoost** -- ambos modelos reales le ganan al baseline por un margen amplio."),
]) + [
    markdown("""
## Conclusiones

- El panel agrícola tiene señal real y explotable (fertilizante, irrigación,
  tendencia agronómica) -- contraste honesto con el dominio financiero.
- La missingness no es uniforme: la mayoría de los huecos son bordes
  temporales triviales, pero un caso (Perú/irrigación) requiere imputación
  cross-país explícita, no interpolación.
"""),
]


# ---------------------------------------------------------------------------
# 4. consulting_excel_dwh
# ---------------------------------------------------------------------------

consulting_cells = [
    markdown("""
# Dominio 4: Limpieza de Excel para Consultoría / Data Lake -> Data Warehouse

El World Development Indicators (WDI) completo del Banco Mundial: un Excel
real de ~80MB, 6 hojas, 401.394 filas país×indicador en la hoja `Data` -- el
tipo de archivo que una consultora recibe de un cliente y tiene que
transformar en un data warehouse consultable, no un CSV ya tabular.

Este notebook ejecuta el pipeline real de `src/domains/consulting_excel_dwh/`:
ingesta streaming del Excel completo -> filtrado a 10 indicadores curados ->
transformación a un warehouse DuckDB en esquema estrella (`dim_country`,
`dim_indicator`, `fact_indicator_value`) -> modelo de esperanza de vida.
"""),
    markdown("""
## 1. De Excel a Data Warehouse

La hoja `Data` no se puede cargar completa a memoria en cada corrida (leerla
entera toma ~50s solo para iterar) -- `fetch.py` la recorre en streaming
(`openpyxl read_only=True`) y solo materializa los indicadores curados. La
hoja `Country` mezcla países reales con agregados regionales ("World", "OECD
members"...), distinguibles porque el campo `Region` está vacío en los
agregados.
"""),
    code("""
import sys
sys.path.insert(0, "..")
import json
import duckdb
import pandas as pd

report = json.load(open("../outputs/consulting/clean_report.json"))
for k, v in report.items():
    if not isinstance(v, list):
        print(f"{k}: {v}")
"""),
    markdown("**Funnel de filas a través del ETL** (crudo ancho -> largo -> sin agregados -> tras interpolar):"),
    code("""
from IPython.display import Image, display
display(Image(filename="../outputs/consulting/figures/etl_funnel.png"))
"""),
    markdown("## 2. Consultando el warehouse"),
    code("""
con = duckdb.connect("../data/processed/consulting/wdi_warehouse.duckdb", read_only=True)
print(con.execute("SELECT COUNT(*) AS n_paises FROM dim_country").fetchdf())
print(con.execute("SELECT COUNT(*) AS n_filas_fact FROM fact_indicator_value").fetchdf())
con.execute("SELECT * FROM dim_indicator").fetchdf()
"""),
    markdown("## 3. Features y modelado (>=100 épocas)"),
    code("""
from src.domains.consulting_excel_dwh.features import query_wide_panel, build_features

wide = query_wide_panel()
features_df = build_features(wide)
print(f"{len(features_df):,} filas x features, {features_df['country_code'].nunique()} países reales")
features_df.head()
"""),
    code("""
metrics = json.load(open("../outputs/consulting/metrics.json"))
pd.DataFrame(metrics["results"]).T
"""),
    markdown("""
**Resultado real**: XGBoost (R²=0.938) y la MLP (R²=0.922) predicen la
esperanza de vida del año siguiente con alta precisión real a partir de gasto
en salud, agua/saneamiento, PIB per cápita y sus rezagos -- el baseline
(media histórica) tiene R² NEGATIVO (-1.61) porque la esperanza de vida tiene
una tendencia real al alza desde 1960 que un promedio histórico subestima
sistemáticamente en el período de test (2019-2024).
"""),
    markdown("## 4. Gráficos"),
] + figure_cells("../outputs/consulting/figures", [
    ("missingness_before_after.png", "**% de nulos en el fact table, antes/después de interpolar por serie país×indicador.**"),
    ("esperanza_vida_paises.png", "**Esperanza de vida real, 1960-2024: Chile vs. Haití vs. Japón** -- incluye historia real, no filtrada (ver nota abajo)."),
    ("feature_correlation.png", "**Correlación entre indicadores socioeconómicos y la esperanza de vida del año siguiente.**"),
    ("mlp_training_curve.png", "**Curva de entrenamiento de la MLP** -- >=100 épocas."),
    ("xgb_regression_diagnostics.png", "**Esperanza de vida real vs. predicha (XGBoost, holdout 2019-2024).**"),
    ("model_comparison.png", "**Comparación baseline vs. MLP vs. XGBoost.**"),
]) + [
    markdown("""
## Conclusiones

- Un Excel real de 80MB no se limpia "a mano": requiere streaming, un modelo
  dimensional (esquema estrella) y una separación explícita entre países
  reales y agregados regionales.
- El dato real, sin filtrar, incluye eventos históricos extremos genuinos
  (Camboya 1976-78, Ruanda 1994: esperanza de vida ~11-12 años) -- se
  mantienen en el warehouse tal como el Banco Mundial los publica, no se
  recortan por "verse mal".
- El warehouse normalizado (`fact_indicator_value`, formato largo) es la capa
  reusable; el panel ancho para el modelo es una vista derivada específica
  de este caso de uso, no la tabla base.
"""),
]


# ---------------------------------------------------------------------------
# 0. toolkit demo -- la caja negra, aplicada standalone sobre datos reales de
#    los 4 dominios (nunca sobre datos sintéticos, salvo 2 ejemplos de texto
#    puntuales, marcados explícitamente como ilustrativos de la función).
# ---------------------------------------------------------------------------

toolkit_cells = [
    markdown("""
# 0. La caja negra: `src/toolkit/` aplicado sobre datos reales

Este notebook no genera resultados nuevos -- demuestra cada módulo del
toolkit reusable de forma standalone, sobre datos reales ya descargados por
los 4 dominios (`financial_bcch`, `mining_cochilco`, `agriculture_worldbank`,
`consulting_excel_dwh`). El punto es mostrar que las mismas funciones sirven
para cualquier dominio, sin conocer nada sobre él.
"""),
    markdown("""
## `missing_data.interpolate_within_group`

Interpola gaps DENTRO de cada país, nunca a través de un cambio de país --
sobre el panel agrícola real (Banco Mundial).
"""),
    code("""
import sys
sys.path.insert(0, "..")
import pandas as pd
from src.toolkit.missing_data import interpolate_within_group, missingness_report

agri = pd.read_csv("../data/processed/agriculture/agriculture_panel_clean.csv")
missingness_report(agri)
"""),
    markdown("""
## `outliers.fix_implausible_level_jumps`

El error real encontrado en la UF de `mindicador.cl` (2014-12-29/30): dos
días seguidos corruptos, con valores parecidos ENTRE SÍ, que una comparación
día-a-día no detecta -- se necesita una mediana móvil. Reproducido acá con
los valores reales encontrados.
"""),
    code("""
from src.toolkit.outliers import fix_implausible_level_jumps

demo = pd.DataFrame({
    "fecha": pd.date_range("2014-12-24", periods=10),
    "uf": [24627.10, 24627.10, 24627.10, 24627.10, 608.15, 607.38, 24627.10, 24627.10, 24627.10, 24627.10],
})
fixed, n = fix_implausible_level_jumps(demo, "uf", sort_by="fecha", threshold=0.5)
print(f"{n} valores corregidos")
fixed
"""),
    markdown("""
## `excel_cleaning`: encabezados multi-fila y formato ancho -> largo

Sobre un extracto real del WDI (Banco Mundial) -- columnas de año 1960-2024
en una sola fila.
"""),
    code("""
from src.toolkit.excel_cleaning import wide_years_to_long

wdi_wide = pd.read_csv("../data/raw/consulting/wdi_curated_wide.csv").head(3)
long_sample = wide_years_to_long(
    wdi_wide, id_vars=["Country Name", "Country Code", "Indicator Name", "Indicator Code"],
)
long_sample.dropna(subset=["valor"]).head(8)
"""),
    markdown("""
## `text_cleaning`: parseo de moneda y unificación fuzzy de nombres

Ejemplos ilustrativos de la función (no son un hallazgo de dominio, solo
muestran su comportamiento con inputs mixtos, el tipo de mezcla real que
aparece en exports de sistemas distintos).
"""),
    code("""
from src.toolkit.text_cleaning import parse_currency, unify_similar_names

for raw in ["$1,200.50", "1.200,50", "(20.00)", "1234 (p)", "N/A"]:
    print(f"{raw!r:15s} -> {parse_currency(raw)}")

names = pd.Series(["SQM S.A.", "SQM SA", "sqm s.a.", "Albemarle Corp"])
unify_similar_names(names, threshold=80)
"""),
    markdown("""
## `encoding.zscore_scale` + `torch_trainer`

El mismo escalador y el mismo loop de entrenamiento (`train_with_early_stopping`,
piso de 100 épocas) que usan los 4 dominios -- acá sobre un slice pequeño y
real de features financieras, solo para mostrar la mecánica sin repetir el
modelo completo (ya está en `01_financial_bcch.ipynb`).
"""),
    code("""
import torch
from torch import nn
from src.toolkit.encoding import zscore_scale
from src.toolkit.torch_trainer import train_with_early_stopping

fin_features = pd.read_csv("../data/processed/financial/financial_features.csv").head(400)
X_cols = ["dolar_log_return_lag1", "dolar_volatility_5d", "tpm"]
scaled, stats = zscore_scale(fin_features, X_cols)
X = torch.tensor(scaled[X_cols].values, dtype=torch.float32)
y = torch.tensor(fin_features["target_next_return"].values, dtype=torch.float32).view(-1, 1)

model = nn.Sequential(nn.Linear(3, 8), nn.LeakyReLU(0.1), nn.Linear(8, 1))
result = train_with_early_stopping(model, X[:300], y[:300], X[300:], y[300:], loss_fn=nn.MSELoss(), min_epochs=100, max_epochs=150)
print(f"épocas corridas: {result.epochs_run}, mejor época: {result.best_epoch}")
"""),
    markdown("""
## Conclusión

Las mismas ~20 funciones de `src/toolkit/` (limpieza, outliers, texto, Excel,
encoding, visualización, entrenamiento) se reusan sin cambios en los 4
dominios -- lo único que cambia entre dominios es el dato de entrada real y
la interpretación del resultado, nunca la técnica.
"""),
]


if __name__ == "__main__":
    build_notebook(toolkit_cells, "notebooks/00_toolkit_demo.ipynb")
    build_notebook(financial_cells, "notebooks/01_financial_bcch.ipynb")
    build_notebook(mining_cells, "notebooks/02_mining_cochilco.ipynb")
    build_notebook(agriculture_cells, "notebooks/03_agriculture_worldbank.ipynb")
    build_notebook(consulting_cells, "notebooks/04_consulting_excel_dwh.ipynb")
