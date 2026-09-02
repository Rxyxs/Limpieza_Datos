"""Descarga la producción real de cobre de mina por empresa, mensual, publicada
por COCHILCO (Comisión Chilena del Cobre, el organismo regulador/estadístico
del cobre en Chile) -- un archivo `.xlsx` público, sin autenticación.

A diferencia de `financial_bcch` (una API JSON ya tabular), acá el desafío de
limpieza no está en la descarga sino en el archivo mismo: replica el layout
visual de un reporte de Excel de organismo público (título institucional,
encabezado en la fila 7, filas de resumen anual intercaladas entre los meses,
fila de cita al pie) -- ver `clean.py` para el detalle de cada mezcla real
encontrada y cómo se trata.
"""
from __future__ import annotations

from pathlib import Path

import requests

RAW_DIR = Path(__file__).resolve().parents[3] / "data" / "raw" / "mining"
BASE_URL = "https://www.cochilco.cl/web/download"

# Producción mensual de cobre de mina por empresa (dataset primario).
COBRE_URL = f"{BASE_URL}/951/cobre/12683/produccion-de-cobre-mina-por-empresa-mensual.xlsx"
COBRE_FILENAME = "cochilco_produccion_mensual.xlsx"


def _download(url: str, out_path: Path) -> Path:
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    out_path.write_bytes(resp.content)
    return out_path


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out_path = _download(COBRE_URL, RAW_DIR / COBRE_FILENAME)
    size_kb = out_path.stat().st_size / 1024
    print(f"cobre: {size_kb:,.0f} KB -> {out_path}")


if __name__ == "__main__":
    main()
