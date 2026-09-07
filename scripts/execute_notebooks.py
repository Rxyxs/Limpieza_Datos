"""Ejecuta los notebooks en su propio directorio y los guarda con sus salidas."""
import sys
from pathlib import Path

import nbclient
import nbformat

for name in sys.argv[1:]:
    path = Path("notebooks") / name
    nb = nbformat.read(path, as_version=4)
    nbclient.NotebookClient(
        nb, timeout=3600, kernel_name="python3", resources={"metadata": {"path": "notebooks"}}
    ).execute()
    nbformat.write(nb, path)
    errores = [
        o for c in nb.cells for o in c.get("outputs", []) if o.get("output_type") == "error"
    ]
    print(f"{name}: {'ERROR -> ' + errores[0]['evalue'] if errores else 'ok'}", flush=True)
