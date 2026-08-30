"""Generador de datos sucios realistas para tickets de soporte IT/SaaS.

Produce un dataset sintético con los defectos típicos de datos reales de operaciones
TI: fechas en formatos y timezones mezclados, JSON anidado de profundidad variable,
nombres de empresa con typos/duplicados, montos en distintos formatos de moneda, y
valores faltantes representados de formas inconsistentes (NaN real, strings tipo
"null"/"N/A", y filas completamente vacías).
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = PROJECT_ROOT / "data" / "raw" / "messy_it_tickets.csv"

N_ROWS = 10_000
RANDOM_SEED = 42

# Fracción del total que son duplicados-con-variación (misma empresa, otro ticket) y
# filas completamente vacías (exports corruptos) — ambas cuentan dentro de N_ROWS,
# no se agregan por encima.
DUPLICATE_FRACTION = 0.03
BLANK_FRACTION = 0.005

BASE_COMPANIES = [
    "Nova Systems", "Bluepeak Technologies", "Orbital Cloud", "Vertex Analytics",
    "Northwind SaaS", "Cascade Data", "Ironclad Security", "Lumen Softworks",
    "Pinecrest Solutions", "Quanta Networks", "Silverline IT", "Zenith Digital",
    "Redwood Systems", "Brightbridge Tech", "Cobalt Cloud Services", "Fjord Analytics",
    "Meridian Software", "Anchorpoint Data", "Solace Technologies", "Driftwood IT",
]
DOMAINS = ["corp.com", "saas.io", "helpdesk.co", "cloudtech.io", "systems.com", "cloud.net"]
PRIORITIES = ["Low", "Medium", "High", "Critical"]
STATUSES = ["Open", "Pending", "Resolved", "Closed"]
CATEGORIES = ["Billing", "Technical", "Account", "Bug Report", "Feature Request"]
AGENTS = [
    "A. Rivera", "J. Kim", "M. Alvarez", "S. Novak", "T. Osei", "P. Duarte",
    "L. Wallace", "R. Chen", "E. Fontaine", "D. Okafor",
]
BROWSERS = ["Chrome", "Firefox", "Safari", "Edge"]
OS_LIST = ["Windows", "macOS", "Linux", "iOS", "Android"]
PLAN_TIERS = ["free", "pro", "business", "enterprise"]

NULL_LIKE_STRINGS = ["", "NaN", "null", "NULL", "N/A", "NA", "-", "None", "?"]

CURRENCY_TEMPLATES = [
    lambda v: f"${v:,.2f}",
    lambda v: f"{v:,.2f} €".replace(",", "X").replace(".", ",").replace("X", "."),
    lambda v: f"€{v:,.2f}",
    lambda v: f"{v:,.2f} USD",
    lambda v: f"US$ {v:,.2f}",
    lambda v: f"{v:.0f},00 EUR",
]

DATE_FORMATS = [
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%dT%H:%M:%S%z",
    "%d/%m/%Y %H:%M",
    "%m/%d/%Y %I:%M %p",
    "%B %d, %Y %H:%M",
    "%d-%b-%Y %H:%M:%S",
]
TIMEZONE_SUFFIXES = ["", "Z", "+00:00", "-05:00", "+02:00", "UTC", "EST", "PST", "CET"]


def _typo(name: str, rng: random.Random) -> str:
    """Aplica una variación aleatoria de formato/tipeo a un nombre de empresa."""
    variants = [
        lambda s: s.upper(),
        lambda s: s.lower(),
        lambda s: s + rng.choice([" Inc.", " Inc", " LLC", " Corp.", " Ltd."]),
        lambda s: s.replace("o", "0", 1) if "o" in s else s,
        lambda s: s.replace(" ", "  "),
        lambda s: s[:-1] if len(s) > 3 else s,
        lambda s: s.replace("a", "e", 1) if "a" in s else s,
        lambda s: " " + s + " ",
        lambda s: s,
    ]
    return rng.choice(variants)(name)


def _random_date(rng: random.Random) -> pd.Timestamp:
    return pd.Timestamp("2024-01-01") + pd.Timedelta(
        days=rng.randint(0, 700), hours=rng.randint(0, 23), minutes=rng.randint(0, 59)
    )


def _format_date_messy(ts: pd.Timestamp, rng: random.Random) -> str:
    fmt = rng.choice(DATE_FORMATS)
    text = ts.strftime(fmt)
    if "%z" not in fmt:
        suffix = rng.choice(TIMEZONE_SUFFIXES)
        if suffix:
            text = f"{text} {suffix}"
    return text


def _random_metadata(rng: random.Random) -> str:
    """JSON de profundidad variable (0 a 3 niveles anidados) para `user_metadata`."""
    depth_choice = rng.random()
    if depth_choice < 0.15:
        return rng.choice(["{}", "null", ""])

    payload: dict = {"os": rng.choice(OS_LIST)}
    if depth_choice < 0.45:
        return json.dumps(payload)

    payload["browser"] = {"name": rng.choice(BROWSERS), "version": f"{rng.randint(90, 125)}.0"}
    if depth_choice < 0.7:
        return json.dumps(payload)

    payload["plan"] = {
        "tier": rng.choice(PLAN_TIERS),
        "seats": rng.randint(1, 200),
        "billing": {"cycle": rng.choice(["monthly", "annual"]), "auto_renew": rng.choice([True, False])},
    }
    payload["tags"] = rng.sample(["vip", "beta", "trial", "churn_risk", "champion"], k=rng.randint(0, 3))
    return json.dumps(payload)


def _random_cost(rng: random.Random) -> str:
    value = round(rng.uniform(5, 5000), 2)
    text = rng.choice(CURRENCY_TEMPLATES)(value)
    if rng.random() < 0.05:
        text = "-" + text  # reembolsos
    return text


def _resolution_hours(category: str, priority: str, rng: random.Random) -> int:
    """Horas hasta resolución, causalmente ligadas a categoría y prioridad -- no
    ruido independiente -- para que un modelo de pronóstico downstream (ver
    `src/models/forecast_response_time.py`) tenga una señal real que recuperar,
    en vez de ajustar contra ruido puro. Base por categoría (Billing/Account se
    resuelven rápido, suelen ser administrativos; Bug Report/Feature Request son
    trabajo de ingeniería, tardan más) x multiplicador por prioridad (Critical se
    escala y se resuelve rápido; Low se posterga), con ruido log-normal multiplicativo
    encima para variabilidad realista entre tickets similares.
    """
    base_hours = {
        "Billing": 8.0,
        "Account": 12.0,
        "Technical": 30.0,
        "Bug Report": 48.0,
        "Feature Request": 60.0,
    }[category]
    priority_multiplier = {"Critical": 0.3, "High": 0.6, "Medium": 1.0, "Low": 1.8}[priority]

    noise = rng.lognormvariate(0, 0.5)
    hours = base_hours * priority_multiplier * noise
    return max(1, min(240, round(hours)))


def _fresh_ticket(ticket_id: str, rng: random.Random) -> dict:
    created = _random_date(rng)
    base_priority = rng.choice(PRIORITIES)
    base_category = rng.choice(CATEGORIES)
    resolved = (
        created + pd.Timedelta(hours=_resolution_hours(base_category, base_priority, rng))
        if rng.random() > 0.08
        else None
    )

    row = {
        "ticket_id": ticket_id,
        "company_name": _typo(rng.choice(BASE_COMPANIES), rng),
        "customer_email": f"user{rng.randint(0, 999_999)}@{rng.choice(DOMAINS)}",
        "created_at": _format_date_messy(created, rng),
        "resolved_at": _format_date_messy(resolved, rng) if resolved is not None else rng.choice(NULL_LIKE_STRINGS),
        "priority": rng.choice([base_priority, base_priority.lower(), base_priority.upper()]),
        "status": rng.choice(STATUSES),
        "category": base_category,
        "agent_name": rng.choice(AGENTS),
        "cost": _random_cost(rng),
        "user_metadata": _random_metadata(rng),
    }

    for col in ["priority", "category", "agent_name", "cost", "company_name"]:
        if rng.random() < 0.04:
            row[col] = rng.choice(NULL_LIKE_STRINGS)

    return row


def generate_dirty_dataset(n_rows: int = N_ROWS, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Genera un DataFrame sintético de `n_rows` tickets de soporte IT con defectos realistas.

    El total incluye (no suma aparte) duplicados con variaciones de nombre de empresa y
    filas completamente vacías, en las proporciones `DUPLICATE_FRACTION`/`BLANK_FRACTION`.
    """
    rng = random.Random(seed)

    n_blank = max(1, int(n_rows * BLANK_FRACTION))
    n_dupe = max(1, int(n_rows * DUPLICATE_FRACTION))
    n_fresh = n_rows - n_blank - n_dupe

    fresh_rows = [_fresh_ticket(f"TCK-{100000 + i}", rng) for i in range(n_fresh)]

    null_like_lower = {s.lower() for s in NULL_LIKE_STRINGS}
    dupe_rows = []
    for i in range(n_dupe):
        source = dict(rng.choice(fresh_rows))
        source["ticket_id"] = f"TCK-{200000 + i}"
        company = str(source["company_name"]).strip()
        # No aplicar typos a un company_name que ya era un placeholder de nulo: mutarlo
        # (p. ej. "NaN" -> "NAN") lo volvería un texto no reconocible como faltante.
        if company.lower() not in null_like_lower:
            source["company_name"] = _typo(company, rng)
        dupe_rows.append(source)

    columns = list(fresh_rows[0].keys())
    blank_rows = [{col: np.nan for col in columns} for _ in range(n_blank)]

    df = pd.DataFrame(fresh_rows + dupe_rows + blank_rows)
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


if __name__ == "__main__":
    df = generate_dirty_dataset()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Dataset sucio generado: {df.shape[0]:,} filas x {df.shape[1]} columnas")
    print(f"Guardado en {OUTPUT_PATH}")
