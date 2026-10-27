"""Sustituye cohesiones nulas o casi nulas por un epsilon de 0.1 kPa.

PyMC evalúa ln(c). c = 0 es indefinido y c < 0.1 kPa no aporta
resistencia distinguible en este conjunto, así que el piso es 0.1.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("cohesion_epsilon")

BASE_DIR = Path(__file__).resolve().parent
RUTA = BASE_DIR / "datosfinales" / "Dataset_Jerarquico_Maestro2.csv"
EPSILON_KPA = 0.1


def main() -> int:
    df = pd.read_csv(RUTA, encoding="utf-8")
    cohesion = pd.to_numeric(df["Cohesion"], errors="coerce")
    reemplazar = cohesion.eq(0.0) | cohesion.lt(EPSILON_KPA)
    n = int(reemplazar.sum())
    log.info("Cohesiones a reemplazar por %.1f kPa: %s", EPSILON_KPA, n)
    if n:
        log.info(
            "Valores originales: %s",
            cohesion.loc[reemplazar].tolist(),
        )
    df.loc[reemplazar, "Cohesion"] = EPSILON_KPA
    df.to_csv(RUTA, index=False, encoding="utf-8")

    verificacion = pd.read_csv(RUTA, encoding="utf-8")
    cohesion_nueva = pd.to_numeric(verificacion["Cohesion"], errors="coerce")
    restantes = int((cohesion_nueva.eq(0.0) | cohesion_nueva.lt(EPSILON_KPA)).sum())
    if restantes != 0:
        raise RuntimeError(f"Quedaron {restantes} cohesiones bajo el epsilon.")
    if len(verificacion) != len(df):
        raise RuntimeError("El número de filas cambió al guardar.")
    log.info("Archivo sobreescrito: %s", RUTA)
    print(f"VALORES_REEMPLAZADOS={n}")
    return n


if __name__ == "__main__":
    main()
