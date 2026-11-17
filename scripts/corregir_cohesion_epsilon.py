"""Sustituye cohesiones nulas o casi nulas por un epsilon de 0.1 kPa.

El piso evita problemas si se trabaja con ln(c), que no existe para c = 0.
Una cohesión menor que 0.1 kPa no aporta resistencia distinguible en este
conjunto, así que el cambio no altera la resistencia en términos prácticos.

Lee datos/finales/Dataset_Jerarquico_Maestro2.csv y escribe
datos/finales/Dataset_Jerarquico_Maestro3.csv.
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

RAIZ = Path(__file__).resolve().parent.parent   # carpeta raíz del repositorio
DATOS_FINALES = RAIZ / "datos" / "finales"
RUTA_ENTRADA = DATOS_FINALES / "Dataset_Jerarquico_Maestro2.csv"
RUTA_SALIDA = DATOS_FINALES / "Dataset_Jerarquico_Maestro3.csv"
EPSILON_KPA = 0.1


def main() -> int:
    df = pd.read_csv(RUTA_ENTRADA, encoding="utf-8")
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
    df.to_csv(RUTA_SALIDA, index=False, encoding="utf-8")

    verificacion = pd.read_csv(RUTA_SALIDA, encoding="utf-8")
    cohesion_nueva = pd.to_numeric(verificacion["Cohesion"], errors="coerce")
    restantes = int((cohesion_nueva.eq(0.0) | cohesion_nueva.lt(EPSILON_KPA)).sum())
    if restantes != 0:
        raise RuntimeError(f"Quedaron {restantes} cohesiones bajo el epsilon.")
    if len(verificacion) != len(df):
        raise RuntimeError("El número de filas cambió al guardar.")
    log.info("Archivo escrito: %s", RUTA_SALIDA)
    print(f"VALORES_REEMPLAZADOS={n}")
    return n


if __name__ == "__main__":
    main()
