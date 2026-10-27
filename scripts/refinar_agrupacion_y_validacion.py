"""Agrupa estratos de la Variante Río Negro por sondeo y aparta Cortinas post-2021.

El likelihood se calibra hasta el colapso de 2021. Las campañas de 2023 y 2025
en Cortinas quedan fuera de muestra, en Dataset_Validacion_Cortinas.csv.
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
log = logging.getLogger("agrupacion_validacion")

BASE_DIR = Path(__file__).resolve().parent
DIR_SALIDA = BASE_DIR / "datosfinales"
RUTA_ENTRADA = DIR_SALIDA / "Dataset_Jerarquico_Maestro3.csv"
RUTA_ENTRENAMIENTO = DIR_SALIDA / "Dataset_Jerarquico_Maestro_Final.csv"
RUTA_VALIDACION = DIR_SALIDA / "Dataset_Validacion_Cortinas.csv"

SONDEOS_RIO_NEGRO = {
    "R-01": "Variante Río Negro, R-01",
    "R-02": "Variante Río Negro, R-02",
    "R-03": "Variante Río Negro, R-03",
}
ANIO_CORTE_CALIBRACION = 2021


def agrupar_sondeos_rio_negro(df: pd.DataFrame) -> pd.DataFrame:
    salida = df.copy()
    sitio = salida["Sitio"].astype("string")
    for codigo, nombre in SONDEOS_RIO_NEGRO.items():
        mascara = sitio.str.contains(codigo, na=False, regex=False)
        salida.loc[mascara, "Sitio"] = nombre
        log.info("Sondeo %s unificado: %s filas", codigo, int(mascara.sum()))
        sitio = salida["Sitio"].astype("string")
    return salida


def separar_validacion_cortinas(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    anio = pd.to_datetime(df["Fecha"], errors="coerce").dt.year
    es_validacion = df["Sitio"].eq("Cortinas") & anio.gt(ANIO_CORTE_CALIBRACION)
    validacion = df.loc[es_validacion].copy()
    entrenamiento = df.loc[~es_validacion].copy()
    log.info(
        "Validación Cortinas (año > %s): %s filas; entrenamiento: %s filas",
        ANIO_CORTE_CALIBRACION,
        len(validacion),
        len(entrenamiento),
    )
    return entrenamiento, validacion


def reindexar_sitio(df: pd.DataFrame) -> pd.DataFrame:
    salida = df.drop(columns=["Sitio_idx"]).copy()
    salida["Sitio_idx"] = pd.factorize(salida["Sitio"])[0]
    return salida


def validar(entrenamiento: pd.DataFrame, validacion: pd.DataFrame) -> None:
    anio_ent = pd.to_datetime(entrenamiento["Fecha"], errors="coerce").dt.year
    anio_val = pd.to_datetime(validacion["Fecha"], errors="coerce").dt.year
    if (entrenamiento["Sitio"].eq("Cortinas") & anio_ent.gt(ANIO_CORTE_CALIBRACION)).any():
        raise RuntimeError("El entrenamiento todavía contiene Cortinas posterior a 2021.")
    if not validacion["Sitio"].eq("Cortinas").all():
        raise RuntimeError("La validación incluye sitios distintos de Cortinas.")
    if not anio_val.gt(ANIO_CORTE_CALIBRACION).all():
        raise RuntimeError("La validación incluye años de calibración.")
    for nombre in SONDEOS_RIO_NEGRO.values():
        if entrenamiento["Sitio"].eq(nombre).sum() != 2:
            raise RuntimeError(f"{nombre} no quedó con sus dos estratos.")
    if entrenamiento["Sitio"].astype("string").str.contains("0.0–", na=False).any():
        raise RuntimeError("Quedó un rango de profundidad en un sondeo agrupado.")
    if int(entrenamiento["Sitio_idx"].max()) != int(entrenamiento["Sitio"].nunique()) - 1:
        raise RuntimeError("Sitio_idx no es un factorize compacto.")


def main() -> None:
    df = pd.read_csv(RUTA_ENTRADA, encoding="utf-8")
    idx_max_antes = int(df["Sitio_idx"].max())
    log.info(
        "Entrada: filas=%s, sitios=%s, Sitio_idx_max=%s",
        len(df),
        int(df["Sitio"].nunique()),
        idx_max_antes,
    )

    df = agrupar_sondeos_rio_negro(df)
    entrenamiento, validacion = separar_validacion_cortinas(df)
    entrenamiento = reindexar_sitio(entrenamiento)
    validar(entrenamiento, validacion)

    idx_max_despues = int(entrenamiento["Sitio_idx"].max())
    DIR_SALIDA.mkdir(parents=True, exist_ok=True)
    validacion.to_csv(RUTA_VALIDACION, index=False, encoding="utf-8")
    entrenamiento.to_csv(RUTA_ENTRENAMIENTO, index=False, encoding="utf-8")

    log.info("Validación: %s", RUTA_VALIDACION)
    log.info("Entrenamiento: %s", RUTA_ENTRENAMIENTO)
    print(f"FILAS_ENTRENAMIENTO={len(entrenamiento)}")
    print(f"FILAS_VALIDACION={len(validacion)}")
    print(f"SITIO_IDX_MAX_ANTES={idx_max_antes}")
    print(f"SITIO_IDX_MAX_DESPUES={idx_max_despues}")
    print(f"SITIOS_ENTRENAMIENTO={int(entrenamiento['Sitio'].nunique())}")


if __name__ == "__main__":
    main()
