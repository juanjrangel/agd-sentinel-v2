"""Corrige el maestro jerárquico para que PyMC pueda muestrear la verosimilitud.

1. Descarta nulos en t, Cohesion o Friccion.
2. Unifica cualquier Sitio que contenga "Cortinas" en el grupo "Cortinas".
3. Ancla Cortinas 2017 a 2017-03-01 y recalcula t desde esa fecha.
4. Regenera Sitio_idx con pd.factorize.
5. Sobreescribe datosfinales/Dataset_Jerarquico_Maestro.csv.
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
log = logging.getLogger("corregir_maestro")

BASE_DIR = Path(__file__).resolve().parent
RUTA_MAESTRO = BASE_DIR / "datosfinales" / "Dataset_Jerarquico_Maestro.csv"
ORIGEN = pd.Timestamp("2017-03-01")
DIAS_POR_ANIO = 365.25
COLUMNAS_LIKELIHOOD = ["t", "Cohesion", "Friccion"]
COLUMNAS = [
    "Sitio",
    "Fecha",
    "Peso_Unitario",
    "Cohesion",
    "Friccion",
    "t",
    "Sitio_idx",
]


def eliminar_nulos_likelihood(df: pd.DataFrame) -> pd.DataFrame:
    nulos = df[COLUMNAS_LIKELIHOOD].isna().any(axis=1)
    limpio = df.loc[~nulos].copy()
    log.info(
        "Nulos en likelihood: eliminadas=%s, conservadas=%s",
        int(nulos.sum()),
        len(limpio),
    )
    return limpio


def unificar_cortinas(df: pd.DataFrame) -> pd.DataFrame:
    salida = df.copy()
    es_cortinas = salida["Sitio"].astype("string").str.contains(
        "Cortinas",
        na=False,
        regex=False,
    )
    salida.loc[es_cortinas, "Sitio"] = "Cortinas"
    log.info("Sitios unificados a Cortinas: %s", int(es_cortinas.sum()))
    return salida


def corregir_tiempo(df: pd.DataFrame) -> pd.DataFrame:
    """Cortinas 2017 queda en el origen de La China. t se rehace para todo el maestro."""
    salida = df.copy()
    salida["Fecha"] = pd.to_datetime(salida["Fecha"], errors="coerce")
    cortinas_2017 = salida["Sitio"].eq("Cortinas") & salida["Fecha"].dt.year.eq(2017)
    salida.loc[cortinas_2017, "Fecha"] = ORIGEN
    salida["t"] = (salida["Fecha"] - ORIGEN).dt.days / DIAS_POR_ANIO
    log.info(
        "Fechas Cortinas 2017 reescritas=%s, t_min=%.6f, t_max=%.6f",
        int(cortinas_2017.sum()),
        float(salida["t"].min()),
        float(salida["t"].max()),
    )
    return salida


def reindexar_sitio(df: pd.DataFrame) -> pd.DataFrame:
    salida = df.drop(columns=["Sitio_idx"]).copy()
    salida["Sitio_idx"] = pd.factorize(salida["Sitio"])[0]
    return salida


def validar(df: pd.DataFrame) -> None:
    if df[COLUMNAS_LIKELIHOOD].isna().any().any():
        raise RuntimeError("Quedaron nulos en t, Cohesion o Friccion.")
    if df["Sitio"].astype("string").str.contains("Cortinas/", na=False).any():
        raise RuntimeError("Cortinas sigue fragmentada por sondeo.")
    if df.loc[df["Sitio"].eq("Cortinas"), "Sitio_idx"].nunique() != 1:
        raise RuntimeError("Cortinas no comparte un único Sitio_idx.")
    if float(df["t"].min()) != 0.0 or (df["t"] < 0).any():
        raise RuntimeError("t mínimo no es exactamente 0.0.")
    cortinas_2017 = df["Sitio"].eq("Cortinas") & df["Fecha"].str.startswith("2017")
    if not df.loc[cortinas_2017, "Fecha"].eq("2017-03-01").all():
        raise RuntimeError("Cortinas 2017 no quedó en 2017-03-01.")


def main() -> pd.DataFrame:
    df = pd.read_csv(RUTA_MAESTRO, encoding="utf-8")
    idx_max_antes = int(df["Sitio_idx"].max())
    sitios_antes = int(df["Sitio"].nunique())
    log.info(
        "Entrada: filas=%s, sitios=%s, Sitio_idx_max=%s",
        len(df),
        sitios_antes,
        idx_max_antes,
    )

    df = eliminar_nulos_likelihood(df)
    df = unificar_cortinas(df)
    df = corregir_tiempo(df)
    df = reindexar_sitio(df)

    df["Fecha"] = df["Fecha"].dt.strftime("%Y-%m-%d")
    df = df[COLUMNAS]
    validar(df)

    idx_max_despues = int(df["Sitio_idx"].max())
    sitios_despues = int(df["Sitio"].nunique())
    df.to_csv(RUTA_MAESTRO, index=False, encoding="utf-8")
    log.info(
        "Salida: filas=%s, sitios=%s, Sitio_idx_max=%s, reducido=%s",
        len(df),
        sitios_despues,
        idx_max_despues,
        idx_max_despues < idx_max_antes,
    )
    print(f"FILAS_FINALES={len(df)}")
    print(f"SITIO_IDX_MAX_ANTES={idx_max_antes}")
    print(f"SITIO_IDX_MAX_DESPUES={idx_max_despues}")
    print(f"SITIOS_ANTES={sitios_antes}")
    print(f"SITIOS_DESPUES={sitios_despues}")
    return df


if __name__ == "__main__":
    main()
