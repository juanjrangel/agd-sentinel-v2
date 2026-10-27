"""Unifica las tres bases geotécnicas en un dataset jerárquico maestro.

Dependencias: pandas, openpyxl.
Ejecución:
    uv run --with pandas --with openpyxl python unificar_dataset_jerarquico.py

Convención de fecha incompleta: un año suelto se ancla al 1 de enero y un
mes/año al día 1 de ese mes. Es el inicio del intervalo conocido, el mismo
criterio con el que pandas interpreta ``%Y`` y ``%m/%Y``. La China ya trae
día explícito y su ``t`` no se recalcula.
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
log = logging.getLogger("dataset_jerarquico")

BASE_DIR = Path(__file__).resolve().parent
RUTA_CORTINAS = BASE_DIR / "cortinas_campaign_data.csv"
RUTA_LA_CHINA = BASE_DIR / "datos_la_china_depurados.csv"
RUTA_CRUDOS = BASE_DIR / "Datos_Crudos_Extraidos.xlsx"
HOJA_CRUDOS = "Datos_Crudos"
DIR_SALIDA = BASE_DIR / "datosfinales"
RUTA_SALIDA = DIR_SALIDA / "Dataset_Jerarquico_Maestro.csv"

COLUMNAS = ["Sitio", "Fecha", "Peso_Unitario", "Cohesion", "Friccion", "t"]
ORIGEN_LA_CHINA = pd.Timestamp("2017-03-01")
DIAS_POR_ANIO = 365.25
UMBRAL_COHESION_KPA = 100.0
FRICCION_MIN_GRADOS = 0.0
FRICCION_MAX_GRADOS = 90.0


def cargar_fuentes() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cortinas = pd.read_csv(RUTA_CORTINAS, encoding="utf-8")
    la_china = pd.read_csv(RUTA_LA_CHINA, encoding="utf-8")
    crudos = pd.read_excel(RUTA_CRUDOS, sheet_name=HOJA_CRUDOS, engine="openpyxl")
    log.info(
        "Carga: cortinas=%s, la_china=%s, Datos_Crudos=%s",
        len(cortinas),
        len(la_china),
        len(crudos),
    )
    return cortinas, la_china, crudos


def filtrar_duplicados_avr(crudos: pd.DataFrame) -> pd.DataFrame:
    """AVR ya está en La China; se retira antes de homologar columnas."""
    ubicacion = crudos["Sitio_Ubicacion"].astype("string")
    es_avr = ubicacion.str.contains("AVR", na=False, regex=False)
    conservados = crudos.loc[~es_avr].copy()
    log.info(
        "Filtro AVR: eliminadas=%s, conservadas=%s",
        int(es_avr.sum()),
        len(conservados),
    )
    return conservados


def parse_fechas(serie: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(serie):
        texto = serie.astype("Int64").astype("string")
    else:
        texto = serie.astype("string").str.strip()

    iso = pd.to_datetime(texto, format="%Y-%m-%d", errors="coerce")
    mes_anio = pd.to_datetime(texto, format="%m/%Y", errors="coerce")
    solo_anio = texto.str.fullmatch(r"\d{4}", na=False)
    anio = pd.to_datetime(texto.where(solo_anio), format="%Y", errors="coerce")
    return iso.fillna(mes_anio).fillna(anio)


def _marco_canonico(
    sitio: pd.Series,
    fecha: pd.Series,
    peso: pd.Series,
    cohesion: pd.Series,
    friccion: pd.Series,
    t: pd.Series,
) -> pd.DataFrame:
    marco = pd.DataFrame(
        {
            "Sitio": sitio.astype("string").str.strip(),
            "Fecha": parse_fechas(fecha),
            "Peso_Unitario": pd.to_numeric(peso, errors="coerce"),
            "Cohesion": pd.to_numeric(cohesion, errors="coerce"),
            "Friccion": pd.to_numeric(friccion, errors="coerce"),
            "t": pd.to_numeric(t, errors="coerce"),
        }
    )
    return marco[COLUMNAS]


def homologar_cortinas(cortinas: pd.DataFrame) -> pd.DataFrame:
    """El sondeo es el identificador espacial. El prefijo evita colisión con otros S1/S2."""
    sitio = "Cortinas/" + cortinas["borehole"].astype("string").str.strip()
    vacio = pd.Series(pd.NA, index=cortinas.index)
    return _marco_canonico(
        sitio=sitio,
        fecha=cortinas["year"],
        peso=vacio,
        cohesion=cortinas["cohesion_kPa"],
        friccion=cortinas["friction_deg"],
        t=vacio,
    )


def homologar_la_china(la_china: pd.DataFrame) -> pd.DataFrame:
    return _marco_canonico(
        sitio=la_china["Sitio"],
        fecha=la_china["Fecha"],
        peso=la_china["Peso_Unitario"],
        cohesion=la_china["Cohesion"],
        friccion=la_china["Friccion"],
        t=la_china["t"],
    )


def homologar_crudos(crudos: pd.DataFrame) -> pd.DataFrame:
    vacio = pd.Series(pd.NA, index=crudos.index)
    return _marco_canonico(
        sitio=crudos["Sitio_Ubicacion"],
        fecha=crudos["Fecha_Exploracion"],
        peso=crudos["Gamma_kN_m3"],
        cohesion=crudos["Cohesion_kPa"],
        friccion=crudos["Friccion_grados"],
        t=vacio,
    )


def limpiar_cortinas(cortinas: pd.DataFrame) -> pd.DataFrame:
    """Resistencia nula o físicamente imposible no entra al modelo."""
    resistencia_nula = cortinas["Cohesion"].isna() | cortinas["Friccion"].isna()
    cohesion_imposible = (cortinas["Cohesion"] < 0) | (
        cortinas["Cohesion"] > UMBRAL_COHESION_KPA
    )
    friccion_imposible = (cortinas["Friccion"] < FRICCION_MIN_GRADOS) | (
        cortinas["Friccion"] > FRICCION_MAX_GRADOS
    )
    descartar = resistencia_nula | cohesion_imposible | friccion_imposible
    limpio = cortinas.loc[~descartar].copy()
    log.info(
        "Limpieza cortinas: entrada=%s, nula=%s, cohesion_imposible=%s, "
        "friccion_imposible=%s, conservadas=%s",
        len(cortinas),
        int(resistencia_nula.sum()),
        int(cohesion_imposible.sum()),
        int(friccion_imposible.sum()),
        len(limpio),
    )
    return limpio


def calcular_t(df: pd.DataFrame) -> pd.DataFrame:
    """Solo completa ``t`` ausente. El valor de La China ya está anclado a t=0."""
    salida = df.copy()
    faltante = salida["t"].isna()
    delta_dias = (salida.loc[faltante, "Fecha"] - ORIGEN_LA_CHINA).dt.days
    salida.loc[faltante, "t"] = delta_dias / DIAS_POR_ANIO
    sin_fecha = int((faltante & salida["t"].isna()).sum())
    log.info(
        "t calculada en %s filas; %s siguen vacías por fecha no interpretable",
        int(faltante.sum() - sin_fecha),
        sin_fecha,
    )
    return salida


def imputar_peso_unitario(df: pd.DataFrame) -> pd.DataFrame:
    salida = df.copy()
    media = float(salida["Peso_Unitario"].mean())
    vacios = int(salida["Peso_Unitario"].isna().sum())
    salida["Peso_Unitario"] = salida["Peso_Unitario"].fillna(media)
    log.info(
        "Imputación Peso_Unitario: media_global=%.6f, celdas_rellenadas=%s",
        media,
        vacios,
    )
    return salida


def asignar_sitio_idx(df: pd.DataFrame) -> pd.DataFrame:
    salida = df.copy()
    salida["Sitio_idx"] = pd.factorize(salida["Sitio"])[0]
    log.info("Sitios únicos=%s", int(salida["Sitio"].nunique()))
    return salida


def exportar(df: pd.DataFrame) -> Path:
    DIR_SALIDA.mkdir(parents=True, exist_ok=True)
    exportable = df.copy()
    exportable["Fecha"] = exportable["Fecha"].dt.strftime("%Y-%m-%d")
    exportable.to_csv(RUTA_SALIDA, index=False, encoding="utf-8")
    log.info("Exportado %s (%s filas)", RUTA_SALIDA, len(exportable))
    return RUTA_SALIDA


def main() -> pd.DataFrame:
    cortinas, la_china, crudos = cargar_fuentes()
    crudos = filtrar_duplicados_avr(crudos)

    cortinas = homologar_cortinas(cortinas)
    la_china = homologar_la_china(la_china)
    crudos = homologar_crudos(crudos)

    cortinas = limpiar_cortinas(cortinas)

    cortinas = calcular_t(cortinas)
    la_china = calcular_t(la_china)
    crudos = calcular_t(crudos)

    maestro = pd.concat([cortinas, la_china, crudos], ignore_index=True)
    maestro = maestro[COLUMNAS]
    log.info(
        "Fusión: cortinas=%s, la_china=%s, crudos=%s, total=%s",
        len(cortinas),
        len(la_china),
        len(crudos),
        len(maestro),
    )

    maestro = imputar_peso_unitario(maestro)
    maestro = asignar_sitio_idx(maestro)
    exportar(maestro)
    return maestro


if __name__ == "__main__":
    resultado = main()
    print(f"FILAS_MAESTRAS={len(resultado)}")
