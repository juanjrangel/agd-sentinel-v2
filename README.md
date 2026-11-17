# AGD-Sentinel V2 — Modelo jerárquico de deterioro geotécnico acumulado

Proyecto final de Métodos de Inferencia Bayesiana (Prof. David Romo Bucheli, PhD).

Juan José Rangel (2243554) · Santiago Sepúlveda (2243557) · Jerónimo Andrés Mateo Bazán (2243590)

Modelo bayesiano jerárquico (PyMC) de la pérdida de cohesión en taludes coluviales de Toledo, Norte de Santander. Usa 48 ensayos de 27 taludes para estimar una tasa regional de deterioro y la cohesión inicial de cada talud, proyecta el factor de seguridad de Cortinas en dos escenarios de fricción (pico 30.8° y residual 21.3°) y compara las predicciones con métodos ingenuos sobre 2 ensayos de Cortinas posteriores a la falla y 29 ensayos reservados de otros taludes.

**Resultado principal:** la cohesión se pierde a 1.37 kPa/año (intervalo de credibilidad del 94 %: 0.36–2.40), pero Cortinas recuperó cohesión después de la falla (16.3 y 17.2 kPa medidos en 2023 y 2025, frente a 9.9 y 7.4 kPa predichos por la recta de deterioro).

## Contenido

```
agd_modelo_jerarquico_final.ipynb   Cuaderno con el análisis completo y sus salidas
agd_modelo_jerarquico_final.py      El mismo análisis como script
generar_figuras_final.py            Figuras a partir de los resultados guardados
datos/
  crudos/                           Fuentes originales (campaña de Cortinas, La China, informes técnicos)
  finales/                          Base de entrenamiento (48 ensayos) y validación de Cortinas (2 ensayos)
scripts/                            Depuración de datos: de datos/crudos a datos/finales
requirements.txt                    Librerías necesarias
```

## Cómo ejecutarlo

**Google Colab o Jupyter.** Suba el cuaderno, `generar_figuras_final.py` y los dos CSV de `datos/finales/` a la misma carpeta, abra el cuaderno y use *Ejecutar todo*.

**Python local (3.10 o posterior).** Desde la carpeta del repositorio:

```
pip install -r requirements.txt
python agd_modelo_jerarquico_final.py
python generar_figuras_final.py
```

El análisis tarda unos 3 minutos. Las tablas quedan en `resultados/` y las figuras en `figuras/`.

**Reconstruir la base desde las fuentes crudas (opcional).** Ejecute los cuatro scripts en este orden; los CSV que producen en `datos/finales/` son los mismos que ya están en el repositorio:

```
python scripts/unificar_dataset_jerarquico.py
python scripts/corregir_dataset_maestro.py
python scripts/corregir_cohesion_epsilon.py
python scripts/refinar_agrupacion_y_validacion.py
```

## Reproducibilidad

Semilla 42 en todos los muestreos y simulaciones. NUTS con 4 cadenas, 2000 iteraciones de calentamiento, 3000 muestras por cadena y `target_accept = 0.95`. Versiones usadas: Python 3.12, PyMC 5.28.5, ArviZ 0.23.4, NumPy 2.2.6. En Windows sin compilador de C, el código usa el backend Numba de PyTensor; el modelo y los resultados no cambian.

Los datos son ensayos geotécnicos de informes técnicos de la región y de la campaña abierta de Cortinas; no contienen información personal.
