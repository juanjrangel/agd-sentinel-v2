"""Figuras de AGD-Sentinel V2.

Lee los resultados que guarda `agd_modelo_jerarquico_final.py` en `resultados/`
(no vuelve a muestrear) y guarda cada figura en `figuras/` como PNG (300 dpi) y PDF.

Cada título resume el resultado que muestra la figura y las etiquetas van junto a
las curvas, para que las figuras se lean sin leyenda.
"""
import json
from pathlib import Path

import arviz as az
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    CARPETA = Path(__file__).resolve().parent
except NameError:
    CARPETA = Path.cwd()
RES, FIG = CARPETA / "resultados", CARPETA / "figuras"
FIG.mkdir(exist_ok=True)


def buscar_csv(nombre):
    for c in [CARPETA, CARPETA / "datos" / "finales", CARPETA / "datosfinales",
              Path("/content"), Path("/content/datosfinales")]:
        if (c / nombre).exists():
            return c / nombre
    raise FileNotFoundError(nombre)


# Colores con un solo papel cada uno (paleta validada para daltonismo)
AZUL = "#2a78d6"        # modelo bayesiano
AZUL_CLARO = "#a9c8ef"  # banda predictiva
NARANJA = "#eb6834"     # escenario de fricción residual / Cortinas
ROJO = "#e34948"        # datos reservados para validar
GRIS = "#8a8984"        # métodos ingenuos y referencias
TINTA, TINTA_2 = "#0b0b0b", "#52514e"

plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 13, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#b5b4ae",
    "axes.labelcolor": TINTA_2, "xtick.color": TINTA_2, "ytick.color": TINTA_2,
    "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.8, "lines.linewidth": 2,
    "figure.dpi": 100, "savefig.bbox": "tight", "font.family": "DejaVu Sans",
})


def guardar(fig, nombre):
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"{nombre}.{ext}", dpi=300)
    plt.close(fig)
    print(f"  figuras/{nombre}.png")


def num(x, d=1):
    """Número con punto decimal (mismo formato del informe)."""
    return f"{x:.{d}f}"


# --------------------------------------------------------------------------
# Datos y resultados
# --------------------------------------------------------------------------
train = pd.read_csv(buscar_csv("Dataset_Jerarquico_Maestro_Final.csv"))
valid = pd.read_csv(buscar_csv("Dataset_Validacion_Cortinas.csv"))
R = json.loads((RES / "resumen.json").read_text(encoding="utf-8"))
pooling = pd.read_csv(RES / "pooling_sitios.csv")
ppc = pd.read_csv(RES / "ppc.csv")
proy = pd.read_csv(RES / "cortinas_proyeccion.csv")
clave = pd.read_csv(RES / "cortinas_puntos_clave.csv")
sens = pd.read_csv(RES / "sensibilidad_priors.csv")
ing_c = pd.read_csv(RES / "ingenuo_cortinas.csv")
ing_c_det = pd.read_csv(RES / "ingenuo_cortinas_detalle.csv")
ing_r = pd.read_csv(RES / "ingenuo_regional.csv")
cv = pd.read_csv(RES / "ingenuo_regional_detalle.csv")
idata = az.from_netcdf(RES / "idata_base.nc")
cort = train[train["Sitio_idx"] == 0]
T_FALLA = float(cort["t"].max())
print("Generando figuras en figuras/ ...")

# --------------------------------------------------------------------------
# Figura 1. Estructura de los datos
# --------------------------------------------------------------------------
n_sitio = train.groupby("Sitio_idx").size().sort_values(ascending=False)
fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1.15, 1]})
col = [ROJO if i == 0 else GRIS for i in n_sitio.index]
a.bar(np.arange(len(n_sitio)), n_sitio.values, color=col, width=0.8)
a.set(xticks=[], xlabel="27 taludes, ordenados por número de ensayos", ylabel="Ensayos por talud")
a.text(0.4, 11.2, "PR-21-01-38: 11 ensayos, una sola fecha", fontsize=9.5, color=TINTA_2)
pos_c = list(n_sitio.index).index(0)
a.annotate("Cortinas: único talud\ncon dos fechas", xy=(pos_c, 2.1), xytext=(pos_c + 3, 6.2),
           fontsize=9.5, color=ROJO, arrowprops=dict(arrowstyle="-", color=ROJO, lw=1))
a.text(17.5, 1.35, f"{(n_sitio == 1).sum()} taludes con un solo ensayo", fontsize=9.5, color=TINTA_2, ha="center")
a.grid(axis="x", visible=False)
a.set_title("(a) La mayoría de taludes tiene un solo ensayo", fontsize=11.5)

otros = train[train["Sitio_idx"] != 0]
b.scatter(otros["t"] + np.random.default_rng(1).uniform(-0.08, 0.08, len(otros)), otros["Cohesion"],
          s=28, color=GRIS, alpha=0.75, edgecolor="white", linewidth=0.5)
b.plot(cort["t"], cort["Cohesion"], "-o", color=ROJO, ms=8, mec="white", mew=1.5)
b.plot(valid["t"], valid["Cohesion"], "D", color=ROJO, ms=8, mfc="white", mew=2)
b.text(0.25, 19.5, "Cortinas 2017", color=ROJO, fontsize=9.5)
b.text(T_FALLA - 0.35, 6.0, "Cortinas 2021\n(después de la falla)", color=ROJO, fontsize=9.5, ha="center")
b.text(valid["t"].iloc[1] - 0.15, 19.3, "Reservados para\nvalidar (2023, 2025)", color=ROJO, fontsize=9.5, ha="right")
b.text(2.0, 31.5, "otros 26 taludes", color=TINTA_2, fontsize=9.5)
b.set(xlabel="t (años desde marzo de 2017)", ylabel="Cohesión c (kPa)", xlim=(-0.4, 8.4), ylim=(-1, 36))
b.set_title("(b) Solo Cortinas aporta información sobre el deterioro", fontsize=11.5)
fig.suptitle("Datos escasos: 48 ensayos en 27 taludes, y solo Cortinas se midió en dos fechas",
             x=0.01, ha="left", fontsize=13.5, fontweight="bold", y=0.99)
fig.tight_layout(rect=[0, 0, 1, 0.95])
guardar(fig, "fig1_datos")

# --------------------------------------------------------------------------
# Figura 2. Pooling parcial
# --------------------------------------------------------------------------
mu_a = R["parametros"]["mu_alpha"]["media"]
p = pooling.sort_values("alpha_media").reset_index(drop=True)
fig, ax = plt.subplots(figsize=(9, 8.6))
ax.axvspan(R["parametros"]["mu_alpha"]["hdi_inf"], R["parametros"]["mu_alpha"]["hdi_sup"], color="#eef3fb", zorder=0)
ax.axvline(mu_a, color=AZUL, lw=1, ls="--", zorder=1)
for y, f in p.iterrows():
    ax.plot([f["alpha_hdi_inf"], f["alpha_hdi_sup"]], [y, y], color=AZUL, lw=1.6, alpha=0.45, solid_capstyle="round")
    ax.annotate("", xy=(f["alpha_media"], y), xytext=(f["solo_sus_datos"], y),
                arrowprops=dict(arrowstyle="-|>", color=GRIS, lw=1, shrinkA=4, shrinkB=5, mutation_scale=9))
    ax.plot(f["solo_sus_datos"], y, "o", ms=7, mfc="white", mec=GRIS, mew=1.6, zorder=3)
    ax.plot(f["alpha_media"], y, "o", ms=8, color=AZUL if f["n"] == 1 else "#1c4f8f", mec="white", mew=1, zorder=4)
etiquetas = [f"{s if s.startswith('Variante') else s.split(',')[0]}  (n={n})" for s, n in zip(p["Sitio"], p["n"])]
ax.set_yticks(range(len(p)), etiquetas, fontsize=8.3)
ax.grid(axis="y", visible=False)
ax.set_xlabel("Cohesión inicial del talud, α (kPa)")
ax.set_ylim(-0.8, len(p) + 1.2)
ax.text(mu_a + 0.4, len(p) + 0.4, f"Media regional μα = {num(mu_a)} kPa", color=AZUL, fontsize=9.5)
jal = R["pooling_n1"]["jalon_mediano"]
ax.set_title(f"El modelo acerca a la media regional a los taludes con un solo ensayo\n"
             f"(n = 1: recorren ~{jal:.0%} del camino; con 11 ensayos casi no se mueven)", fontsize=12)
ax.text(0.0, -0.07, "Círculo hueco: cohesión inicial que daría el talud por sí solo (sin pooling). Punto azul (claro: n = 1; oscuro: n ≥ 2)\n"
        "y barra: media posterior del modelo jerárquico e intervalo de credibilidad del 94 %.\nFranja azul: intervalo del 94 % de la media regional.",
        transform=ax.transAxes, fontsize=8.5, color=TINTA_2, va="top")
guardar(fig, "fig2_pooling_parcial")

# --------------------------------------------------------------------------
# Figura 3. Comprobación predictiva posterior
# --------------------------------------------------------------------------
c_rep = idata.posterior_predictive["c_obs"].stack(m=("chain", "draw")).values.T
c_obs = train["Cohesion"].to_numpy()
rng = np.random.default_rng(42)
fig, axs = plt.subplots(1, 3, figsize=(13, 4))
bins = np.arange(0, 46, 2.5)
for r in rng.choice(len(c_rep), 80, replace=False):
    h, _ = np.histogram(c_rep[r], bins=bins, density=True)
    axs[0].step(bins[:-1], h, where="post", color=AZUL, alpha=0.06, lw=0.8)
h, _ = np.histogram(c_obs, bins=bins, density=True)
axs[0].step(bins[:-1], h, where="post", color=TINTA, lw=2.2)
axs[0].text(22, 0.075, "datos reales", color=TINTA, fontsize=10)
axs[0].text(30, 0.03, "80 réplicas\ndel modelo", color=AZUL, fontsize=10)
axs[0].axvline(0, color=TINTA_2, lw=1)
axs[0].set(xlabel="Cohesión c (kPa)", ylabel="Densidad", xlim=(-1, 45))
axs[0].set_title("(a) Forma y piso en cero", fontsize=11.5)

for ax, fila, f, xl in [
    (axs[1], ppc.iloc[1], lambda x: x.std(axis=1, ddof=1), "Desviación estándar de las 48 cohesiones (kPa)"),
    (axs[2], ppc.iloc[5], lambda x: x[:, 0] - x[:, 1], "Caída de Cortinas 2017 → 2021 (kPa)"),
]:
    v = f(c_rep)
    ax.hist(v, bins=50, color=AZUL_CLARO, edgecolor="white", linewidth=0.3)
    ax.axvline(fila["observado"], color=TINTA, lw=2)
    ax.text(fila["observado"], ax.get_ylim()[1] * 0.93, f"  real: {num(fila['observado'])}\n  p = {num(fila['p_bayes'], 2)}",
            fontsize=10, va="top")
    ax.set(xlabel=xl, yticks=[])
    ax.grid(axis="y", visible=False)
axs[1].set_title("(b) Dispersión entre ensayos", fontsize=11.5)
axs[2].set_title("(c) Caída observada en Cortinas", fontsize=11.5)
fig.suptitle("Las réplicas del modelo reproducen la dispersión, el piso en cero y la caída de Cortinas",
             x=0.01, ha="left", fontsize=13.5, fontweight="bold", y=0.99)
fig.tight_layout(rect=[0, 0, 1, 0.94])
guardar(fig, "fig3_ppc")

# --------------------------------------------------------------------------
# Figura 7. Cortinas: pronóstico post-falla frente a lo medido
# --------------------------------------------------------------------------
det = ing_c_det
fig, ax = plt.subplots(figsize=(10, 5.6))
# El borde de la banda predictiva sale con ruido de Monte Carlo: lo suavizamos solo para dibujarlo
suave = lambda col: proy[col].rolling(9, center=True, min_periods=1).mean()
ax.fill_between(proy["t"], suave("c_pred_inf"), suave("c_pred_sup"), color=AZUL_CLARO, alpha=0.45, lw=0)
ax.fill_between(proy["t"], proy["c_cred_inf"], proy["c_cred_sup"], color=AZUL, alpha=0.30, lw=0)
ax.plot(proy["t"], proy["c_media"], color=AZUL, lw=2.5)
ax.axvline(T_FALLA, color=TINTA_2, lw=1, ls=":")
ax.text(T_FALLA + 0.06, 33.3, "falla\n(2021)", fontsize=9.5, color=TINTA_2, va="top")
ax.hlines(12.1, T_FALLA, 8, color=GRIS, lw=1.8, ls="--")
ax.text(8.05, 12.1, "Ingenuo: persistencia\n12.1 kPa", color=GRIS, fontsize=9.5, va="center")
ax.hlines(R["datos"]["mediana_train"], 0, 8, color=GRIS, lw=1.2, ls=":")
ax.text(8.05, R["datos"]["mediana_train"] + 1.6, f"Ingenuo: mediana\n{num(R['datos']['mediana_train'])} kPa",
        color=GRIS, fontsize=9.5, va="center")
ax.plot(cort["t"], cort["Cohesion"], "o", color=TINTA, ms=9, mec="white", mew=1.5, zorder=5)
ax.plot(valid["t"], valid["Cohesion"], "D", color=ROJO, ms=10, mec="white", mew=1.5, zorder=6)
for _, f in det.iterrows():
    ax.annotate(f"{num(f['c_real'])} kPa medidos\nmodelo: {num(f['Modelo bayesiano jerárquico'])} kPa",
                xy=(f["t"], f["c_real"]), xytext=(f["t"] - 0.15, f["c_real"] + 5.5), ha="center",
                fontsize=9.5, color=ROJO, arrowprops=dict(arrowstyle="-", color=ROJO, lw=0.8))
ax.text(0.15, 19.6, "entrenamiento (2017, 2021)", fontsize=9.5, color=TINTA)
ax.text(1.0, 27.5, "Intervalo predictivo 94 %\n(un ensayo nuevo)", color="#4a7fc1", fontsize=9.5)
ax.text(0.9, 7.6, "Intervalo de credibilidad 94 %\nde la recta media", color=AZUL, fontsize=9.5)
ax.set(xlabel="t (años desde marzo de 2017)", ylabel="Cohesión c (kPa)", xlim=(-0.15, 8), ylim=(0, 34))
mae_b = ing_c.loc[0, "MAE_kPa"]
ax.set_title(f"Después de la falla, Cortinas recuperó cohesión que la recta de deterioro no anticipa\n"
             f"(error medio del modelo: {num(mae_b)} kPa en 2023 y 2025)", fontsize=12.5)
guardar(fig, "fig7_cortinas_pronostico")

# --------------------------------------------------------------------------
# Figura 8. Factor de seguridad en los dos escenarios de fricción
# --------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 5))
for nombre, color, phi in [("pico", AZUL, 30.8), ("residual", NARANJA, 21.3)]:
    ax.fill_between(proy["t"], proy[f"fos_{nombre}_inf"], proy[f"fos_{nombre}_sup"], color=color, alpha=0.25, lw=0)
    ax.plot(proy["t"], proy[f"fos_{nombre}"], color=color, lw=2.5)
ax.axhline(1, color=TINTA, lw=1.2, ls="--")
ax.axvline(T_FALLA, color=TINTA_2, lw=1, ls=":")
k0, k8 = clave.iloc[0], clave.iloc[-1]
ax.text(8.08, k8["fos_pico"], f"Fricción pico 30.8°\nFoS {num(k0['fos_pico'], 2)} → {num(k8['fos_pico'], 2)}\nP(FoS ≤ 1) = {k8['p_falla_pico']:.0%}",
        color=AZUL, fontsize=9.5, va="center")
ax.text(8.08, k8["fos_residual"], f"Fricción residual 21.3°\nFoS {num(k0['fos_residual'], 2)} → {num(k8['fos_residual'], 2)}\nP(FoS ≤ 1) = {k8['p_falla_residual']:.0%}",
        color=NARANJA, fontsize=9.5, va="center")
ax.text(0.1, 1.012, "FoS = 1 (equilibrio límite)", fontsize=9.5, color=TINTA, va="bottom")
ax.text(T_FALLA + 0.06, 1.3, "falla (2021)", fontsize=9.5, color=TINTA_2)
ax.set(xlabel="t (años desde marzo de 2017)", ylabel="Factor de seguridad (–)", xlim=(0, 8), ylim=(0.6, 1.35))
caida = k0["fos_pico"] - k8["fos_pico"]
ax.set_title(f"La fricción decide si Cortinas falla; la pérdida de cohesión solo resta ~{num(caida, 2)} de FoS en 8 años",
             fontsize=12.5)
ax.text(0.0, -0.15, "Bandas: intervalo de credibilidad del 94 % (solo incertidumbre de la cohesión). Talud infinito "
        "seco, β = 30°, z = 13 m, γ = 19.01 kN/m³.", transform=ax.transAxes, fontsize=8.5, color=TINTA_2)
guardar(fig, "fig8_fos_escenarios")

# --------------------------------------------------------------------------
# Figura 4. Sensibilidad al prior de delta_region
# --------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 3.9))
nombres = ["Concentrado\nLogN(ln 1, 0.35)", "Base\nLogN(ln 1, 0.75)", "Difuso\nLogN(ln 1, 1.50)"]
orden = [1, 0, 2]   # filas de sens: 0 base, 1 concentrado, 2 difuso
for y, i in enumerate(orden):
    f = sens.iloc[i]
    ax.plot([f["prior_q03"], min(f["prior_q97"], 6)], [y + 0.18] * 2, color=GRIS, lw=3, alpha=0.6, solid_capstyle="round")
    ax.plot([f["delta_inf"], f["delta_sup"]], [y - 0.12] * 2, color=AZUL, lw=5, solid_capstyle="round")
    ax.plot(f["delta_media"], y - 0.12, "o", color="white", mec=AZUL, mew=2.5, ms=10)
    ax.text(f["delta_sup"] + 0.1, y - 0.12, f"{num(f['delta_media'], 2)} kPa/año", va="center", fontsize=10, color=AZUL)
    if f["prior_q97"] > 6:
        ax.text(6.02, y + 0.18, "→ 16.8", va="center", fontsize=9, color=GRIS)
pendiente = (cort["Cohesion"].iloc[0] - cort["Cohesion"].iloc[1]) / T_FALLA
ax.axvline(pendiente, color=ROJO, lw=1, ls="--")
ax.text(pendiente + 0.05, 2.55, f"caída bruta de Cortinas: {num(pendiente, 2)} kPa/año", color=ROJO, fontsize=9.5)
ax.set_yticks(range(3), nombres, fontsize=10)
ax.set(xlabel="Tasa regional de pérdida de cohesión, δ región (kPa/año)", xlim=(0, 6.6), ylim=(-0.6, 2.85))
ax.grid(axis="y", visible=False)
ax.text(0.05, 2.32, "gris: prior (94 %)   azul: posterior (HDI 94 %)", fontsize=9, color=TINTA_2)
rango = sens["delta_media"]
ax.set_title(f"La tasa estimada cambia poco con el prior ({num(rango.min(), 2)} a {num(rango.max(), 2)} kPa/año) "
             f"y las conclusiones sobre el FoS no cambian", fontsize=12)
guardar(fig, "fig4_sensibilidad_prior")

# --------------------------------------------------------------------------
# Figura 5. Comparación con métodos ingenuos (MAE en kPa)
# --------------------------------------------------------------------------
fig, (a, b) = plt.subplots(1, 2, figsize=(12.5, 4.2))


def barras(ax, tabla, titulo):
    t = tabla.iloc[::-1].reset_index(drop=True)
    colores = [AZUL if m.startswith("Modelo") else GRIS for m in t["metodo"]]
    ax.barh(range(len(t)), t["MAE_kPa"], color=colores, height=0.6)
    for y, v in enumerate(t["MAE_kPa"]):
        ax.text(v + 0.12, y, f"{num(v, 2)} kPa", va="center", fontsize=10)
    ax.set_yticks(range(len(t)), [m.replace(" (", "\n(") for m in t["metodo"]], fontsize=9.5)
    ax.set(xlabel="Error absoluto medio (kPa)", xlim=(0, t["MAE_kPa"].max() * 1.3))
    ax.grid(axis="y", visible=False)
    ax.set_title(titulo, fontsize=11.5)


barras(a, ing_c, "(a) Cortinas después de la falla: 2 ensayos (2023, 2025)")
barras(b, ing_r, f"(b) Ensayo nuevo en talud conocido: {int(ing_r.loc[0, 'n_casos'])} ensayos, 9 taludes")
fig.suptitle("El modelo pierde contra la persistencia en la recuperación post-falla, pero gana a la mediana global entre taludes",
             x=0.01, ha="left", fontsize=13, fontweight="bold", y=0.99)
fig.tight_layout(rect=[0, 0, 1, 0.95])
guardar(fig, "fig5_comparacion_ingenuo")

# --------------------------------------------------------------------------
# Figura 6. Predicho frente a observado en la validación cruzada entre taludes
# --------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 6))
ax.plot([0, 36], [0, 36], color=TINTA_2, lw=1, ls="--")
ax.vlines(cv["c_real"], cv["bayes_inf"], cv["bayes_sup"], color=AZUL, alpha=0.25, lw=2)
ax.scatter(cv["c_real"], cv["mediana_global"], marker="_", s=120, color=GRIS, lw=2, zorder=3)
ax.scatter(cv["c_real"], cv["bayes"], s=40, color=AZUL, edgecolor="white", zorder=4)
ax.text(23, 23.6, "predicción perfecta", rotation=45, rotation_mode="anchor", ha="center", va="bottom",
        fontsize=9, color=TINTA_2)
ax.text(20.8, cv["mediana_global"].iloc[0] - 1.6, "— ingenuo: mediana global", color=GRIS, fontsize=9.5)
ax.text(1, 31, "modelo jerárquico\n(punto: mediana; barra: intervalo predictivo 94 %)", color=AZUL, fontsize=9.5)
ax.set(xlabel="Cohesión medida en el ensayo reservado (kPa)", ylabel="Cohesión predicha (kPa)", xlim=(0, 36), ylim=(-1, 36))
cob = ing_r.loc[0, "cobertura"]
ax.set_title(f"En taludes ya conocidos, el modelo se acerca al valor de cada talud;\nla mediana global predice lo mismo para todos ({cob} dentro del 94 %)",
             fontsize=12)
guardar(fig, "fig6_predicho_vs_observado")

# --------------------------------------------------------------------------
# Anexo. Trazas de las cadenas
# --------------------------------------------------------------------------
axes = az.plot_trace(idata, var_names=["mu_alpha", "sigma_alpha", "delta_region", "sigma_obs"],
                     compact=True, figsize=(11, 8))
fig = axes.ravel()[0].figure
fig.suptitle("Anexo: las 4 cadenas se mezclan bien (R-hat ≤ 1.01; 0 divergencias)", x=0.01, ha="left",
             fontsize=13, fontweight="bold")
fig.tight_layout()
guardar(fig, "anexo_trazas")
print("Listo.")
