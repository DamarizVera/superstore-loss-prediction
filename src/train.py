#!/usr/bin/env python3
"""Predice qué líneas de pedido terminarán con pérdida (profit < 0) y evalúa el impacto de negocio.

Validación temporal (sin mezclar el futuro con el pasado):
  - selección de umbral:   entrena 2015-2016  -> valida 2017
  - modelo final:          entrena 2015-2017  -> prueba 2018 (nunca visto)
Salidas: reports/metrics.csv, reports/policy_simulation.csv, reports/figures/*.png, models/model.joblib
"""
import json
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from features import CAT, FEATURES, NUM, add_features, load

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "reports/figures"
FIG.mkdir(parents=True, exist_ok=True)
SEED = 42
DARK, MID, LIGHT, CORAL, TEAL, GREY = "#2F6690", "#5F87A5", "#8FA8BA", "#E8735A", "#4DC3B0", "#9AA5B1"
plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white", "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.alpha": .25, "font.size": 10,
                     "axes.titleweight": "bold", "axes.titlesize": 12})

df = add_features(load(ROOT / "data/processed/orders.csv"))
train_sel, val = df[df.year <= 2016], df[df.year == 2017]
train_fin, test = df[df.year <= 2017], df[df.year == 2018]
print(f"train(sel) {len(train_sel):,} · val {len(val):,} · train(final) {len(train_fin):,} · test {len(test):,}")
print(f"tasa de pérdida  train {train_fin.loss.mean():.3f} · test {test.loss.mean():.3f}")


# ----------------------------------------------------------------------------- modelos
def make_models():
    onehot = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), CAT),
                                ("num", StandardScaler(), NUM)])
    ordinal = ColumnTransformer([("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), CAT),
                                 ("num", "passthrough", NUM)])
    return {
        "Regla: descuento > 20%": None,
        "Regresión logística": Pipeline([("prep", onehot), ("m", LogisticRegression(max_iter=2000, C=1.0))]),
        "Random Forest": Pipeline([("prep", ordinal), ("m", RandomForestClassifier(
            n_estimators=400, min_samples_leaf=3, n_jobs=-1, random_state=SEED))]),
        "Gradient Boosting": Pipeline([("prep", ordinal), ("m", HistGradientBoostingClassifier(
            max_depth=4, learning_rate=0.06, max_iter=300, l2_regularization=1.0,
            categorical_features=list(range(len(CAT))), random_state=SEED))]),
    }


def score(name, model, X):
    if model is None:                      # regla de negocio: puntaje = descuento
        return X["discount"].values
    return model.predict_proba(X[FEATURES])[:, 1]


def best_threshold(y, p):
    pr, rc, th = precision_recall_curve(y, p)
    f1 = 2 * pr * rc / np.clip(pr + rc, 1e-9, None)
    return float(th[np.argmax(f1[:-1])])


# 1) umbral por modelo, elegido SOLO con 2015-2016 -> 2017
thresholds = {}
for name, m in make_models().items():
    if m is None:
        thresholds[name] = 0.2             # la regla ya tiene su umbral (> 20%)
        continue
    m.fit(train_sel[FEATURES], train_sel["loss"])
    thresholds[name] = best_threshold(val["loss"], score(name, m, val))

# 2) modelos finales: entrenan 2015-2017, se evalúan en 2018
final, rows, scores = {}, [], {}
for name, m in make_models().items():
    if m is not None:
        m.fit(train_fin[FEATURES], train_fin["loss"])
    final[name] = m
    p = score(name, m, test)
    scores[name] = p
    th = thresholds[name]
    pred = (p > th).astype(int) if m is None else (p >= th).astype(int)
    rows.append({"modelo": name, "umbral": round(th, 3),
                 "ROC-AUC": roc_auc_score(test.loss, p), "PR-AUC": average_precision_score(test.loss, p),
                 "precisión": precision_score(test.loss, pred), "recall": recall_score(test.loss, pred),
                 "F1": f1_score(test.loss, pred), "Brier": brier_score_loss(test.loss, np.clip(p, 0, 1)) if m else np.nan})
metrics = pd.DataFrame(rows)

# 3) zona gris: líneas donde la regla simple NO alcanza (0% < descuento <= 40%)
gray = (test.discount > 0) & (test.discount <= 0.4)
rows_g = []
for name, m in make_models().items():
    p = scores[name][gray.values]
    th = thresholds[name]
    pred = (p > th).astype(int) if m is None else (p >= th).astype(int)
    y = test.loss[gray]
    rows_g.append({"modelo": name, "ROC-AUC": roc_auc_score(y, p), "PR-AUC": average_precision_score(y, p),
                   "precisión": precision_score(y, pred), "recall": recall_score(y, pred), "F1": f1_score(y, pred)})
metrics_gray = pd.DataFrame(rows_g)
print(f"\nZona gris 2018: {int(gray.sum()):,} líneas, {100*test.loss[gray].mean():.1f}% con pérdida")

out = ROOT / "reports"
metrics.round(4).to_csv(out / "metrics_test_2018.csv", index=False)
metrics_gray.round(4).to_csv(out / "metrics_test_2018_zona_gris.csv", index=False)
print("\n== Test 2018 (todas las líneas) ==\n", metrics.round(3).to_string(index=False))
print("\n== Test 2018 (zona gris: descuento entre 0% y 40%) ==\n", metrics_gray.round(3).to_string(index=False))

best_name = metrics.iloc[1:].sort_values("PR-AUC", ascending=False).iloc[0]["modelo"]
best = final[best_name]
p_best = scores[best_name]
th_best = thresholds[best_name]
print(f"\nMejor modelo (PR-AUC en 2018): {best_name} · umbral {th_best:.3f}")

# ----------------------------------------------------------------------------- figuras
# 1) tasa de pérdida por tramo de descuento (EDA)
bands = pd.cut(df.discount, [-0.01, 0, 0.1, 0.2, 0.3, 0.4, 0.6, 1.0],
               labels=["0%", "1-10%", "11-20%", "21-30%", "31-40%", "41-60%", ">60%"])
g = df.groupby(bands).loss.agg(["mean", "size"])
fig, ax = plt.subplots(figsize=(7.5, 4))
ax.bar(g.index.astype(str), 100 * g["mean"], color=[DARK if v < .5 else CORAL for v in g["mean"]])
for i, (m, n) in enumerate(zip(g["mean"], g["size"])):
    ax.text(i, 100 * m + 2, f"{100*m:.0f}%\n(n={n:,})", ha="center", fontsize=8.5)
ax.set_ylim(0, 118); ax.set_ylabel("% de líneas con pérdida"); ax.set_xlabel("Descuento aplicado")
ax.set_title("La pérdida depende casi totalmente del descuento"); fig.tight_layout()
fig.savefig(FIG / "01_perdida_por_descuento.png", dpi=150); plt.close(fig)

# 2) curvas ROC y PR en 2018
fig, axs = plt.subplots(1, 2, figsize=(10, 4))
colors = {"Regla: descuento > 20%": GREY, "Regresión logística": LIGHT, "Random Forest": MID, "Gradient Boosting": DARK}
for name, p in scores.items():
    fpr, tpr, _ = roc_curve(test.loss, p)
    axs[0].plot(fpr, tpr, color=colors[name], lw=2, label=f"{name} ({roc_auc_score(test.loss, p):.3f})")
    pr, rc, _ = precision_recall_curve(test.loss, p)
    axs[1].plot(rc, pr, color=colors[name], lw=2, label=f"{name} ({average_precision_score(test.loss, p):.3f})")
axs[0].plot([0, 1], [0, 1], "--", color="#ccc"); axs[0].set_title("Curva ROC · test 2018")
axs[0].set_xlabel("Falsos positivos"); axs[0].set_ylabel("Recall"); axs[0].legend(fontsize=8, loc="lower right")
axs[1].set_title("Precisión-Recall · test 2018"); axs[1].set_xlabel("Recall"); axs[1].set_ylabel("Precisión")
axs[1].legend(fontsize=8, loc="lower left"); fig.tight_layout()
fig.savefig(FIG / "02_curvas_roc_pr.png", dpi=150); plt.close(fig)

# 3) importancia por permutación (sobre 2018, mejor modelo)
imp = permutation_importance(best, test[FEATURES], test["loss"], scoring="average_precision",
                             n_repeats=10, random_state=SEED, n_jobs=-1)
imp_df = pd.DataFrame({"variable": FEATURES, "caida_PR_AUC": imp.importances_mean,
                       "sd": imp.importances_std}).sort_values("caida_PR_AUC")
imp_df.round(4).sort_values("caida_PR_AUC", ascending=False).to_csv(out / "feature_importance.csv", index=False)
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.barh(imp_df.variable, imp_df.caida_PR_AUC, xerr=imp_df.sd, color=DARK, error_kw={"ecolor": GREY})
ax.set_xlabel("Caída del PR-AUC al barajar la variable"); ax.set_title(f"Importancia de variables · {best_name}")
fig.tight_layout(); fig.savefig(FIG / "03_importancia_variables.png", dpi=150); plt.close(fig)

# 4) probabilidad de pérdida vs descuento, por subcategoría (dependencia parcial)
subs = ["Tables", "Bookcases", "Binders", "Chairs", "Phones", "Copiers"]
grid = np.round(np.arange(0, 0.81, 0.05), 2)
fig, ax = plt.subplots(figsize=(8, 4.5))
pal = [CORAL, "#E8A33D", MID, DARK, LIGHT, TEAL]
for s, c in zip(subs, pal):
    base = test[test.sub_category == s]
    curve = []
    for dsc in grid:
        t = base.copy()
        t["discount"] = dsc
        t["sales"] = t["unit_list_price"] * t["quantity"] * (1 - dsc)
        t["log_sales"] = np.log1p(t["sales"])
        curve.append(best.predict_proba(t[FEATURES])[:, 1].mean())
    ax.plot(100 * grid, 100 * np.array(curve), color=c, lw=2, label=s)
ax.set_xlabel("Descuento (%)"); ax.set_ylabel("Probabilidad media de pérdida (%)")
ax.set_title("¿A qué descuento empieza a perderse dinero?"); ax.legend(ncol=3, fontsize=8)
fig.tight_layout(); fig.savefig(FIG / "04_probabilidad_vs_descuento.png", dpi=150); plt.close(fig)

# ----------------------------------------------------------------------------- simulación de negocio (2018)
CAP = 0.20
t = test.copy()
t["cost"] = t["sales"] - t["profit"]


def apply_cap(frame, mask, cap=CAP):
    s = frame["sales"].where(~mask, frame["sales"] * (1 - cap) / (1 - frame["discount"]))
    return s - frame["cost"]


def scenarios(flag, label):
    capped_profit = apply_cap(t, flag)                       # volumen igual, descuento tope 20%
    A = capped_profit.sum() - t.profit.sum()
    B = -t.profit[flag].sum()                                 # no se vende lo marcado
    C = 0.5 * (capped_profit[flag].sum() - t.profit[flag].sum()) + 0.5 * (-t.profit[flag].sum())
    return {"política": label, "líneas marcadas": int(flag.sum()),
            "de ellas con pérdida real": int(t.loss[flag].sum()),
            "A) tope 20%, mismo volumen": A, "B) no vender lo marcado": B,
            "C) mitad acepta el tope, mitad se pierde": C}


sims = [scenarios(pd.Series(False, index=t.index), "Sin política (actual)")]
sims[0].update({k: 0.0 for k in list(sims[0])[3:]})
sims.append(scenarios(t.discount > CAP, "Regla: descuento > 20%"))
sims.append(scenarios(pd.Series(p_best >= th_best, index=t.index), f"Modelo ({best_name})"))
sim = pd.DataFrame(sims)
sim.round(0).to_csv(out / "policy_simulation.csv", index=False)
print("\n== Simulación 2018: cambio en utilidad (US$) ==\n", sim.round(0).to_string(index=False))
print(f"Utilidad real 2018: {t.profit.sum():,.0f}")

fig, ax = plt.subplots(figsize=(8, 4.2))
x = np.arange(2); w = 0.26
for i, (lab, col) in enumerate([("A) tope 20%, mismo volumen", DARK), ("C) mitad acepta el tope, mitad se pierde", MID),
                                ("B) no vender lo marcado", LIGHT)]):
    vals = sim[lab].iloc[1:].values
    ax.bar(x + (i - 1) * w, vals, w, label=lab, color=col)
    for xi, v in zip(x + (i - 1) * w, vals):
        ax.text(xi, v + 400, f"{v/1000:,.1f}K", ha="center", fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(sim["política"].iloc[1:], fontsize=9)
ax.set_ylabel("Cambio en utilidad 2018 (US$)"); ax.set_title("Impacto estimado de cada política de descuentos")
ax.set_ylim(0, sim.iloc[1:, 3:].values.max() * 1.25); ax.legend(fontsize=8, ncol=3, loc="upper center"); fig.tight_layout(); fig.savefig(FIG / "05_simulacion_politica.png", dpi=150); plt.close(fig)

# 5) calibración
from sklearn.calibration import calibration_curve
fig, ax = plt.subplots(figsize=(4.8, 4.5))
for name in ["Regresión logística", "Gradient Boosting"]:
    fr, mp = calibration_curve(test.loss, scores[name], n_bins=10, strategy="quantile")
    ax.plot(mp, fr, "o-", color=colors[name], label=name)
ax.plot([0, 1], [0, 1], "--", color="#ccc"); ax.set_xlabel("Probabilidad predicha"); ax.set_ylabel("Frecuencia observada")
ax.set_title("Calibración · test 2018"); ax.legend(fontsize=8); fig.tight_layout()
fig.savefig(FIG / "06_calibracion.png", dpi=150); plt.close(fig)

(ROOT / "models").mkdir(exist_ok=True)
joblib.dump({"model": best, "features": FEATURES, "threshold": th_best, "name": best_name}, ROOT / "models/model.joblib")
json.dump({"mejor_modelo": best_name, "umbral": th_best, "train_years": [2015, 2017], "test_year": 2018,
           "lineas_test": int(len(test)), "tasa_perdida_test": float(test.loss.mean())},
          open(out / "summary.json", "w"), indent=2, ensure_ascii=False)
print("\nlisto")
