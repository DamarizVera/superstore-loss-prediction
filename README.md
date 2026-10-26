# Superstore — Predicting Loss-Making Orders (Machine Learning)

[Leer en español](README.es.md)

A classifier that flags **which order lines will lose money before the discount is approved**, plus a simulation of what acting on it would be worth. It is the predictive follow-up to my [Superstore Power BI dashboard](https://github.com/damarizvera/superstore-powerbi-dashboard), where discounts above 20% turned out to be the main driver of losses.

![Probability of loss vs discount](reports/figures/04_probabilidad_vs_descuento.png)

## Results (held-out year 2018, 3,312 order lines, 18.7% loss-making)

| Model | ROC-AUC | PR-AUC | F1 | F1 in the "grey zone"* |
|---|---|---|---|---|
| Rule: discount > 20% (baseline) | 0.940 | 0.814 | 0.826 | 0.594 |
| Logistic regression | 0.985 | 0.947 | 0.842 | 0.663 |
| Random Forest | 0.988 | 0.958 | 0.868 | 0.735 |
| **Gradient Boosting (selected)** | **0.990** | **0.963** | **0.881** | **0.754** |

\*Lines with a discount between 0% and 40%, where a simple threshold is not enough (1,415 lines, 22% loss-making).

**Honest reading:** most of the signal is the discount itself, so a one-line rule is already a strong baseline. The model's added value is where the rule is blind: it recovers 70% of losses in the grey zone (rule: 44%) because the break-even discount depends on the sub-category (Tables, Bookcases and Chairs start losing money at 10–15%; Copiers and Phones tolerate ~25–40%).

## Business simulation (2018, explicit assumptions)

Flagged lines are re-priced with a 20% discount cap. Profit change versus what actually happened (real 2018 profit: US$93,439):

| Policy | Lines flagged | Truly loss-making | A) Cap, same volume | C) Half accept, half walk away | B) Don't sell flagged |
|---|---|---|---|---|---|
| Rule: discount > 20% | 455 | 444 | +US$71.5K | +US$59.6K | +US$47.6K |
| Gradient Boosting | 572 | 525 | +US$70.9K | +US$60.8K | +US$50.7K |

The model and the rule are **economically similar**; the model flags more lines and wins when customers walk away, while the rule is simpler to explain. Scenario A assumes customers still buy at the capped price, which is optimistic; B is the conservative floor. These are what-if estimates from historical data, not causal effects.

## Approach

- **Target:** `profit < 0` per order line. **Features known at approval time only:** discount, quantity, sales, implied list price, category, sub-category, segment, region, state, ship mode, month, weekday. Excluded `profit`, margin and shipping dates (leakage).
- **Temporal validation:** thresholds chosen training 2015–16 → validating 2017; final models trained 2015–17 and tested once on 2018. No random split, so the future never leaks into training.
- **Imbalance:** 18.7% positives; evaluated with PR-AUC, precision/recall and F1, not accuracy.
- **Interpretability:** permutation importance (discount ≫ sub-category ≫ list price), partial-dependence-style curves per sub-category, calibration plot.

| | |
|---|---|
| ![ROC and PR](reports/figures/02_curvas_roc_pr.png) | ![Importance](reports/figures/03_importancia_variables.png) |
| ![Simulation](reports/figures/05_simulacion_politica.png) | ![Calibration](reports/figures/06_calibracion.png) |

## Limitations

- One retailer, four years; the dataset is a public sample, so conclusions illustrate method, not a real company.
- Without product cost, "loss" is inferred from `profit`; the model cannot see why a line lost money, only how it correlates.
- Effects of changing discounts on demand are assumed, not estimated (see scenarios).
- Nearly all of the signal is the discount, so the model should complement, not replace, a discount policy.

## Reproduce

```bash
pip install -r requirements.txt
python src/prepare_data.py   # cleans data/raw/superstore.csv -> data/processed/orders.csv
python src/train.py          # trains, evaluates, simulates; writes reports/ and models/
```

```
data/raw · data/processed   source and cleaned data
src/                        prepare_data.py · features.py · train.py
reports/                    metrics, simulation, importance, figures/
models/model.joblib         selected model + threshold
```

Data: public "Sample - Superstore" dataset.
