"""Construcción de variables. Solo se usa información disponible AL APROBAR el pedido:
precio, descuento, cantidad, producto, cliente y destino. Nunca profit, margen ni ship_date."""
import numpy as np
import pandas as pd

CAT = ["category", "sub_category", "segment", "region", "state", "ship_mode"]
NUM = ["discount", "quantity", "sales", "log_sales", "unit_list_price", "month", "dow"]
FEATURES = CAT + NUM


def load(path):
    df = pd.read_csv(path, parse_dates=["order_date", "ship_date"])
    df["loss"] = (df["profit"] < 0).astype(int)
    df["year"] = df["order_date"].dt.year
    return df


def add_features(df):
    d = df.copy()
    d["log_sales"] = np.log1p(d["sales"])
    # Sales viene con el descuento aplicado: precio de lista unitario = sales / (1 - descuento) / cantidad
    d["unit_list_price"] = d["sales"] / (1 - d["discount"]) / d["quantity"]
    d["month"] = d["order_date"].dt.month
    d["dow"] = d["order_date"].dt.dayofweek
    return d
