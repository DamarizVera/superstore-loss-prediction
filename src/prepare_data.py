#!/usr/bin/env python3
"""Limpia el Superstore crudo y genera una tabla plana para modelado.

Entrada : data/raw/superstore.csv
Salida  : data/processed/orders.csv
El CSV original trae pegadas, debajo de los pedidos, las hojas 'People' y 'Returns'
(filas sin fecha de pedido); se descartan aquí.
"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
raw = pd.read_csv(ROOT / "data/raw/superstore.csv", dtype={"Postal Code": "string"})
df = raw[raw["Order Date"].notna()].copy()

df = df.rename(columns={
    "Row ID": "row_id", "Order ID": "order_id", "Order Date": "order_date", "Ship Date": "ship_date",
    "Ship Mode": "ship_mode", "Customer ID": "customer_id", "Segment": "segment", "State": "state",
    "Region": "region", "Product ID": "product_id", "Category": "category", "Sub-Category": "sub_category",
    "Sales": "sales", "Quantity": "quantity", "Discount": "discount", "Profit": "profit"})
df["row_id"] = df["row_id"].astype(int)
df["order_date"] = pd.to_datetime(df["order_date"], format="%m/%d/%Y")
df["ship_date"] = pd.to_datetime(df["ship_date"], format="%m/%d/%Y")
df["quantity"] = df["quantity"].astype(int)

cols = ["row_id", "order_id", "order_date", "ship_date", "ship_mode", "customer_id", "segment", "state", "region",
        "product_id", "category", "sub_category", "sales", "quantity", "discount", "profit"]
df = df[cols].sort_values(["order_date", "row_id"]).reset_index(drop=True)
df.to_csv(ROOT / "data/processed/orders.csv", index=False)
print(f"{len(df):,} líneas · {df.order_date.min().date()} → {df.order_date.max().date()}")
print(f"líneas con pérdida: {(df.profit < 0).sum():,} ({100*(df.profit < 0).mean():.1f}%)")
