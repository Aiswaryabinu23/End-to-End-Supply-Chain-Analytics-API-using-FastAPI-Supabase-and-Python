from __future__ import annotations
from fastapi import FastAPI, Body, Header, Query, HTTPException
from typing import Optional, Any, Dict, List
from dataclasses import dataclass
from datetime import timedelta
import os
import io

import pandas as pd
import numpy as np
from dotenv import load_dotenv

# ---------- Settings ----------
load_dotenv()

@dataclass
class Settings:
    SUPABASE_URL: str = os.getenv("SUPABASE_URL", "")
    SUPABASE_KEY: str = os.getenv("SUPABASE_KEY", "")
    API_SECRET: str = os.getenv("API_SECRET", "changeme")

settings = Settings()

# ---------- Optional Supabase Client ----------
SupabaseClientFactory = None
try:
    from supabase import create_client  # type: ignore
    def _make_client():
        return create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)
    SupabaseClientFactory = _make_client if settings.SUPABASE_URL and settings.SUPABASE_KEY else None
except Exception:
    SupabaseClientFactory = None

# ---------- Minimal Persistence Layer ----------
class DB:
    """
    If Supabase creds provided, writes/reads to Supabase tables.
    Otherwise, stores everything in memory (Python dicts).
    """
    def __init__(self):
        self.client = SupabaseClientFactory() if SupabaseClientFactory else None
        # in-memory store mirrors the logical tables
        self.mem: Dict[str, List[Dict[str, Any]]] = {
            "suppliers": [],
            "skus": [],
            "supplier_shipments": [],
            "sales_orders": [],
            "inventory_snapshots": [],
            "kpi_daily": [],
            "forecasts": [],
        }

    def upsert(self, table: str, rows: List[Dict[str, Any]]):
        if not rows:
            return
        if self.client:
            # naive upsert: relies on PKs defined in your DB schema
            self.client.table(table).upsert(rows).execute()
        else:
            # in-memory: append then deduplicate
            self.mem.setdefault(table, [])
            self.mem[table].extend(rows)
            seen = set()
            deduped = []
            for r in self.mem[table]:
                key = tuple(sorted(r.items()))
                if key not in seen:
                    seen.add(key)
                    deduped.append(r)
            self.mem[table] = deduped

    def fetch_all(self, table: str) -> List[Dict[str, Any]]:
        if self.client:
            resp = self.client.table(table).select("*").execute()
            return resp.data or []
        return list(self.mem.get(table, []))

db = DB()

# ---------- Analytics: KPIs ----------
def compute_kpis(shipments_df: pd.DataFrame, orders_df: pd.DataFrame, inv_df: pd.DataFrame) -> pd.DataFrame:
    # Ensure date types
    if "shipment_date" in shipments_df.columns:
        shipments_df = shipments_df.copy()
        shipments_df["shipment_date"] = pd.to_datetime(shipments_df["shipment_date"])
    if "order_date" in orders_df.columns:
        orders_df = orders_df.copy()
        orders_df["order_date"] = pd.to_datetime(orders_df["order_date"])
    if "snapshot_date" in inv_df.columns:
        inv_df = inv_df.copy()
        inv_df["snapshot_date"] = pd.to_datetime(inv_df["snapshot_date"])

    # Daily demand per SKU
    if not orders_df.empty:
        daily = (
            orders_df.groupby(["sku", pd.Grouper(key="order_date", freq="D")])["quantity"]
            .sum()
            .reset_index()
            .rename(columns={"order_date": "kpi_date"})
        )
    else:
        daily = pd.DataFrame(columns=["sku", "kpi_date", "quantity"])

    # Rolling sums (7d, 28d) per SKU
    if not daily.empty:
        daily = daily.sort_values(["sku", "kpi_date"])
        kpi = []
        for sku, g in daily.groupby("sku"):
            g = g.set_index("kpi_date").asfreq("D", fill_value=0)
            g["demand_7d"] = g["quantity"].rolling(7, min_periods=1).sum()
            g["demand_28d"] = g["quantity"].rolling(28, min_periods=1).sum()
            g = g.reset_index()
            g["sku"] = sku
            kpi.append(g[["kpi_date", "sku", "demand_7d", "demand_28d"]])
        kpi_df = pd.concat(kpi, ignore_index=True) if kpi else pd.DataFrame(columns=["kpi_date","sku","demand_7d","demand_28d"])
    else:
        kpi_df = pd.DataFrame(columns=["kpi_date","sku","demand_7d","demand_28d"])

    # Stockout proxy & avg lead time
    stockout = pd.DataFrame(columns=["sku","stockout_flag"])
    if not inv_df.empty:
        inv_latest = inv_df.sort_values("snapshot_date").groupby("sku").tail(1)
        inv_latest["stockout_flag"] = inv_latest["on_hand"] <= 0
        stockout = inv_latest[["sku","stockout_flag"]]

    lead = pd.DataFrame(columns=["sku","avg_lead_time"])
    if not shipments_df.empty and "lead_time_days" in shipments_df.columns:
        lead = shipments_df.groupby("sku")["lead_time_days"].mean().reset_index().rename(columns={"lead_time_days":"avg_lead_time"})

    kpi_df = (
        kpi_df.merge(lead, on="sku", how="left")
              .merge(stockout, on="sku", how="left")
              .sort_values(["sku","kpi_date"])
              .reset_index(drop=True)
    )
    # Fill NA
    if "avg_lead_time" in kpi_df.columns:
        kpi_df["avg_lead_time"] = kpi_df["avg_lead_time"].fillna(0)
    if "stockout_flag" in kpi_df.columns:
        kpi_df["stockout_flag"] = kpi_df["stockout_flag"].fillna(False)
    return kpi_df

# ---------- Analytics: Simple Forecast ----------
def seasonal_moving_average(orders_df: pd.DataFrame, sku: str, season_days: int = 7, horizon: int = 14) -> pd.DataFrame:
    df = orders_df[orders_df["sku"] == sku].copy()
    if df.empty:
        return pd.DataFrame(columns=["sku","forecast_date","horizon_days","forecast_qty","method"])
    df["order_date"] = pd.to_datetime(df["order_date"])
    daily = df.groupby("order_date")["quantity"].sum().asfreq("D", fill_value=0)
    sma = daily.rolling(season_days, min_periods=1).mean().iloc[-1]
    last_date = daily.index.max()
    rows = []
    for h in range(1, horizon+1):
        fd = (last_date + pd.Timedelta(days=h)).date()
        rows.append({
            "sku": sku,
            "forecast_date": fd,
            "horizon_days": h,
            "forecast_qty": float(sma),
            "method": f"SMA_{season_days}"
        })
    return pd.DataFrame(rows)

def train_all(orders_df: pd.DataFrame, skus: Optional[List[str]] = None, season_days: int = 7, horizon: int = 14) -> pd.DataFrame:
    if skus is None:
        skus = sorted(orders_df["sku"].unique()) if "sku" in orders_df.columns and not orders_df.empty else []
    frames = []
    for s in skus:
        frames.append(seasonal_moving_average(orders_df, s, season_days, horizon))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["sku","forecast_date","horizon_days","forecast_qty","method"])

# ---------- Helpers: sample seed data ----------
def sample_data() -> Dict[str, pd.DataFrame]:
    shipments_csv = """shipment_id,supplier_id,sku,quantity,shipment_date,lead_time_days,cost_per_unit
SHP-0001,SUP-1,SKU-1,500,2025-07-01,9,5.5
SHP-0002,SUP-1,SKU-2,300,2025-07-05,12,4.2
SHP-0003,SUP-2,SKU-1,400,2025-07-10,10,5.7
"""
    orders_csv = """order_id,sku,quantity,order_date,price_per_unit
ORD-0001,SKU-1,120,2025-07-02,11.0
ORD-0002,SKU-1,140,2025-07-09,11.2
ORD-0003,SKU-2,80,2025-07-03,9.0
ORD-0004,SKU-2,75,2025-07-11,9.1
"""
    inv_csv = """snapshot_date,sku,on_hand,in_transit
2025-07-01,SKU-1,1000,400
2025-07-08,SKU-1,850,300
2025-07-01,SKU-2,600,200
2025-07-08,SKU-2,520,150
"""
    return {
        "supplier_shipments": pd.read_csv(io.StringIO(shipments_csv)),
        "sales_orders": pd.read_csv(io.StringIO(orders_csv)),
        "inventory_snapshots": pd.read_csv(io.StringIO(inv_csv)),
    }

def ensure_minimum_seed():
    # If store empty and running in memory, seed with sample data so endpoints work immediately.
    if db.client:
        return
    if not db.fetch_all("sales_orders"):
        data = sample_data()
        for table, df in data.items():
            db.upsert(table, df.to_dict(orient="records"))

# ---------- Security ----------
def auth_guard(secret_header: Optional[str]):
    expected = settings.API_SECRET or "changeme"
    if expected and secret_header != expected:
        raise HTTPException(status_code=401, detail="Unauthorized")

# ---------- FastAPI ----------
app = FastAPI(title="Supply Chain Analytics (Single-File)")

@app.on_event("startup")
def _startup():
    ensure_minimum_seed()

@app.post("/ingest")
def ingest(payload: Dict[str, Any] = Body(...), x_api_secret: Optional[str] = Header(None)):
    auth_guard(x_api_secret)
    table = payload.get("table")
    if not table:
        raise HTTPException(status_code=400, detail="Missing table")
    if "rows" in payload:
        rows = payload["rows"]
        if not isinstance(rows, list):
            raise HTTPException(status_code=400, detail="rows must be a list")
        db.upsert(table, rows)
    elif "csv" in payload:
        df = pd.read_csv(io.StringIO(payload["csv"]))
        db.upsert(table, df.to_dict(orient="records"))
    else:
        raise HTTPException(status_code=400, detail="Provide rows[] or csv")
    return {"status": "ok", "table": table, "rows": len(payload.get("rows", []))}

@app.post("/compute")
def compute(x_api_secret: Optional[str] = Header(None)):
    auth_guard(x_api_secret)
    shipments = pd.DataFrame(db.fetch_all("supplier_shipments"))
    orders = pd.DataFrame(db.fetch_all("sales_orders"))
    inv = pd.DataFrame(db.fetch_all("inventory_snapshots"))
    if orders.empty:
        seed = sample_data()
        shipments = seed["supplier_shipments"]
        orders = seed["sales_orders"]
        inv = seed["inventory_snapshots"]
    kpi = compute_kpis(shipments, orders, inv)
    db.upsert("kpi_daily", kpi.to_dict(orient="records"))
    return {"status": "ok", "rows": int(kpi.shape[0])}

@app.post("/retrain")
def retrain(x_api_secret: Optional[str] = Header(None)):
    auth_guard(x_api_secret)
    orders = pd.DataFrame(db.fetch_all("sales_orders"))
    if orders.empty:
        orders = sample_data()["sales_orders"]
    fc = train_all(orders, season_days=7, horizon=14)
    db.upsert("forecasts", fc.to_dict(orient="records"))
    return {"status": "ok", "rows": int(fc.shape[0])}

@app.get("/kpis")
def get_kpis() -> List[Dict[str, Any]]:
    return db.fetch_all("kpi_daily")

@app.get("/forecast")
def get_forecast(sku: str = Query(...)) -> List[Dict[str, Any]]:
    rows = db.fetch_all("forecasts")
    if rows:
        return [r for r in rows if r.get("sku") == sku]
    orders = pd.DataFrame(db.fetch_all("sales_orders"))
    if orders.empty:
        orders = sample_data()["sales_orders"]
    fc = train_all(orders, skus=[sku], season_days=7, horizon=14)
    return fc.to_dict(orient="records")
