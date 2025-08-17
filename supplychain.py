# concise_supplychain_app.py
import os, io, logging, warnings
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import pandas as pd, numpy as np
from fastapi import FastAPI, Body, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from dataclasses import dataclass
from sklearn.linear_model import LinearRegression
from scipy import stats

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()

@dataclass
class Settings:
    SUPABASE_URL: str = os.getenv("SUPABASE_URL", "")
    SUPABASE_KEY: str = os.getenv("SUPABASE_KEY", "")
    API_SECRET: str = os.getenv("API_SECRET", "changeme")

settings = Settings()

# ---------- Database Wrapper ----------
class DB:
    def __init__(self):
        try:
            from supabase import create_client
            self.client = create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY) if settings.SUPABASE_URL else None
        except:
            self.client = None
        self.mem: Dict[str, List[Dict[str, Any]]] = {t: [] for t in [
            "suppliers","skus","supplier_shipments","sales_orders","inventory_snapshots","kpi_daily","forecasts"
        ]}

    def upsert(self, table: str, rows: List[Dict[str, Any]]) -> bool:
        if not rows: return True
        try:
            if self.client:
                self.client.table(table).upsert(rows).execute()
            else:
                self.mem[table].extend(rows)
            return True
        except Exception as e:
            logger.error(f"DB upsert error {table}: {e}")
            return False

    def fetch_all(self, table: str) -> List[Dict[str, Any]]:
        try:
            return (self.client.table(table).select("*").execute().data if self.client else self.mem.get(table, [])) or []
        except Exception as e:
            logger.error(f"DB fetch error {table}: {e}")
            return []

    def clear(self, table: str):
        self.mem[table] = []

db = DB()

# ---------- Sample Data ----------
def sample_data():
    orders_csv = """order_id,sku,quantity,order_date
ORD-0001,SKU-1,120,2025-07-02
ORD-0002,SKU-1,140,2025-07-09
ORD-0003,SKU-2,80,2025-07-03
ORD-0004,SKU-2,75,2025-07-11
ORD-0005,SKU-1,95,2025-07-16
ORD-0006,SKU-3,60,2025-07-18
ORD-0007,SKU-1,110,2025-07-23
ORD-0008,SKU-2,85,2025-07-25
ORD-0009,SKU-1,130,2025-07-30
ORD-0010,SKU-3,45,2025-08-01
"""
    return pd.read_csv(io.StringIO(orders_csv))

def ensure_seed():
    if not db.fetch_all("sales_orders"):
        logger.info("Seeding with sample sales orders")
        db.upsert("sales_orders", sample_data().to_dict(orient="records"))

# ---------- Forecasting Engine ----------
class ForecastEngine:
    def sma(self, df, sku, horizon=14, season=7):
        df = df[df["sku"] == sku]
        if df.empty: return self._simple(sku, 0, horizon)
        daily = df.groupby("order_date")["quantity"].sum().asfreq("D", fill_value=0)
        if len(daily) < 2: return self._simple(sku, daily.mean(), horizon)
        sma = daily.rolling(season, min_periods=1).mean()
        slope = 0
        if len(sma) >= 3:
            x = np.arange(len(sma)); y = sma.values
            slope, _, _, p, _ = stats.linregress(x, y)
            slope = slope if p < 0.1 else 0
        last = daily.index.max()
        rows = []
        for h in range(1, horizon + 1):
            f = max(0.0, float(sma.tail(season).mean() + slope * h))
            rows.append({"sku": sku, "forecast_date": last + timedelta(days=h),
                         "horizon_days": h, "forecast_qty": f, "method": "SMA"})
        return pd.DataFrame(rows)

    def linear(self, df, sku, horizon=14):
        df = df[df["sku"] == sku]
        if df.empty: return self._simple(sku, 0, horizon)
        daily = df.groupby("order_date")["quantity"].sum().asfreq("D", fill_value=0)
        if len(daily) < 7: return self._simple(sku, daily.mean(), horizon)
        X = np.arange(len(daily)).reshape(-1, 1); y = daily.values
        m = LinearRegression().fit(X, y); last = daily.index.max()
        return pd.DataFrame([{"sku": sku, "forecast_date": last + timedelta(days=h),
                              "horizon_days": h, "forecast_qty": max(0, float(m.predict([[len(daily) + h - 1]])[0])),
                              "method": "Linear"} for h in range(1, horizon + 1)])

    def ensemble(self, df, sku, horizon=14):
        sma, lin = self.sma(df, sku, horizon), self.linear(df, sku, horizon)
        if sma.empty and lin.empty: return self._simple(sku, 0, horizon)
        rows = []
        for h in range(1, horizon + 1):
            s = sma.loc[sma.horizon_days == h, "forecast_qty"].mean() if not sma.empty else 0
            l = lin.loc[lin.horizon_days == h, "forecast_qty"].mean() if not lin.empty else 0
            rows.append({"sku": sku, "forecast_date": df["order_date"].max() + timedelta(days=h),
                         "horizon_days": h, "forecast_qty": 0.6 * s + 0.4 * l, "method": "Ensemble"})
        return pd.DataFrame(rows)

    def _simple(self, sku, avg, horizon):
        return pd.DataFrame([{"sku": sku, "forecast_date": datetime.now().date() + timedelta(days=h),
                              "horizon_days": h, "forecast_qty": avg, "method": "Simple"} for h in range(1, horizon + 1)])

forecast_engine = ForecastEngine()

# ---------- FastAPI ----------
app = FastAPI(title="SupplyChain API", version="2.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])

@app.on_event("startup")
def startup_event():
    ensure_seed()

@app.get("/")
def root():
    return {"msg": "SupplyChain API v2.0", "endpoints": ["/ingest", "/compute", "/retrain", "/kpis", "/forecast"]}

@app.post("/ingest")
def ingest(payload: Dict[str, Any] = Body(...), x_api_secret: Optional[str] = Header(None)):
    if settings.API_SECRET not in ["changeme", x_api_secret]:
        raise HTTPException(401, "Unauthorized")
    table = payload.get("table")
    rows = payload.get("rows") or pd.read_csv(io.StringIO(payload.get("csv", ""))).to_dict("records")
    if not table or not db.upsert(table, rows):
        raise HTTPException(400, "Invalid ingest")
    return {"status": "ok", "rows": len(rows)}

@app.post("/retrain")
def retrain(method: str = Query("ensemble"), horizon: int = Query(14)):
    df = pd.DataFrame(db.fetch_all("sales_orders"))
    if df.empty: return {"status": "warn", "msg": "No data"}
    df["order_date"] = pd.to_datetime(df["order_date"])
    frames = [getattr(forecast_engine, method)(df, sku, horizon) for sku in df["sku"].unique()]
    forecasts = pd.concat(frames, ignore_index=True)
    db.clear("forecasts"); db.upsert("forecasts", forecasts.to_dict("records"))
    return {"status": "ok", "rows": len(forecasts)}

@app.get("/forecast/{sku}")
def get_forecast(sku: str, horizon: int = 14):
    # Always ensure data is present
    if not db.fetch_all("sales_orders"):
        logger.info("Seeding inside forecast endpoint")
        db.upsert("sales_orders", sample_data().to_dict(orient="records"))

    df = pd.DataFrame(db.fetch_all("sales_orders"))
    if df.empty:
        return {"status": "warn", "msg": "No sales data"}
    
    df["order_date"] = pd.to_datetime(df["order_date"])
    f = forecast_engine.ensemble(df, sku, horizon)
    return f.to_dict("records")
