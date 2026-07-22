from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import uvicorn
from pydantic import BaseModel
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

app = FastAPI(title="Store Sales Evaluation")
templates = Jinja2Templates(directory="templates")

# Real Kaggle "Store Sales - Time Series Forecasting" (Favorita, Ecuador) data,
# stored one directory above this app (repo_root/store_sales_data/).
DATA_DIR = Path(__file__).resolve().parent.parent / "store_sales_data"

FEATURE_COLUMNS = [
    "store_nbr",
    "family_encoded",
    "onpromotion",
    "city_encoded",
    "state_encoded",
    "type_encoded",
    "cluster",
    "dcoilwtico",
    "day_of_week",
    "month",
    "is_weekend",
    "day_of_month",
    "is_holiday",
]

model = None
le_family = LabelEncoder()
le_city = LabelEncoder()
le_state = LabelEncoder()
le_type = LabelEncoder()

FAMILIES: list[str] = []
CITIES: list[str] = []
STATES: list[str] = []
STORE_TYPES: list[str] = []

MODEL_R2 = None
MODEL_MAE = None
USED_LOG_TARGET = False
FEATURE_IMPORTANCE: list[dict] = []
N_TRAIN_ROWS = 0


def _safe_encode(le: LabelEncoder, value: str):
    """Encode a category, falling back to the first known class if unseen."""
    if value in le.classes_:
        return int(le.transform([value])[0])
    return int(le.transform([le.classes_[0]])[0])


@app.on_event("startup")
def train():
    global model, le_family, le_city, le_state, le_type
    global FAMILIES, CITIES, STATES, STORE_TYPES
    global MODEL_R2, MODEL_MAE, USED_LOG_TARGET, FEATURE_IMPORTANCE, N_TRAIN_ROWS

    # --- Load raw Kaggle CSVs ---
    df = pd.read_csv(DATA_DIR / "train_sample.csv", parse_dates=["date"])
    stores = pd.read_csv(DATA_DIR / "stores.csv")
    oil = pd.read_csv(DATA_DIR / "oil.csv", parse_dates=["date"])
    holidays = pd.read_csv(DATA_DIR / "holidays_events.csv", parse_dates=["date"])

    # --- Merge store metadata (city, state, type, cluster) ---
    df = df.merge(stores, on="store_nbr", how="left")

    # --- Merge oil price, fill missing days with ffill/bfill ---
    oil = oil.sort_values("date")
    oil["dcoilwtico"] = oil["dcoilwtico"].ffill().bfill()
    df = df.merge(oil, on="date", how="left")
    df["dcoilwtico"] = df["dcoilwtico"].ffill().bfill()

    # --- Derive is_holiday: date present in holidays_events AND not transferred ---
    holiday_dates = set(holidays.loc[~holidays["transferred"], "date"].dt.normalize())
    df["is_holiday"] = df["date"].dt.normalize().isin(holiday_dates)

    # --- Date feature engineering ---
    df["day_of_week"] = df["date"].dt.dayofweek  # 0=Mon ... 6=Sun
    df["month"] = df["date"].dt.month
    df["is_weekend"] = df["day_of_week"] >= 5
    df["day_of_month"] = df["date"].dt.day

    # --- Categorical encoding ---
    df["family_encoded"] = le_family.fit_transform(df["family"])
    df["city_encoded"] = le_city.fit_transform(df["city"])
    df["state_encoded"] = le_state.fit_transform(df["state"])
    df["type_encoded"] = le_type.fit_transform(df["type"])

    FAMILIES = sorted(df["family"].unique().tolist())
    CITIES = sorted(df["city"].unique().tolist())
    STATES = sorted(df["state"].unique().tolist())
    STORE_TYPES = sorted(stores["type"].unique().tolist())

    N_TRAIN_ROWS = len(df)

    X = df[FEATURE_COLUMNS].astype(float)
    y = df["sales"].astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # --- Try both raw-target and log1p-target models, keep whichever wins on R2 ---
    # Sales is heavily right-skewed (many zero/low-value rows), so log1p often helps.
    raw_model = RandomForestRegressor(
        n_estimators=200, max_depth=None, n_jobs=-1, random_state=42
    )
    raw_model.fit(X_train, y_train)
    raw_pred = raw_model.predict(X_test)
    raw_r2 = r2_score(y_test, raw_pred)
    raw_mae = mean_absolute_error(y_test, raw_pred)

    log_model = RandomForestRegressor(
        n_estimators=200, max_depth=None, n_jobs=-1, random_state=42
    )
    log_model.fit(X_train, np.log1p(y_train))
    log_pred = np.expm1(log_model.predict(X_test))
    log_pred = np.clip(log_pred, 0, None)
    log_r2 = r2_score(y_test, log_pred)
    log_mae = mean_absolute_error(y_test, log_pred)

    if log_r2 > raw_r2:
        model = log_model
        USED_LOG_TARGET = True
        MODEL_R2 = log_r2
        MODEL_MAE = log_mae
    else:
        model = raw_model
        USED_LOG_TARGET = False
        MODEL_R2 = raw_r2
        MODEL_MAE = raw_mae

    importances = sorted(
        zip(FEATURE_COLUMNS, model.feature_importances_.tolist()),
        key=lambda pair: pair[1],
        reverse=True,
    )
    FEATURE_IMPORTANCE = [
        {"feature": name, "importance": round(score, 4)} for name, score in importances
    ]

    print(
        f"[train] rows={N_TRAIN_ROWS} log_target={USED_LOG_TARGET} "
        f"r2={MODEL_R2:.4f} mae={MODEL_MAE:.2f}"
    )


class PredictRequest(BaseModel):
    store_nbr: int
    family: str
    onpromotion: int
    city: str
    state: str
    store_type: str
    cluster: int
    oil_price: float
    day_of_week: int
    month: int
    is_weekend: bool
    day_of_month: int
    is_holiday: bool


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/options")
async def options():
    return {
        "families": FAMILIES,
        "cities": CITIES,
        "states": STATES,
        "store_types": STORE_TYPES,
    }


@app.post("/predict")
async def predict(data: PredictRequest):
    family_enc = _safe_encode(le_family, data.family)
    city_enc = _safe_encode(le_city, data.city)
    state_enc = _safe_encode(le_state, data.state)
    type_enc = _safe_encode(le_type, data.store_type)

    row = [
        data.store_nbr,
        family_enc,
        data.onpromotion,
        city_enc,
        state_enc,
        type_enc,
        data.cluster,
        data.oil_price,
        data.day_of_week,
        data.month,
        int(data.is_weekend),
        data.day_of_month,
        int(data.is_holiday),
    ]
    X = pd.DataFrame([row], columns=FEATURE_COLUMNS).astype(float)

    raw_pred = model.predict(X)[0]
    pred_sales = float(np.expm1(raw_pred)) if USED_LOG_TARGET else float(raw_pred)
    pred_sales = max(pred_sales, 0.0)

    return {
        "predicted_sales": round(pred_sales, 2),
        "feature_importance": FEATURE_IMPORTANCE,
        "model_r2": round(MODEL_R2, 4),
        "model_mae": round(MODEL_MAE, 2),
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5004)
