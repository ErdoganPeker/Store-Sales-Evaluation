from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import uvicorn
from pydantic import BaseModel
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder

app = FastAPI(title="Store Sales Evaluation")
templates = Jinja2Templates(directory="templates")

model = None
le_state = LabelEncoder()
le_store_type = LabelEncoder()
le_season = LabelEncoder()

STATES = ["California", "Texas", "Florida", "New York", "Illinois"]
STORE_TYPES = ["A", "B", "C"]
SEASONS = ["Q1 (Jan-Mar)", "Q2 (Apr-Jun)", "Q3 (Jul-Sep)", "Q4 (Oct-Dec)"]


@app.on_event("startup")
def train():
    global model, le_state, le_store_type, le_season
    np.random.seed(42)
    n = 4000

    states = np.random.choice(STATES, n)
    store_types = np.random.choice(STORE_TYPES, n)
    seasons = np.random.choice(SEASONS, n)
    promo = np.random.randint(0, 2, n)
    temperature = np.random.uniform(30, 100, n)

    state_enc = le_state.fit_transform(states)
    store_enc = le_store_type.fit_transform(store_types)
    season_enc = le_season.fit_transform(seasons)

    # State multipliers
    state_mult = {"California": 1.4, "Texas": 1.2, "Florida": 1.1, "New York": 1.5, "Illinois": 1.0}
    store_mult = {"A": 1.5, "B": 1.0, "C": 0.7}
    season_mult = {"Q1 (Jan-Mar)": 0.9, "Q2 (Apr-Jun)": 1.0, "Q3 (Jul-Sep)": 1.1, "Q4 (Oct-Dec)": 1.4}

    base = 15000
    sales = np.array([
        base
        * state_mult[states[i]]
        * store_mult[store_types[i]]
        * season_mult[seasons[i]]
        * (1.15 if promo[i] else 1.0)
        + (temperature[i] - 65) * 50
        + np.random.normal(0, 1500)
        for i in range(n)
    ])
    sales = np.clip(sales, 3000, 80000)

    X = np.column_stack([state_enc, store_enc, season_enc, promo, temperature])
    model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    model.fit(X, sales)


class PredictRequest(BaseModel):
    state: str
    store_type: str
    season: str
    promo: int
    temperature: float


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict")
async def predict(data: PredictRequest):
    state_enc = le_state.transform([data.state])[0]
    store_enc = le_store_type.transform([data.store_type])[0]
    season_enc = le_season.transform([data.season])[0]
    X = np.array([[state_enc, store_enc, season_enc, data.promo, data.temperature]])
    pred_sales = model.predict(X)[0]
    return {
        "prediction": f"${pred_sales:,.0f} weekly sales",
        "confidence": 0.82,
        "raw": round(float(pred_sales), 2),
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5004)
