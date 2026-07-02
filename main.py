"""
Rossmann Sales Forecast API (v1)
--------------------------------
Сервис прогноза дневных продаж магазина Rossmann на основе обученной
модели XGBoost (xgb_tuned.json).

Запуск:  uvicorn main:app --reload
Docs:    http://127.0.0.1:8000/docs
"""

from fastapi import FastAPI
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
from xgboost import XGBRegressor

# 1. Загрузка модели при старте сервиса ----------
MODEL_PATH = "xgb_tuned.json"
model = XGBRegressor()
model.load_model(MODEL_PATH)

# Порядок признаков должен совпадать с обучением ( x.columns).
FEATURE_ORDER = [
    "Store", "DayOfWeek", "Open", "Promo", "StateHoliday", "SchoolHoliday",
    "StoreType", "Assortment", "CompetitionDistance",
    "CompetitionOpenSinceMonth", "CompetitionOpenSinceYear",
    "Promo2", "Promo2SinceWeek", "Promo2SinceYear",
    "Year", "Month", "Day", "WeekOfYear", "IsPromoMonth",
    "lag_7", "lag_14", "lag_28", "rolling_mean_7", "rolling_mean_28",
]

# Кодирование букв в числа. Повторяет sklearn LabelEncoder (алфавитный порядок)
STORETYPE_MAP    = {"a": 0, "b": 1, "c": 2, "d": 3}
ASSORTMENT_MAP   = {"a": 0, "b": 1, "c": 2}
STATEHOLIDAY_MAP = {"0": 0, "a": 1, "b": 2, "c": 3}

app = FastAPI(title="Rossmann Sales Forecast API", version="1.0")


# 2. Схема входных данных
# Pydantic проверяет типы и границы. Если придёт мусор - вернётся ошибка 422.
class StoreDayFeatures(BaseModel):
    Store: int
    DayOfWeek: int = Field(ge=1, le=7)
    Open: int = Field(ge=0, le=1)
    Promo: int = Field(ge=0, le=1)
    StateHoliday: str = "0"          # '0','a','b','c'
    SchoolHoliday: int = Field(ge=0, le=1)
    StoreType: str                   # 'a'..'d'
    Assortment: str                  # 'a','b','c'
    CompetitionDistance: float
    CompetitionOpenSinceMonth: float = 0
    CompetitionOpenSinceYear: float = 0
    Promo2: int = Field(ge=0, le=1)
    Promo2SinceWeek: float = 0
    Promo2SinceYear: float = 0
    Year: int
    Month: int = Field(ge=1, le=12)
    Day: int = Field(ge=1, le=31)
    WeekOfYear: int
    IsPromoMonth: int = Field(ge=0, le=1)
    lag_7: float
    lag_14: float
    lag_28: float
    rolling_mean_7: float
    rolling_mean_28: float

    # Пример, который подставится на странице /docs (кнопка "Try it out")
    model_config = {
        "json_schema_extra": {
            "example": {
                "Store": 1, "DayOfWeek": 5, "Open": 1, "Promo": 1,
                "StateHoliday": "0", "SchoolHoliday": 1,
                "StoreType": "c", "Assortment": "a",
                "CompetitionDistance": 1270,
                "CompetitionOpenSinceMonth": 9, "CompetitionOpenSinceYear": 2008,
                "Promo2": 0, "Promo2SinceWeek": 0, "Promo2SinceYear": 0,
                "Year": 2015, "Month": 7, "Day": 31, "WeekOfYear": 31,
                "IsPromoMonth": 0,
                "lag_7": 5263, "lag_14": 5020, "lag_28": 4890,
                "rolling_mean_7": 5100, "rolling_mean_28": 5000
            }
        }
    }


# 3. Эндпоинты
@app.get("/health")
def health():
    """Проверка, что сервис жив."""
    return {"status": "ok"}


@app.post("/predict")
def predict(features: StoreDayFeatures):
    """Принимает признаки одного дня магазина и возвращает прогноз продаж."""
    data = features.model_dump()

    # буквы - числа
    data["StoreType"]    = STORETYPE_MAP[data["StoreType"]]
    data["Assortment"]   = ASSORTMENT_MAP[data["Assortment"]]
    data["StateHoliday"] = STATEHOLIDAY_MAP[str(data["StateHoliday"])]

    # строка признаков в правильном порядке
    row = pd.DataFrame([[data[c] for c in FEATURE_ORDER]], columns=FEATURE_ORDER)

    # прогноз
    pred = float(model.predict(row)[0])
    pred = max(pred, 0.0)            

    return {"predicted_sales": round(pred, 2)}
