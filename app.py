"""
Rossmann Sales Forecast — веб-демо (Gradio), логика v2.

Пользователь выбирает магазин и дату, а сервис САМ считает признаки
(лаги из истории продаж, свойства магазина, признаки даты) и выдаёт
прогноз дневной выручки. Модель — финальный tuned XGBoost.

Запуск локально:  python app.py   (откроется http://127.0.0.1:7860)
На Hugging Face Spaces (SDK: Gradio) запускается автоматически.
"""

import gradio as gr
import pandas as pd
import numpy as np
from xgboost import XGBRegressor

# ---------- Порядок признаков и кодировки — ровно как при обучении ----------
FEATURE_ORDER = [
    "Store", "DayOfWeek", "Open", "Promo", "StateHoliday", "SchoolHoliday",
    "StoreType", "Assortment", "CompetitionDistance",
    "CompetitionOpenSinceMonth", "CompetitionOpenSinceYear",
    "Promo2", "Promo2SinceWeek", "Promo2SinceYear",
    "Year", "Month", "Day", "WeekOfYear", "IsPromoMonth",
    "lag_7", "lag_14", "lag_28", "rolling_mean_7", "rolling_mean_28",
]
STORETYPE_MAP    = {"a": 0, "b": 1, "c": 2, "d": 3}
ASSORTMENT_MAP   = {"a": 0, "b": 1, "c": 2}
STATEHOLIDAY_MAP = {"0": 0, "a": 1, "b": 2, "c": 3}
MONTH_MAP = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
             7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"}

# ---------- Загрузка артефактов (один раз при старте) ----------
model = XGBRegressor()
model.load_model("xgb_tuned.json")

store = pd.read_csv("store.csv")
store["CompetitionDistance"] = store["CompetitionDistance"].fillna(store["CompetitionDistance"].median())
for c in ["CompetitionOpenSinceMonth", "CompetitionOpenSinceYear",
          "Promo2SinceWeek", "Promo2SinceYear"]:
    store[c] = store[c].fillna(0)
store["PromoInterval"] = store["PromoInterval"].fillna("None")

history = pd.read_csv("sales_history.csv")
history["Date"] = pd.to_datetime(history["Date"])
history = history.sort_values(["Store", "Date"])

STORE_IDS = sorted(history["Store"].unique().tolist())


def predict(store_id, date_str, promo, school, state_holiday):
    # разбираем дату
    try:
        d = pd.Timestamp(date_str)
    except Exception:
        return "❌ Неверный формат даты. Используйте ГГГГ-ММ-ДД, например 2015-07-31."

    store_hist = history[history["Store"] == store_id]
    prior = store_hist[store_hist["Date"] < d]["Sales"].values   # продажи строго ДО даты

    if len(prior) < 28:
        return (f"❌ Недостаточно истории до {d.date()} "
                f"(нужно ≥28 дней, есть {len(prior)}). Возьмите дату ближе к 2015-07-31.")

    srow = store[store["Store"] == store_id].iloc[0]
    is_promo_month = int(MONTH_MAP[d.month] in str(srow["PromoInterval"]))

    # собираем признаки ровно как при обучении
    feat = {
        "Store": store_id,
        "DayOfWeek": d.isoweekday(),                 # Пн=1 ... Вс=7
        "Open": 1,                                   # прогнозируем открытый день
        "Promo": int(promo),
        "StateHoliday": STATEHOLIDAY_MAP[state_holiday],
        "SchoolHoliday": int(school),
        "StoreType": STORETYPE_MAP[srow["StoreType"]],
        "Assortment": ASSORTMENT_MAP[srow["Assortment"]],
        "CompetitionDistance": srow["CompetitionDistance"],
        "CompetitionOpenSinceMonth": srow["CompetitionOpenSinceMonth"],
        "CompetitionOpenSinceYear": srow["CompetitionOpenSinceYear"],
        "Promo2": srow["Promo2"],
        "Promo2SinceWeek": srow["Promo2SinceWeek"],
        "Promo2SinceYear": srow["Promo2SinceYear"],
        "Year": d.year, "Month": d.month, "Day": d.day,
        "WeekOfYear": int(d.isocalendar()[1]),
        "IsPromoMonth": is_promo_month,
        "lag_7": prior[-7], "lag_14": prior[-14], "lag_28": prior[-28],
        "rolling_mean_7": prior[-7:].mean(),
        "rolling_mean_28": prior[-28:].mean(),
    }
    row = pd.DataFrame([[feat[c] for c in FEATURE_ORDER]], columns=FEATURE_ORDER)
    pred = max(float(model.predict(row)[0]), 0.0)

    out = f"### 📈 Прогноз продаж: **{pred:,.0f} €**\n"

    # если дата внутри истории — покажем реальное значение и ошибку
    actual = store_hist[store_hist["Date"] == d]["Sales"]
    if len(actual):
        a = float(actual.iloc[0])
        err = abs(a - pred) / a * 100
        out += f"\nРеальные продажи: **{a:,.0f} €**  •  Ошибка: **{err:.1f}%**"
    return out


demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Dropdown(choices=STORE_IDS, value=1, label="Магазин"),
        gr.Textbox(value="2015-07-31", label="Дата прогноза (ГГГГ-ММ-ДД)"),
        gr.Checkbox(value=True, label="Промо в этот день"),
        gr.Checkbox(value=False, label="Школьные каникулы"),
        gr.Radio(choices=["0", "a", "b", "c"], value="0", label="Гос. праздник (0 = нет)"),
    ],
    outputs=gr.Markdown(label="Результат"),
    title="🛒 Прогноз продаж Rossmann",
    description=("Выберите магазин и дату — модель (tuned XGBoost, MAPE 9.42%) сама посчитает "
                 "признаки из истории продаж и даст прогноз дневной выручки. "
                 "Для дат в известном периоде (до 2015-07-31) показывается и реальное значение."),
)

if __name__ == "__main__":
    demo.launch()
