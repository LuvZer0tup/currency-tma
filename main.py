"""Бэкенд Telegram Mini App: учёт закупа валюты + ИИ-прогноз.

Запуск локально:  uvicorn main:app --reload
Переменные окружения (все необязательные):
  GROQ_API_KEY  — ключ Groq (бесплатный); без него используется встроенная аналитика
  GROQ_MODEL    — модель Groq (по умолчанию llama-3.1-8b-instant)
  BOT_TOKEN     — токен бота, для проверки подписи Telegram initData
  REQUIRE_TG    — "1": принимать запросы только из Telegram (нужен BOT_TOKEN)
"""
import asyncio
import hashlib
import hmac
import os
import statistics
import time
from datetime import date, timedelta
from pathlib import Path
from urllib.parse import parse_qsl

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).parent
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
BOT_TOKEN = os.getenv("BOT_TOKEN", "")
REQUIRE_TG = os.getenv("REQUIRE_TG", "0") == "1"

NBU_URL = "https://bank.gov.ua/NBUStatService/v1/statdirectory/exchange"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
CURRENCY_NAMES = {"USD": "долар США", "EUR": "євро", "PLN": "польський злотий", "GBP": "фунт стерлінгів"}
HISTORY_DAYS = 14
HORIZON_DAYS = 7
CACHE_TTL = 30 * 60

app = FastAPI(title="Currency TMA", docs_url=None, redoc_url=None)
_cache: dict[str, tuple[float, list[float]]] = {}


class ForecastIn(BaseModel):
    currency: str = Field(min_length=3, max_length=3)
    amount: float = Field(gt=0, le=1_000_000_000)
    rate: float = Field(gt=0, le=100_000)
    commission: float = Field(default=0, ge=0, le=20)
    init_data: str = ""


def verify_init_data(init_data: str) -> bool:
    """Проверка подписи Telegram WebApp initData."""
    if not init_data or not BOT_TOKEN:
        return not REQUIRE_TG
    pairs = dict(parse_qsl(init_data, keep_blank_values=True))
    received = pairs.pop("hash", None)
    if not received:
        return False
    check_string = "\n".join(f"{k}={v}" for k, v in sorted(pairs.items()))
    secret = hmac.new(b"WebAppData", BOT_TOKEN.encode(), hashlib.sha256).digest()
    calc = hmac.new(secret, check_string.encode(), hashlib.sha256).hexdigest()
    return hmac.compare_digest(calc, received)


async def _nbu_rate(client: httpx.AsyncClient, code: str, d: date) -> float | None:
    r = await client.get(NBU_URL, params={"valcode": code, "date": d.strftime("%Y%m%d"), "json": ""})
    r.raise_for_status()
    data = r.json()
    return float(data[0]["rate"]) if data else None


async def get_history(code: str) -> list[float]:
    """Курсы НБУ за последние дни (от старых к новым)."""
    cached = _cache.get(code)
    if cached and time.time() - cached[0] < CACHE_TTL:
        return cached[1]
    today = date.today()
    days = [today - timedelta(days=i) for i in range(HISTORY_DAYS - 1, -1, -1)]
    async with httpx.AsyncClient(timeout=10) as client:
        results = await asyncio.gather(*[_nbu_rate(client, code, d) for d in days], return_exceptions=True)
    rates = [r for r in results if isinstance(r, float)]
    if len(rates) < 3:
        raise HTTPException(502, "Не вдалося отримати курси НБУ. Спробуйте пізніше.")
    _cache[code] = (time.time(), rates)
    return rates


def make_forecast(rates: list[float]) -> tuple[float, float]:
    """Линейный тренд -> прогноз на HORIZON_DAYS. Возвращает (прогноз, наклон/день)."""
    xs = list(range(len(rates)))
    slope, _ = statistics.linear_regression(xs, rates)
    current = rates[-1]
    forecast = current + slope * HORIZON_DAYS
    limit = current * 0.05  # не больше ±5% за горизонт
    forecast = max(current - limit, min(current + limit, forecast))
    return forecast, slope


def fallback_analysis(code: str, rates: list[float], current: float, forecast: float) -> str:
    change_pct = (forecast - current) / current * 100
    week_pct = (rates[-1] - rates[0]) / rates[0] * 100
    volatility = statistics.pstdev(rates) / statistics.mean(rates) * 100
    name = CURRENCY_NAMES.get(code, code)
    if change_pct > 0.15:
        trend = f"Курс {code} ({name}) за останній період зріс на {week_pct:+.2f}%, і лінійний тренд вказує на помірне зростання."
    elif change_pct < -0.15:
        trend = f"Курс {code} ({name}) за останній період змінився на {week_pct:+.2f}%, а тренд вказує на помірне зниження."
    else:
        trend = f"Курс {code} ({name}) останнім часом достатньо стабільний ({week_pct:+.2f}%), різких рухів тренд не показує."
    vol = "Волатильність невисока." if volatility < 0.5 else "Волатильність підвищена — враховуйте ризики."
    return f"{trend} Орієнтир на {HORIZON_DAYS} днів: близько {forecast:.2f} ₴ ({change_pct:+.2f}%). {vol}"


async def groq_analysis(code: str, rates: list[float], current: float, forecast: float) -> str | None:
    if not GROQ_API_KEY:
        return None
    name = CURRENCY_NAMES.get(code, code)
    series = ", ".join(f"{r:.2f}" for r in rates)
    prompt = (
        f"Курс {code} ({name}) до гривні за останні {len(rates)} днів за даними НБУ: {series}. "
        f"Поточний курс {current:.2f}, розрахунковий прогноз на {HORIZON_DAYS} днів: {forecast:.2f}. "
        "Напиши короткий аналітичний прогноз українською мовою рівно у 2-3 реченнях: тренд, "
        "ймовірний напрям руху та головний ризик. Без списків, без заголовків, без заклику купувати чи продавати."
    )
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            r = await client.post(
                GROQ_URL,
                headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
                json={
                    "model": GROQ_MODEL,
                    "messages": [
                        {"role": "system", "content": "Ти валютний аналітик. Відповідай лише українською, стисло."},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": 0.4,
                    "max_tokens": 220,
                },
            )
            r.raise_for_status()
            text = r.json()["choices"][0]["message"]["content"].strip()
            return text or None
    except Exception:
        return None


@app.post("/api/forecast")
async def forecast_endpoint(body: ForecastIn):
    if not verify_init_data(body.init_data):
        raise HTTPException(401, "Доступ дозволено лише з Telegram.")
    code = body.currency.upper()
    if code not in CURRENCY_NAMES:
        raise HTTPException(400, "Непідтримувана валюта.")

    rates = await get_history(code)
    current = rates[-1]
    forecast, _ = make_forecast(rates)

    invested = body.amount * body.rate
    revenue = body.amount * forecast * (1 - body.commission / 100)
    profit = revenue - invested
    profit_pct = profit / invested * 100

    text = await groq_analysis(code, rates, current, forecast)
    source = "groq" if text else "local"
    if not text:
        text = fallback_analysis(code, rates, current, forecast)

    return {
        "currency": code,
        "invested": round(invested, 2),
        "current_rate": round(current, 4),
        "forecast_rate": round(forecast, 4),
        "revenue": round(revenue, 2),
        "profit": round(profit, 2),
        "profit_pct": round(profit_pct, 2),
        "analysis": text,
        "ai_source": source,
    }


@app.get("/health")
async def health():
    return {"ok": True}


@app.get("/")
async def index():
    return FileResponse(BASE_DIR / "index.html")
