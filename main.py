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
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Literal
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

NBU_DAY_URL = "https://bank.gov.ua/NBUStatService/v1/statdirectory/exchange"
NBU_RANGE_URL = "https://bank.gov.ua/NBU_Exchange/exchange_site"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
CURRENCIES = {"USD": "US dollar", "EUR": "euro", "PLN": "Polish zloty", "GBP": "British pound"}
LANG_NAMES = {"uk": "Ukrainian", "ru": "Russian", "en": "English"}
CACHE_TTL = 30 * 60

# Локализованные сообщения и шаблоны встроенной аналитики
TXT = {
    "uk": {
        "forbidden": "Доступ дозволено лише з Telegram.",
        "bad_currency": "Непідтримувана валюта.",
        "upstream": "Не вдалося отримати курси НБУ. Спробуйте пізніше.",
        "up": "Курс {code} за останні {n} днів змінився на {hist:+.2f}%, а лінійний тренд вказує на зростання.",
        "down": "Курс {code} за останні {n} днів змінився на {hist:+.2f}%, а лінійний тренд вказує на зниження.",
        "flat": "Курс {code} за останні {n} днів майже не змінився ({hist:+.2f}%), різких рухів тренд не показує.",
        "outlook": "Орієнтир на {h} дн.: близько {f:.2f} ₴ ({c:+.2f}%).",
        "vol_low": "Волатильність низька.",
        "vol_high": "Волатильність підвищена — враховуйте ризики.",
        "far": "Чим довший горизонт, тим менша точність прогнозу.",
    },
    "ru": {
        "forbidden": "Доступ разрешён только из Telegram.",
        "bad_currency": "Неподдерживаемая валюта.",
        "upstream": "Не удалось получить курсы НБУ. Попробуйте позже.",
        "up": "Курс {code} за последние {n} дн. изменился на {hist:+.2f}%, а линейный тренд указывает на рост.",
        "down": "Курс {code} за последние {n} дн. изменился на {hist:+.2f}%, а линейный тренд указывает на снижение.",
        "flat": "Курс {code} за последние {n} дн. почти не изменился ({hist:+.2f}%), резких движений тренд не показывает.",
        "outlook": "Ориентир на {h} дн.: около {f:.2f} ₴ ({c:+.2f}%).",
        "vol_low": "Волатильность низкая.",
        "vol_high": "Волатильность повышена — учитывайте риски.",
        "far": "Чем длиннее горизонт, тем ниже точность прогноза.",
    },
    "en": {
        "forbidden": "Access is allowed from Telegram only.",
        "bad_currency": "Unsupported currency.",
        "upstream": "Could not fetch NBU rates. Please try again later.",
        "up": "The {code} rate changed by {hist:+.2f}% over the last {n} days, and the linear trend points upward.",
        "down": "The {code} rate changed by {hist:+.2f}% over the last {n} days, and the linear trend points downward.",
        "flat": "The {code} rate has barely moved over the last {n} days ({hist:+.2f}%), and the trend shows no sharp moves.",
        "outlook": "Outlook for {h} days: about {f:.2f} UAH ({c:+.2f}%).",
        "vol_low": "Volatility is low.",
        "vol_high": "Volatility is elevated, so keep the risks in mind.",
        "far": "The longer the horizon, the lower the accuracy of the forecast.",
    },
}

app = FastAPI(title="Currency TMA", docs_url=None, redoc_url=None)
_cache: dict[str, tuple[float, list[float]]] = {}


class UpstreamError(Exception):
    pass


class ForecastIn(BaseModel):
    currency: str = Field(min_length=3, max_length=3)
    amount: float = Field(gt=0, le=1_000_000_000)
    rate: float = Field(gt=0, le=100_000)
    commission: float = Field(default=0, ge=0, le=20)
    horizon: Literal[7, 14, 30, 90] = 7
    lang: Literal["uk", "ru", "en"] = "uk"
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


async def _fetch_range(client: httpx.AsyncClient, code: str, days: int) -> list[float]:
    end = date.today()
    start = end - timedelta(days=days)
    r = await client.get(
        NBU_RANGE_URL,
        params={
            "start": start.strftime("%Y%m%d"),
            "end": end.strftime("%Y%m%d"),
            "valcode": code.lower(),
            "sort": "exchangedate",
            "order": "asc",
            "json": "",
        },
    )
    r.raise_for_status()
    rows = r.json()
    rows.sort(key=lambda x: datetime.strptime(x["exchangedate"], "%d.%m.%Y"))
    return [float(x["rate"]) for x in rows]


async def _fetch_day(client: httpx.AsyncClient, code: str, d: date) -> float | None:
    r = await client.get(NBU_DAY_URL, params={"valcode": code, "date": d.strftime("%Y%m%d"), "json": ""})
    r.raise_for_status()
    data = r.json()
    return float(data[0]["rate"]) if data else None


async def get_history(code: str, days: int) -> list[float]:
    """Курсы НБУ за последние `days` дней (от старых к новым)."""
    key = f"{code}:{days}"
    cached = _cache.get(key)
    if cached and time.time() - cached[0] < CACHE_TTL:
        return cached[1]
    rates: list[float] = []
    async with httpx.AsyncClient(timeout=12) as client:
        try:
            rates = await _fetch_range(client, code, days)
        except Exception:
            rates = []
        if len(rates) < 3:  # запасной вариант: запросы по дням (последние 14)
            today = date.today()
            ds = [today - timedelta(days=i) for i in range(13, -1, -1)]
            res = await asyncio.gather(*[_fetch_day(client, code, d) for d in ds], return_exceptions=True)
            rates = [r for r in res if isinstance(r, float)]
    if len(rates) < 3:
        raise UpstreamError()
    _cache[key] = (time.time(), rates)
    return rates


def make_forecast(rates: list[float], horizon: int) -> float:
    """Линейный тренд -> прогноз на horizon дней, с ограничением максимального отклонения."""
    xs = list(range(len(rates)))
    slope, _ = statistics.linear_regression(xs, rates)
    current = rates[-1]
    forecast = current + slope * horizon
    limit = current * 0.05 * (horizon / 7) ** 0.5  # 7д: ±5%, 30д: ±10%, 90д: ±18%
    return max(current - limit, min(current + limit, forecast))


def fallback_analysis(lang: str, code: str, rates: list[float], current: float, forecast: float, horizon: int) -> str:
    t = TXT[lang]
    change = (forecast - current) / current * 100
    hist = (rates[-1] - rates[0]) / rates[0] * 100
    volatility = statistics.pstdev(rates) / statistics.mean(rates) * 100
    kind = "up" if change > 0.15 else "down" if change < -0.15 else "flat"
    parts = [
        t[kind].format(code=code, n=len(rates), hist=hist),
        t["outlook"].format(h=horizon, f=forecast, c=change),
        t["vol_low"] if volatility < 0.5 else t["vol_high"],
    ]
    if horizon >= 30:
        parts.append(t["far"])
    return " ".join(parts)


async def groq_analysis(lang: str, code: str, rates: list[float], current: float, forecast: float, horizon: int) -> str | None:
    if not GROQ_API_KEY:
        return None
    series = ", ".join(f"{r:.2f}" for r in rates[-45:])
    prompt = (
        f"NBU exchange rate of {code} ({CURRENCIES[code]}) to UAH, last {len(rates)} days (showing up to last 45): {series}. "
        f"Current rate: {current:.2f}. Computed linear-trend estimate for {horizon} days ahead: {forecast:.2f}. "
        f"Write a short analytical forecast in exactly 2-3 sentences: the trend, the likely direction over the next "
        f"{horizon} days and the main risk. No lists, no headings, no buy/sell advice. Respond in {LANG_NAMES[lang]}."
    )
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            r = await client.post(
                GROQ_URL,
                headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
                json={
                    "model": GROQ_MODEL,
                    "messages": [
                        {"role": "system", "content": f"You are a concise currency analyst. Always answer in {LANG_NAMES[lang]}."},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": 0.4,
                    "max_tokens": 260,
                },
            )
            r.raise_for_status()
            text = r.json()["choices"][0]["message"]["content"].strip()
            return text or None
    except Exception:
        return None


@app.post("/api/forecast")
async def forecast_endpoint(body: ForecastIn):
    t = TXT[body.lang]
    if not verify_init_data(body.init_data):
        raise HTTPException(401, t["forbidden"])
    code = body.currency.upper()
    if code not in CURRENCIES:
        raise HTTPException(400, t["bad_currency"])

    history_days = min(max(30, body.horizon * 2), 180)
    try:
        rates = await get_history(code, history_days)
    except UpstreamError:
        raise HTTPException(502, t["upstream"])

    current = rates[-1]
    forecast = make_forecast(rates, body.horizon)

    invested = body.amount * body.rate
    revenue = body.amount * forecast * (1 - body.commission / 100)
    profit = revenue - invested
    profit_pct = profit / invested * 100

    text = await groq_analysis(body.lang, code, rates, current, forecast, body.horizon)
    source = "groq" if text else "local"
    if not text:
        text = fallback_analysis(body.lang, code, rates, current, forecast, body.horizon)

    return {
        "currency": code,
        "horizon": body.horizon,
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
