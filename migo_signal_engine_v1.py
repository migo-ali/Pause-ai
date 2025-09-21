"""
Migo Signal Engine v1.3 — 70‑Tools Scoring + Risk Model (Strict Local, No Network, Auto‑Export)
---------------------------------------------------------------------------------
Why this rewrite?
- Privacy requirement: “مصادر بياناتنا طبعًا مش متاحة لأي حد”.
- Removed any optional online fetch and added **Strict Local Mode** (no fallbacks) so the
  script runs only on your **private CSVs**. No network calls. No yfinance. No leaks.

How to use (Windows / macOS / Linux)
1) Strict‑local with your private CSVs (recommended):
      set MIGO_DATA_DIR=./data   # Windows (or export MIGO_DATA_DIR=./data on macOS/Linux)
      set MIGO_STRICT_LOCAL=1    # (or export MIGO_STRICT_LOCAL=1) default strict mode
      python migo_signal_engine_v1.py --tickers TRML NBIS WOLF --equity 20000 --risk 0.5 --atr 1.5
   Or pass the path explicitly:
      python migo_signal_engine_v1.py --data-dir ./data --strict-local --tickers TRML NBIS WOLF

2) File layout (per ticker):
      {TICKER}_1d.csv   # daily OHLCV (~1y)
      {TICKER}_1m.csv   # intraday 1‑minute (~5d)
   Required columns (case‑insensitive accepted):
      Datetime|Date, Open, High, Low, Close, Volume

3) Auto‑Export of results (enabled by default):
      Outputs written to ./runs (or set --export-dir).
      - latest_scan.csv / latest_scan.json
      - scan_YYYYMMDD_HHMMSS.csv & .json

4) Dry‑run without private data (for dev only):
      python migo_signal_engine_v1.py --synth --tickers ABC XYZ   # uses local synthetic generator

5) Built‑in tests:
      python migo_signal_engine_v1.py --run-tests

Notes:
- Final Buy Strength ∈ [0,100] = weighted average × gates (trend/volume/event).
- Position sizing: size = (equity × risk%) ÷ distance_to_stop  (long only in this file).
- Edit **WEIGHTS** to reflect your 70‑tool mapping. Event flags (earnings/news/low‑float) are injectible per ticker.
- **Security guardrails**: Strict‑local blocks synthetic fallback and requires a valid local directory. The script does
  not import any network‑enabled clients.
"""
from __future__ import annotations
import argparse
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable

import numpy as np
import pandas as pd

# -----------------------------
# 0) Utilities — robust errors & safety
# -----------------------------
class UserError(RuntimeError):
    pass


def die(msg: str) -> None:
    raise UserError(msg)


# Hard safety: this module must never perform network I/O
NO_NETWORK = True  # for documentation; we have no network libs imported

if NO_NETWORK:
    import builtins
    _real_import = builtins.__import__

    def _blocked_import(name, globals=None, locals=None, fromlist=(), level=0):
        forbidden = {"socket", "requests", "urllib", "http", "ftplib", "ssl"}
        root_mod = name.split(".")[0]
        if root_mod in forbidden:
            raise UserError(f"Network modules disabled: attempted import of '{name}'")
        return _real_import(name, globals, locals, fromlist, level)

    builtins.__import__ = _blocked_import


# -----------------------------
# 1) Weights Registry (edit freely)
# -----------------------------
# Weight range suggestion: -1.0 .. +1.0 (negative weight penalizes)
WEIGHTS: Dict[str, float] = {
    # Trend + Structure
    "ma50_above_ma200": 0.10,
    "price_above_ma200": 0.10,
    "higher_highs_ltf": 0.06,
    "adx_trend": 0.06,
    "squeeze_on": 0.05,
    "52w_high_proximity": 0.05,

    # Momentum + Reclaim
    "vwap_reclaim": 0.12,
    "rsi_momentum": 0.05,
    "macd_cross": 0.04,

    # Volume + Liquidity
    "rel_volume": 0.12,
    "obv_up": 0.06,

    # Gaps/News
    "gap_up": 0.06,
    "news_catalyst": 0.08,

    # Risk Filters (negative weights)
    "earnings_within_24h": -0.08,
    "low_float_high_iv": -0.04,
}

# -----------------------------
# 2) Helper math
# -----------------------------
_DEF_EPS = 1e-9


def _to01(x: float) -> float:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return 0.0
    return max(0.0, min(1.0, float(x)))


def weighted_score(indicators: Dict[str, float], weights: Dict[str, float]) -> float:
    """Return weighted average in [0,1]. Missing indicators = 0.
    indicators: dict(name -> value in [0,1] or bool)
    """
    s, w = 0.0, 0.0
    for k, wt in weights.items():
        val = indicators.get(k, 0.0)
        if isinstance(val, bool):
            val = 1.0 if val else 0.0
        val = _to01(val)
        s += wt * val
        w += abs(wt)
    return _to01(0.5 + s / (2.0 * w) if w > 0 else 0.0)  # map negative totals back toward 0..1


# -----------------------------
# 3) Global Gates (trend / volume / event)
# -----------------------------
@dataclass
class Gates:
    trend_gate: float = 1.0  # 0..1
    volume_gate: float = 1.0  # 0..1
    event_gate: float = 1.0   # 0..1

    def apply(self, strength01: float) -> float:
        return strength01 * self.trend_gate * self.volume_gate * self.event_gate


# -----------------------------
# 4) Risk — ATR, Stops, and Position Sizing
# -----------------------------

def true_range(h: pd.Series, l: pd.Series, c: pd.Series) -> pd.Series:
    prev_close = c.shift(1)
    tr1 = h - l
    tr2 = (h - prev_close).abs()
    tr3 = (l - prev_close).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)


def atr(df: pd.DataFrame, period: int = 14) -> float:
    """Average True Range on given DataFrame with columns [High, Low, Close]."""
    if df.empty:
        return float("nan")
    tr = true_range(df["High"], df["Low"], df["Close"]).rolling(period, min_periods=period).mean()
    return float(tr.iloc[-1]) if len(tr.dropna()) else float("nan")


def position_size(equity: float, risk_pct: float, entry: float, stop: float, lot: int = 1) -> int:
    risk_cash = equity * (risk_pct / 100.0)
    dist = max(_DEF_EPS, entry - stop)  # long only here; extend for shorts if needed
    size = math.floor(risk_cash / dist / lot) * lot
    return max(0, size)


# -----------------------------
# 5) Data Provider — CSV (default) or Synthetic (opt‑in)
# -----------------------------
class DataProvider:
    def __init__(self, data_dir: Optional[str] = None, prefer_csv: bool = True, allow_synth: bool = True):
        self.data_dir = Path(data_dir) if data_dir else None
        self.prefer_csv = prefer_csv
        self.allow_synth = allow_synth

    # Basic path allow‑listing: require an explicit directory; resolve to prevent sneaky paths
    def _resolve(self, p: Path) -> Path:
        rp = p.resolve()
        if self.data_dir is not None and not str(rp).startswith(str(self.data_dir.resolve())):
            die(f"Refusing path outside data-dir: {rp}")
        return rp

    def load(self, ticker: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        ticker = str(ticker).upper().strip()
        df_1d, df_1m = None, None
        if self.prefer_csv and self.data_dir:
            df_1d = self._read_csv(self._resolve(self.data_dir / f"{ticker}_1d.csv"))
            df_1m = self._read_csv(self._resolve(self.data_dir / f"{ticker}_1m.csv"))
        if (df_1d is None or df_1m is None) and self.allow_synth:
            df_1d, df_1m = self._synth(ticker)
        if df_1d is None or df_1m is None:
            die(f"No data for {ticker}. Strict‑local requires CSVs: {ticker}_1d.csv & {ticker}_1m.csv")
        return df_1d, df_1m

    @staticmethod
    def _read_csv(path: Path) -> Optional[pd.DataFrame]:
        try:
            if not path.exists():
                return None
            df = pd.read_csv(path)
            # Normalize columns
            cols = {c.lower(): c for c in df.columns}
            # Try to find date column
            date_col = None
            for candidate in ("datetime", "date", "time"):
                if candidate in cols:
                    date_col = cols[candidate]
                    break
            if date_col is None:
                return None
            df[date_col] = pd.to_datetime(df[date_col])
            df = df.rename(columns={
                date_col: "Datetime",
                **{cols.get(k, k): k.capitalize() for k in ["open", "high", "low", "close", "volume"] if k in cols}
            })
            # Ensure required columns
            for k in ["Open", "High", "Low", "Close", "Volume"]:
                if k not in df.columns:
                    return None
            df = df[["Datetime", "Open", "High", "Low", "Close", "Volume"]].sort_values("Datetime").reset_index(drop=True)
            return df
        except Exception:
            return None

    @staticmethod
    def _synth(ticker: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        rng_d = pd.date_range(end=pd.Timestamp.utcnow().normalize(), periods=300, freq="1D")
        rng_m = pd.date_range(end=pd.Timestamp.utcnow(), periods=800, freq="1min")
        rs = np.random.RandomState(abs(hash(ticker)) % (2**32))
        def make_series(n: int, start: float = 50.0) -> np.ndarray:
            steps = rs.normal(loc=0.02/252, scale=0.02, size=n).cumsum()
            base = start * (1.0 + steps)
            return base
        d_close = make_series(len(rng_d), start=40.0)
        d_high = d_close * (1 + rs.rand(len(rng_d)) * 0.01)
        d_low  = d_close * (1 - rs.rand(len(rng_d)) * 0.01)
        d_open = (d_high + d_low) / 2
        d_vol  = rs.randint(5e5, 2e6, size=len(rng_d))
        df_1d = pd.DataFrame({
            "Datetime": rng_d,
            "Open": d_open,
            "High": d_high,
            "Low": d_low,
            "Close": d_close,
            "Volume": d_vol,
        })
        m_close = make_series(len(rng_m), start=float(d_close[-1]))
        m_high = m_close * (1 + rs.rand(len(rng_m)) * 0.003)
        m_low  = m_close * (1 - rs.rand(len(rng_m)) * 0.003)
        m_open = (m_high + m_low) / 2
        m_vol  = rs.randint(1e3, 5e4, size=len(rng_m))
        df_1m = pd.DataFrame({
            "Datetime": rng_m,
            "Open": m_open,
            "High": m_high,
            "Low": m_low,
            "Close": m_close,
            "Volume": m_vol,
        })
        return df_1d, df_1m


# -----------------------------
# 6) Minimal indicator builders from price/volume
# -----------------------------

def _ensure_cols(df: pd.DataFrame) -> pd.DataFrame:
    need = {"Open", "High", "Low", "Close", "Volume"}
    missing = need - set(df.columns)
    if missing:
        die(f"DataFrame missing columns: {missing}")
    return df


def compute_basic_indicators(df_1m: pd.DataFrame, df_1d: pd.DataFrame) -> Dict[str, float]:
    df_1m = _ensure_cols(df_1m)
    df_1d = _ensure_cols(df_1d)
    out: Dict[str, float] = {}

    # Daily MAs
    ma50 = df_1d["Close"].rolling(50, min_periods=50).mean().iloc[-1]
    ma200 = df_1d["Close"].rolling(200, min_periods=200).mean().iloc[-1]
    last = df_1d["Close"].iloc[-1]

    out["ma50_above_ma200"] = 1.0 if (not math.isnan(ma50) and not math.isnan(ma200) and ma50 > ma200) else 0.0
    out["price_above_ma200"] = 1.0 if (not math.isnan(ma200) and last > ma200) else 0.0

    # Lower timeframe structure (1m HH/HL proxy): slope of 30m SMA vs 60m SMA
    sma30 = df_1m["Close"].rolling(30, min_periods=30).mean().iloc[-1]
    sma60 = df_1m["Close"].rolling(60, min_periods=60).mean().iloc[-1]
    out["higher_highs_ltf"] = _to01(1.0 if (not math.isnan(sma30) and not math.isnan(sma60) and sma30 > sma60) else 0.0)

    # ADX proxy: normalized ATR / price
    atr14 = atr(df_1d, 14)
    out["adx_trend"] = _to01((atr14 / last) * 10.0) if (not math.isnan(atr14) and last > 0) else 0.0

    # Squeeze proxy: stddev(1m) low over last 2h (120 bars)
    std_1m = df_1m["Close"].rolling(120, min_periods=120).std().iloc[-1]
    pct_vol = std_1m / last if last > 0 and not math.isnan(std_1m) else 0.3
    out["squeeze_on"] = _to01(0.3 - pct_vol)  # lower vol → higher score

    # 52w high proximity
    rollmax = df_1d["High"].rolling(252, min_periods=10).max().iloc[-1]
    if math.isnan(rollmax) or rollmax <= 0:
        out["52w_high_proximity"] = 0.0
    else:
        out["52w_high_proximity"] = _to01(1.0 - ((rollmax - last) / max(_DEF_EPS, rollmax)))

    # VWAP reclaim (intraday)
    tp = (df_1m["High"] + df_1m["Low"] + df_1m["Close"]) / 3.0
    vol = df_1m["Volume"].replace(0, np.nan)
    vwap = (tp * vol).cumsum() / vol.cumsum()
    out["vwap_reclaim"] = 1.0 if df_1m["Close"].iloc[-1] > vwap.iloc[-1] else 0.0

    # RSI‑like momentum (normalized 1m change over last 30 bars)
    chg = df_1m["Close"].pct_change().rolling(30, min_periods=30).mean().iloc[-1]
    if math.isnan(chg):
        out["rsi_momentum"] = 0.5
    else:
        out["rsi_momentum"] = _to01((chg + 0.02) / 0.04)  # map ~[-2%,+2%] to [0,1]

    # MACD cross proxy (12/26 EMA cross on 1m)
    ema12 = df_1m["Close"].ewm(span=12, adjust=False).mean().iloc[-1]
    ema26 = df_1m["Close"].ewm(span=26, adjust=False).mean().iloc[-1]
    out["macd_cross"] = 1.0 if ema12 > ema26 else 0.0

    # Relative volume (today vs 20‑day avg)
    today_vol = float(df_1m["Volume"].tail(390).sum())  # approx one session
    avg_daily_vol = float(df_1d["Volume"].tail(20).mean())
    relv = today_vol / max(1.0, avg_daily_vol) if avg_daily_vol > 0 else 0.0
    out["rel_volume"] = _to01(relv / 2.0)  # relvol 2.0 → score ~1.0

    # OBV up (close up‑weighted)
    obv = ((np.sign(df_1m["Close"].diff().fillna(0)) * df_1m["Volume"]).cumsum()).iloc[-1]
    out["obv_up"] = _to01(0.5 + np.tanh(obv / (today_vol + _DEF_EPS)) / 2)

    # Gap up
    prev_close = df_1d["Close"].iloc[-2] if len(df_1d) >= 2 else df_1d["Close"].iloc[-1]
    gap = (last - prev_close) / max(_DEF_EPS, prev_close)
    out["gap_up"] = _to01((gap + 0.05) / 0.10)  # +5% gap→~0.5, +15%→~1.0

    return out


# -----------------------------
# 7) Event filters (fill using your calendar/news feed)
# -----------------------------
@dataclass
class EventContext:
    earnings_within_24h: bool = False
    news_catalyst: bool = False
    low_float_high_iv: bool = False


def build_gates(ind: Dict[str, float], ev: EventContext) -> Gates:
    # Trend gate: only allow full strength if above MA200 and ma50>ma200
    trend = 1.0 if (ind.get("price_above_ma200", 0) and ind.get("ma50_above_ma200", 0)) else 0.6
    # Volume gate: at least relvol ~1.0 for full credit
    vol_gate = 1.0 if ind.get("rel_volume", 0) >= 0.5 else 0.7
    # Event gate: penalize earnings window; slightly boost bona fide catalyst
    ev_gate = 1.0
    if ev.earnings_within_24h:
        ev_gate *= 0.6
    if ev.news_catalyst:
        ev_gate *= 1.05
    if ev.low_float_high_iv:
        ev_gate *= 0.85
    return Gates(trend_gate=trend, volume_gate=vol_gate, event_gate=ev_gate)


# -----------------------------
# 8) Trade plan per ticker
# -----------------------------
@dataclass
class TradePlan:
    ticker: str
    buy_strength_pct: float
    entry: float
    stop: float
    target1: float
    target2: float
    size_shares: int
    r_multiple: float


def plan_for_ticker(ticker: str, equity: float, risk_pct: float, atr_mult: float,
                    ev: Optional[EventContext] = None,
                    loader: Optional[Callable[[str], Tuple[pd.DataFrame, pd.DataFrame]]] = None) -> TradePlan:
    if loader is None:
        die("No data loader supplied. Strict‑local requires DataProvider(...).load")
    ev = ev or EventContext()

    # Load data (local only)
    df_1d, df_1m = loader(ticker)
    if df_1d is None or df_1m is None or len(df_1d) == 0 or len(df_1m) == 0:
        die(f"No data for {ticker}")

    # Build indicators
    ind = compute_basic_indicators(df_1m, df_1d)
    ind["news_catalyst"] = 1.0 if ev.news_catalyst else 0.0
    ind["earnings_within_24h"] = 1.0 if ev.earnings_within_24h else 0.0
    ind["low_float_high_iv"] = 1.0 if ev.low_float_high_iv else 0.0

    base01 = weighted_score(ind, WEIGHTS)
    g = build_gates(ind, ev)
    final01 = g.apply(base01)
    buy_strength_pct = round(final01 * 100.0, 1)

    # Risk model: ATR stop, entry = last close 1m, targets at 2R / 3R
    last = float(df_1m["Close"].iloc[-1])
    atr_val = atr(df_1m[["High", "Low", "Close"]], 14)
    if math.isnan(atr_val) or atr_val <= 0:
        atr_val = atr(df_1d[["High", "Low", "Close"]], 14)
    if math.isnan(atr_val) or atr_val <= 0:
        atr_val = max(_DEF_EPS, last * 0.01)

    stop = max(_DEF_EPS, last - atr_mult * atr_val)

    tp = (df_1m["High"] + df_1m["Low"] + df_1m["Close"]) / 3.0
    vwap = (tp * df_1m["Volume"].replace(0, np.nan)).cumsum() / df_1m["Volume"].replace(0, np.nan).cumsum()
    vwap_stop = float(vwap.iloc[-1] * 0.995)
    stop = min(stop, vwap_stop)  # tighter stop wins

    dist = max(_DEF_EPS, last - stop)
    size = position_size(equity, risk_pct, last, stop)

    target1 = last + 2.0 * dist
    target2 = last + 3.0 * dist

    r_mult = (target1 - last) / dist if dist > 0 else 0.0

    return TradePlan(
        ticker=ticker,
        buy_strength_pct=buy_strength_pct,
        entry=round(last, 4),
        stop=round(stop, 4),
        target1=round(target1, 4),
        target2=round(target2, 4),
        size_shares=size,
        r_multiple=round(r_mult, 2),
    )


# -----------------------------
# 9) Export helpers
# -----------------------------
from datetime import datetime

def export_scan(df: pd.DataFrame, export_dir: Optional[str]) -> Optional[Tuple[str, str]]:
    if not export_dir:
        return None
    outdir = Path(export_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    latest_csv = outdir / "latest_scan.csv"
    latest_json = outdir / "latest_scan.json"
    df.to_csv(latest_csv, index=False)
    df.to_json(latest_json, orient="records", indent=2)
    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    ts_csv = outdir / f"scan_{stamp}.csv"
    ts_json = outdir / f"scan_{stamp}.json"
    df.to_csv(ts_csv, index=False)
    df.to_json(ts_json, orient="records", indent=2)
    return str(latest_csv), str(latest_json)

# -----------------------------
# 10) CLI — quick scan
# -----------------------------
DEFAULT_TICKERS = ["TRML", "NBIS", "WOLF"]


def run_scan(tickers: List[str], equity: float, risk: float, atr_mult: float,
             data_dir: Optional[str], allow_synth: bool, strict_local: bool) -> pd.DataFrame:
    if strict_local:
        # In strict‑local we must have a dir and we must NOT allow synth fallback.
        resolved_dir = data_dir or os.environ.get("MIGO_DATA_DIR")
        if not resolved_dir:
            die("--strict-local requires --data-dir or MIGO_DATA_DIR env var.")
        if not Path(resolved_dir).exists():
            die(f"--strict-local directory does not exist: {resolved_dir}")
        dp = DataProvider(data_dir=resolved_dir, prefer_csv=True, allow_synth=False)
    else:
        dp = DataProvider(data_dir=data_dir or os.environ.get("MIGO_DATA_DIR"),
                          prefer_csv=not allow_synth, allow_synth=allow_synth)

    plans: List[TradePlan] = []
    for t in tickers:
        try:
            ev = EventContext(earnings_within_24h=False, news_catalyst=False, low_float_high_iv=False)
            plan = plan_for_ticker(t, equity, risk, atr_mult, ev, loader=dp.load)
            plans.append(plan)
        except Exception as e:
            print(f"[WARN] {t}: {e}")
    if not plans:
        die("No plans generated. Strict‑local needs valid CSVs in data-dir.")
    df = pd.DataFrame([p.__dict__ for p in plans]).sort_values("buy_strength_pct", ascending=False)
    return df


# -----------------------------
# 11) TESTS (do not change; add more below)
# -----------------------------
# -----------------------------

def _almost(a: float, b: float, eps: float = 1e-6) -> bool:
    return abs(float(a) - float(b)) <= eps


def run_tests() -> None:
    print("Running tests…")
    # Test 1: position sizing
    size = position_size(equity=10000, risk_pct=1.0, entry=105.0, stop=100.0)
    assert size == 20, f"position_size expected 20, got {size}"

    # Test 2: weighted_score bounds and symmetry
    w = {"a": 1.0, "b": -1.0}
    s0 = weighted_score({"a": 0.0, "b": 1.0}, w)  # cancels
    s1 = weighted_score({"a": 1.0, "b": 0.0}, w)
    assert 0.0 <= s0 <= 1.0 and 0.0 <= s1 <= 1.0, "weighted_score out of bounds"

    # Test 3: indicators from synthetic data produce required keys
    dp = DataProvider(prefer_csv=False, allow_synth=True)
    d1, m1 = dp.load("TEST")
    ind = compute_basic_indicators(m1, d1)
    needed = set(WEIGHTS.keys()) - {"earnings_within_24h", "low_float_high_iv", "news_catalyst"}
    assert needed.issubset(set(ind.keys())), f"indicators missing keys: {needed - set(ind.keys())}"

    # Test 4: plan_for_ticker runs and returns sane ranges (synth)
    plan = plan_for_ticker("TEST", equity=20000, risk_pct=0.5, atr_mult=1.5, loader=dp.load)
    assert 0.0 <= plan.buy_strength_pct <= 100.0, "buy_strength out of range"
    assert plan.size_shares >= 0 and plan.entry > plan.stop, "invalid trade plan sizing/levels"

    # Test 5: event gates penalize/boost
    plan_base = plan_for_ticker("TEST", 20000, 0.5, 1.5, EventContext(False, False, False), dp.load)
    plan_earn = plan_for_ticker("TEST", 20000, 0.5, 1.5, EventContext(True, False, False), dp.load)
    plan_news = plan_for_ticker("TEST", 20000, 0.5, 1.5, EventContext(False, True, False), dp.load)
    assert plan_earn.buy_strength_pct <= plan_base.buy_strength_pct + 1e-6, "earnings should not boost"
    assert plan_news.buy_strength_pct >= plan_base.buy_strength_pct - 1e-6, "news should not penalize"

    # Test 6: strict‑local must fail if CSVs missing
    try:
        _ = run_scan(["A"], 20000, 0.5, 1.5, data_dir=None, allow_synth=False, strict_local=True)
        raise AssertionError("strict-local should have failed without data-dir")
    except UserError:
        pass

    # Test 7: CSV loader round‑trip (create temp CSVs)
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        d_dates = pd.date_range("2024-01-01", periods=3, freq="D")
        d = pd.DataFrame({
            "Date": d_dates,
            "Open": [100, 101, 102],
            "High": [101, 102, 103],
            "Low":  [ 99, 100, 101],
            "Close": [100.5, 101.5, 102.5],
            "Volume": [1_000_000, 1_100_000, 1_050_000],
        })
        m_dates = pd.date_range("2024-01-03 09:30", periods=200, freq="1min")
        m = pd.DataFrame({
            "Datetime": m_dates,
            "Open": np.linspace(102, 103, 200),
            "High": np.linspace(102.1, 103.1, 200),
            "Low":  np.linspace(101.9, 102.9, 200),
            "Close": np.linspace(102.0, 103.0, 200),
            "Volume": np.random.randint(1000, 5000, size=200),
        })
        d.to_csv(td_path / "CSV_1d.csv", index=False)
        m.to_csv(td_path / "CSV_1m.csv", index=False)
        dp2 = DataProvider(data_dir=str(td_path), prefer_csv=True, allow_synth=False)
        d1, m1 = dp2.load("CSV")
        assert not d1.empty and not m1.empty, "CSV loader failed to read files"
        plan_csv = plan_for_ticker("CSV", 10000, 1.0, 1.5, loader=dp2.load)
        assert plan_csv.entry > plan_csv.stop, "CSV plan produced invalid levels"

    # Test 8: export writes both CSV and JSON
    with tempfile.TemporaryDirectory() as td:
        df = run_scan(["A", "B", "C"], 20000, 0.5, 1.5, data_dir=None, allow_synth=True, strict_local=False)
        csv_path, json_path = export_scan(df, td)
        assert Path(csv_path).exists() and Path(json_path).exists(), "Export files missing"

    # Test 9: network imports are blocked
    try:
        __import__("socket")
        raise AssertionError("socket import should be blocked")
    except UserError:
        pass

    print("All tests passed ✔")


# -----------------------------
# 12) Main
# -----------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tickers", nargs="*", default=DEFAULT_TICKERS)
    p.add_argument("--equity", type=float, default=20000.0, help="Account equity in USD")
    p.add_argument("--risk", type=float, default=0.5, help="Risk per trade in % of equity")
    p.add_argument("--atr", type=float, default=1.5, help="ATR multiplier for stop")
    p.add_argument("--data-dir", type=str, default=None, help="Folder with {TICKER}_1d.csv and {TICKER}_1m.csv")
    p.add_argument("--synth", action="store_true", help="Allow synthetic data (for dev only)")
    p.add_argument("--strict-local", action="store_true", help="Enforce local CSVs only; no fallbacks")
    p.add_argument("--export-dir", type=str, default=os.environ.get("MIGO_EXPORT_DIR", "./runs"), help="Directory to write latest_scan.csv/json")
    p.add_argument("--no-export", action="store_true", help="Disable writing export files")
    p.add_argument("--run-tests", action="store_true", help="Run built‑in tests and exit")
    args = p.parse_args()

    if args.run_tests:
        run_tests()
        return

    allow_synth = bool(args.synth)
    strict_env = os.environ.get("MIGO_STRICT_LOCAL", "0").strip() in ("1", "true", "True")
    strict_local = bool(args.strict_local or (strict_env and not args.synth))

    try:
        df = run_scan(args.tickers, args.equity, args.risk, args.atr,
                      data_dir=args.data_dir or os.environ.get("MIGO_DATA_DIR"),
                      allow_synth=allow_synth, strict_local=strict_local)
    except UserError as ue:
        print(f"[ERROR] {ue}")
        return

    print("\n=== Migo Signal Engine — Trade Plans ===")
    print(df.to_string(index=False))

    best = df.head(3)
    print("\nTop 3 by Buy Strength:")
    for _, r in best.iterrows():
        print(f"- {r['ticker']}: {r['buy_strength_pct']}% | entry {r['entry']} | stop {r['stop']} | size {int(r['size_shares'])} | T1 {r['target1']} | T2 {r['target2']}")

    # Auto-export unless disabled
    if not args.no_export:
        paths = export_scan(df, args.export_dir)
        if paths:
            print(f"[Saved] latest CSV/JSON → {paths[0]} , {paths[1]}")


if __name__ == "__main__":
    main()
