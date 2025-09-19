"""Pause BOOM backtest runner with Optuna tuning and Telegram reporting.

This script fetches intraday market data from multiple vendors, engineers
technical indicators, performs walk-forward cross validation with Optuna tuned
LightGBM models, simulates trade execution, and optionally sends the summary via
Telegram.
"""

import argparse
import os
import warnings
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional, Sequence

import backoff
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import requests
import yfinance as yf
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ---------------- CONFIG ----------------
POLYGON_KEY = os.getenv("POLYGON_KEY")
ALPHAVANTAGE_KEY = os.getenv("ALPHAVANTAGE_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

CACHE_DIR = Path("./cache")
CACHE_DIR.mkdir(exist_ok=True)
OUT_DIR = Path("./outputs")
OUT_DIR.mkdir(exist_ok=True)

RISK_PER_TRADE = 0.01
BASE_SPREAD_PCT = 0.0008
BASE_SLIPPAGE_PCT = 0.0005
COMMISSION_PCT = 0.0005
MIN_AVG_VOLUME = 10000

LOOKBACK = 20
HOLD_DAYS = 5
LABEL_HORIZON = 5
LABEL_THRESHOLD = 0.02

DEFAULT_OPTUNA_TRIALS = 20


# ---------------- helpers ----------------
def nowtag() -> str:
    return datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")


def cache_path(ticker: str, tag: str) -> Path:
    return CACHE_DIR / f"{ticker}__{tag}.parquet"


def send_telegram_text(token: Optional[str], chat_id: Optional[str], text: str) -> bool:
    if not token or not chat_id:
        return False
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    data = {"chat_id": chat_id, "text": text, "parse_mode": "HTML"}
    try:
        response = requests.post(url, data=data, timeout=20)
        return response.ok
    except Exception as exc:  # pragma: no cover - network failures are logged
        print("[tg err]", exc)
        return False


def send_telegram_file(
    token: Optional[str], chat_id: Optional[str], filepath: str, caption: Optional[str] = None
) -> bool:
    if not token or not chat_id:
        return False
    url = f"https://api.telegram.org/bot{token}/sendDocument"
    try:
        with open(filepath, "rb") as file_pointer:
            files = {"document": file_pointer}
            data = {"chat_id": chat_id, "caption": caption or ""}
            response = requests.post(url, data=data, files=files, timeout=60)
        return response.ok
    except Exception as exc:  # pragma: no cover - network failures are logged
        print("[tg file err]", exc)
        return False


@backoff.on_exception(
    backoff.expo, (requests.exceptions.RequestException, ValueError), max_tries=3
)
def requests_get(url: str, params: Optional[dict] = None, timeout: int = 20) -> requests.Response:
    response = requests.get(url, params=params, timeout=timeout)
    response.raise_for_status()
    return response


# ---------------- DATA Fetch (Polygon range preferred) ----------------
def fetch_polygon_range(
    ticker: str, from_dt: pd.Timestamp, to_dt: pd.Timestamp, multiplier: int = 60, timespan: str = "minute"
) -> Optional[pd.DataFrame]:
    if not POLYGON_KEY:
        return None
    url = (
        "https://api.polygon.io/v2/aggs/ticker/"
        f"{ticker}/range/{multiplier}/{timespan}/{from_dt.strftime('%Y-%m-%d')}/{to_dt.strftime('%Y-%m-%d')}"
    )
    params = {"adjusted": "true", "sort": "asc", "limit": 50000, "apiKey": POLYGON_KEY}
    try:
        response = requests_get(url, params=params)
        data = response.json()
        if "results" not in data:
            return None
        rows = []
        for payload in data["results"]:
            dt = pd.to_datetime(payload["t"], unit="ms")
            rows.append([dt, payload["o"], payload["h"], payload["l"], payload["c"], payload["v"]])
        df = pd.DataFrame(rows, columns=["datetime", "open", "high", "low", "close", "volume"])
        return df.set_index("datetime")
    except Exception as exc:  # pragma: no cover - network failures are logged
        print("[polygon fetch err]", exc)
        return None


def fetch_alpha_intraday(ticker: str, interval: str = "60min") -> Optional[pd.DataFrame]:
    if not ALPHAVANTAGE_KEY:
        return None
    url = "https://www.alphavantage.co/query"
    params = {
        "function": "TIME_SERIES_INTRADAY",
        "symbol": ticker,
        "interval": interval,
        "outputsize": "compact",
        "apikey": ALPHAVANTAGE_KEY,
    }
    try:
        response = requests_get(url, params=params)
        raw = response.json()
        key = next((k for k in raw if "Time Series" in k), None)
        if not key:
            return None
        df = pd.DataFrame.from_dict(raw[key], orient="index")
        df.columns = [col.split(". ")[-1] for col in df.columns]
        df = df.rename(columns={"open": "open", "high": "high", "low": "low", "close": "close", "volume": "volume"})
        df = df.astype(float)
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        return df[["open", "high", "low", "close", "volume"]]
    except Exception as exc:  # pragma: no cover - network failures are logged
        print("[alpha fetch err]", exc)
        return None


def fetch_yfinance_intraday(
    ticker: str, period_days: int = 30, interval: str = "60m"
) -> Optional[pd.DataFrame]:
    try:
        df = yf.download(ticker, period=f"{period_days}d", interval=interval, progress=False)
        if df.empty:
            return None
        df = df.rename(
            columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"}
        )
        df.index = pd.to_datetime(df.index)
        return df[["open", "high", "low", "close", "volume"]]
    except Exception as exc:  # pragma: no cover - network failures are logged
        print("[yf err]", exc)
        return None


def unify_resample(df: pd.DataFrame, interval_minutes: int = 60) -> pd.DataFrame:
    freq = f"{interval_minutes}T"
    if len(df) > 1:
        ohlc = df.resample(freq).agg(
            {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
        )
        return ohlc.dropna()
    return df


def get_intraday_df(
    ticker: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    interval_minutes: int = 60,
    prefer: Sequence[str] = ("polygon", "alpha", "yfinance"),
) -> pd.DataFrame:
    """Fetch intraday OHLCV data using the configured vendor preference order."""

    start_dt = pd.to_datetime(start)
    end_dt = pd.to_datetime(end)
    tag = f"intraday_{interval_minutes}m_{start_dt.strftime('%Y%m%d')}_{end_dt.strftime('%Y%m%d')}"
    cached = cache_path(ticker, tag)
    if cached.exists():
        try:
            return pd.read_parquet(cached)
        except Exception:
            pass
    df = None
    if "polygon" in prefer:
        df = fetch_polygon_range(ticker, start_dt, end_dt, multiplier=interval_minutes, timespan="minute")
    if (df is None or df.empty) and "alpha" in prefer:
        df = fetch_alpha_intraday(ticker, interval=f"{interval_minutes}min")
    if (df is None or df.empty) and "yfinance" in prefer:
        df = fetch_yfinance_intraday(
            ticker, period_days=(end_dt - start_dt).days + 1, interval=f"{interval_minutes}m"
        )
    if df is None or df.empty:
        raise RuntimeError(f"No data for {ticker}")
    df = unify_resample(df, interval_minutes)
    try:
        df.to_parquet(cached)
    except Exception:
        df.to_csv(cached.with_suffix(".csv"))
    return df


# ------------- features, labels, model -------------
def add_features_intraday(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["ret_1"] = d["close"].pct_change(1)
    d["ret_5"] = d["close"].pct_change(5)
    d["sma_10"] = d["close"].rolling(10).mean()
    d["sma_20"] = d["close"].rolling(20).mean()
    d["ema_20"] = d["close"].ewm(span=20, adjust=False).mean()
    tr = pd.concat(
        [
            d["high"] - d["low"],
            (d["high"] - d["close"].shift(1)).abs(),
            (d["low"] - d["close"].shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    d["atr_14"] = tr.rolling(14).mean()
    delta = d["close"].diff()
    up = delta.clip(lower=0).rolling(14).mean()
    down = -delta.clip(upper=0).rolling(14).mean()
    d["rsi_14"] = 100 - 100 / (1 + up / (down.replace(0, np.nan)))
    d["vwap"] = (d["close"] * d["volume"]).rolling(20).sum() / (d["volume"].rolling(20).sum() + 1e-9)
    d["obv"] = (np.sign(d["close"].diff()) * d["volume"]).fillna(0).cumsum()
    d["vol_z"] = (d["volume"] - d["volume"].rolling(20).mean()) / (
        d["volume"].rolling(20).std() + 1e-9
    )
    d["mom_5"] = d["close"].pct_change(5)
    d["rolling_high_20"] = d["close"].rolling(20).max().shift(1)
    d["rolling_low_20"] = d["close"].rolling(20).min().shift(1)
    return d.dropna()


def make_labels(df: pd.DataFrame, horizon: int = LABEL_HORIZON, thr: float = LABEL_THRESHOLD) -> pd.DataFrame:
    future = df["close"].shift(-horizon) / df["close"] - 1
    df["label_bin"] = (future > thr).astype(int)
    df["target_return"] = future
    return df.dropna(subset=["label_bin"])


def walk_forward_train(
    df: pd.DataFrame, n_splits: int = 5, optuna_trials: int = DEFAULT_OPTUNA_TRIALS
):
    X = df.drop(columns=["label_bin", "target_return", "open", "high", "low", "close", "volume"], errors="ignore")
    y = df["label_bin"]
    tscv = TimeSeriesSplit(n_splits=n_splits)
    preds = pd.Series(index=X.index, dtype=float)
    models = []
    fold_metrics = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        def objective(trial: optuna.Trial) -> float:
            params = {
                "objective": "binary",
                "metric": "binary_logloss",
                "verbosity": -1,
                "num_leaves": trial.suggest_int("num_leaves", 16, 256),
                "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.2, log=True),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 200),
                "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
            }
            dtrain = lgb.Dataset(X_train_s, label=y_train)
            cv = lgb.cv(
                params,
                dtrain,
                nfold=3,
                early_stopping_rounds=30,
                num_boost_round=200,
                verbose_eval=False,
            )
            return min(cv["binary_logloss-mean"])

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=optuna_trials, show_progress_bar=False)
        best = study.best_params
        best.update({"objective": "binary", "metric": "binary_logloss", "verbosity": -1})
        dtrain = lgb.Dataset(X_train_s, label=y_train)
        model = lgb.train(best, dtrain, num_boost_round=200, verbose_eval=False)
        y_pred_prob = model.predict(X_test_s)
        preds.iloc[test_idx] = y_pred_prob
        models.append((model, scaler))
        thr = 0.6
        y_pred_bin = (y_pred_prob > thr).astype(int)
        fold_metrics.append(
            {
                "fold": fold,
                "accuracy": float(accuracy_score(y_test, y_pred_bin)),
                "precision": float(precision_score(y_test, y_pred_bin, zero_division=0)),
                "recall": float(recall_score(y_test, y_pred_bin, zero_division=0)),
                "n_test": len(y_test),
            }
        )
        print(
            f"[fold {fold}] optuna best num_leaves={best.get('num_leaves')}, lr={best.get('learning_rate')}"
        )
    return models, preds, pd.DataFrame(fold_metrics)


def dynamic_slippage(order_notional: float, avg_volume: float, base_spread: float = BASE_SPREAD_PCT) -> float:
    k = 0.1
    alpha = 0.6
    if avg_volume <= 0:
        return BASE_SLIPPAGE_PCT + base_spread
    liquidity_ratio = order_notional / (avg_volume + 1e-9)
    slippage = BASE_SLIPPAGE_PCT + base_spread + k * (liquidity_ratio ** alpha)
    return min(max(slippage, 0.0001), 0.05)


def simulate_trades(
    df: pd.DataFrame,
    preds_prob: dict,
    capital: float = 100_000.0,
    risk_per_trade: float = RISK_PER_TRADE,
    threshold: float = 0.6,
) -> pd.DataFrame:
    trades = []
    avg_vol = df["volume"].rolling(20).mean().fillna(method="bfill")
    i = 0
    while i < len(df):
        row = df.iloc[i]
        prob = float(preds_prob.get(row.name, 0))
        if prob >= threshold:
            entry_price = row["open"] * (1 + BASE_SLIPPAGE_PCT)
            atr = max(row.get("atr_14", row["high"] - row["low"]), 1e-9)
            risk_distance_pct = atr / entry_price
            if risk_distance_pct < 0.005:
                risk_distance_pct = 0.02
            position_usd = capital * risk_per_trade
            n_shares = max(int(position_usd / (entry_price * risk_distance_pct)), 1)
            order_notional = n_shares * entry_price
            local_avg = avg_vol.loc[row.name] if row.name in avg_vol.index else float(avg_vol.mean())
            slippage_pct = dynamic_slippage(order_notional, local_avg)
            entry_exec = entry_price * (1 + slippage_pct + BASE_SPREAD_PCT)
            stop_price = entry_exec * (1 - (risk_distance_pct * 0.9))
            target_price = entry_exec * (1 + 0.06)
            exit_price = entry_exec
            exit_date = row.name
            win = False
            loss = False
            for step in range(1, HOLD_DAYS + 1):
                idx = i + step
                if idx >= len(df):
                    break
                high = df.iloc[idx]["high"]
                low = df.iloc[idx]["low"]
                if low <= stop_price:
                    exit_price = stop_price * (1 - slippage_pct)
                    exit_date = df.index[idx]
                    loss = True
                    break
                if high >= target_price:
                    exit_price = target_price * (1 - slippage_pct)
                    exit_date = df.index[idx]
                    win = True
                    break
                if step == HOLD_DAYS:
                    exit_price = df.iloc[idx]["close"] * (1 - slippage_pct)
                    exit_date = df.index[idx]
            ret_per = (exit_price - entry_exec) / entry_exec
            net_return = ret_per - COMMISSION_PCT
            trades.append(
                {
                    "entry_date": row.name,
                    "exit_date": exit_date,
                    "entry_exec": entry_exec,
                    "exit_exec": exit_price,
                    "n_shares": n_shares,
                    "order_notional": order_notional,
                    "return": net_return,
                    "win": win,
                    "loss": loss,
                    "pred_prob": prob,
                    "slippage_pct": slippage_pct,
                }
            )
            try:
                i = df.index.get_loc(exit_date) + 1
            except KeyError:
                i += 1
        else:
            i += 1
    return pd.DataFrame(trades)


def compute_metrics(tr_df: pd.DataFrame) -> dict:
    if tr_df.empty:
        return {"trades": 0}
    wins = tr_df[tr_df["return"] > 0]
    losses = tr_df[tr_df["return"] <= 0]
    gross_profit = wins["return"].sum()
    gross_loss = -losses["return"].sum() if len(losses) > 0 else 0.0
    pf = gross_profit / gross_loss if gross_loss > 0 else np.nan
    hit = len(wins) / len(tr_df) if len(tr_df) > 0 else np.nan
    equity = [100_000.0]
    for r in tr_df["return"]:
        equity.append(equity[-1] * (1 + r * 0.1))
    eq = np.array(equity)
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / peak
    max_dd = dd.min() * 100
    rets = np.diff(eq) / eq[:-1] if len(eq) > 1 else np.array([0.0])
    sharpe = (
        (np.mean(rets) / (np.std(rets) if np.std(rets) > 0 else 1e-9)) * np.sqrt(252)
        if len(rets) > 1
        else np.nan
    )
    return {
        "trades": len(tr_df),
        "hit_rate": round(hit, 4),
        "profit_factor": round(pf, 4) if not np.isnan(pf) else np.nan,
        "max_drawdown_pct": round(max_dd, 3),
        "sharpe": round(sharpe, 4) if not np.isnan(sharpe) else np.nan,
        "avg_return_per_trade": round(tr_df["return"].mean(), 6),
    }


# ----------------- pipeline per ticker -----------------
def run_ticker(
    ticker: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    interval_minutes: int = 60,
    optuna_trials: int = DEFAULT_OPTUNA_TRIALS,
):
    print(f"[+] {ticker} fetching {start} -> {end}")
    df_raw = get_intraday_df(
        ticker,
        start,
        end,
        interval_minutes=interval_minutes,
        prefer=("polygon", "alpha", "yfinance"),
    )
    df_feat = add_features_intraday(df_raw)
    df_lab = make_labels(df_feat, horizon=LABEL_HORIZON, thr=LABEL_THRESHOLD)
    avg_vol = df_lab["volume"].rolling(20).mean()
    if avg_vol.median() < MIN_AVG_VOLUME:
        print(f"[!] {ticker} low liquidity median avg_vol={avg_vol.median():.0f} — skipping")
        return None
    models, preds_prob, fold_metrics = walk_forward_train(
        df_lab, n_splits=5, optuna_trials=optuna_trials
    )
    trades = simulate_trades(df_lab, preds_prob.to_dict())
    metrics = compute_metrics(trades)
    trades.to_csv(OUT_DIR / f"{ticker}_trades.csv", index=False)
    pd.DataFrame([metrics]).to_csv(OUT_DIR / f"{ticker}_metrics.csv", index=False)
    fold_metrics.to_csv(OUT_DIR / f"{ticker}_fold_metrics.csv", index=False)
    print(
        f"[=] {ticker} done — trades:{metrics['trades']} hit:{metrics.get('hit_rate')} pf:{metrics.get('profit_factor')}"
    )
    return {"ticker": ticker, "metrics": metrics, "trades": trades}


# ---------------- reporting (PDF summary) ----------------
def make_pdf_report(aggregate_rows: Iterable[dict], filename: str) -> str:
    fig, ax = plt.subplots(figsize=(8.27, 11.69))
    ax.axis("off")
    title = f"Pause BOOM — Backtest Report — {nowtag()}"
    ax.text(0.5, 0.95, title, ha="center", va="center", fontsize=14, weight="bold")
    y = 0.88
    for row in aggregate_rows:
        lines = [
            f"Ticker: {row.get('ticker')}",
            f"Trades: {row.get('trades')}",
            f"Hit rate: {row.get('hit_rate')}",
            f"Profit factor: {row.get('profit_factor')}",
            f"Max Drawdown %: {row.get('max_drawdown_pct')}",
            f"Sharpe: {row.get('sharpe')}",
            "-" * 60,
        ]
        for line in lines:
            ax.text(0.08, y, line, fontsize=10, family="monospace")
            y -= 0.04
            if y < 0.08:
                fig.savefig(filename, bbox_inches="tight")
                plt.close(fig)
                fig, ax = plt.subplots(figsize=(8.27, 11.69))
                ax.axis("off")
                y = 0.95
    fig.savefig(filename, bbox_inches="tight")
    plt.close(fig)
    return filename


# ----------------- main runner -----------------
def main_run(
    tickers: Sequence[str],
    start_str: str,
    end_str: str,
    interval_minutes: int = 60,
    optuna_trials: int = DEFAULT_OPTUNA_TRIALS,
):
    start = pd.to_datetime(start_str)
    end = pd.to_datetime(end_str)
    results = []
    for ticker in tickers:
        try:
            result = run_ticker(
                ticker,
                start,
                end,
                interval_minutes=interval_minutes,
                optuna_trials=optuna_trials,
            )
            if result:
                results.append(result)
        except Exception as exc:
            print(f"[err] {ticker} -> {exc}")
    agg = []
    for res in results:
        metrics = res["metrics"]
        row = {"ticker": res["ticker"]}
        row.update(metrics)
        agg.append(row)
    if agg:
        pd.DataFrame(agg).to_csv(OUT_DIR / "aggregate_metrics.csv", index=False)
        pdf_path = OUT_DIR / f"backtest_report_{nowtag()}.pdf"
        make_pdf_report(agg, str(pdf_path))
        summary_text = "<b>Pause BOOM — Backtest Summary</b>\n"
        for a in agg:
            summary_text += (
                f"\nTicker: {a['ticker']} — trades:{a['trades']} — hit:{a.get('hit_rate')} — pf:{a.get('profit_factor')}\n"
            )
        print("[+] Sending Telegram report...")
        send_telegram_text(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, summary_text)
        send_telegram_file(
            TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, str(pdf_path), caption="Backtest PDF — Pause BOOM"
        )
    print("[+++] Done. Outputs in ./outputs")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", nargs="+", default=["MU", "VRT", "DELL"])
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default=datetime.today().strftime("%Y-%m-%d"))
    parser.add_argument("--interval", default=60, type=int)
    parser.add_argument("--optuna_trials", default=DEFAULT_OPTUNA_TRIALS, type=int)
    args = parser.parse_args()
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("[warn] TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set. Telegram reporting disabled.")
    main_run(args.tickers, args.start, args.end, interval_minutes=args.interval, optuna_trials=args.optuna_trials)
