from __future__ import annotations

import math
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests


BASE_URL = "https://api.binance.com"
QUOTE = "USDT"

START_DATE = "2015-01-01"
END_DATE = None

TOP_B = 10
BUFFER_K = 0
LOAD_TOP = TOP_B + BUFFER_K

REFRESH_LAST_DAYS = 5

LOOKBACKS = [10, 20, 30, 60, 90, 150]
VOL_WINDOW_D = 30
TARGET_VOL_ANNUAL = 1.5
LEVERAGE_CAP = 10.0

TC_BPS = 10
TC = TC_BPS / 10_000.0
REBAL_THRESH = 0.20

LIQ_VOL_MEDIAN_USD = 1_000_000.0
LIQ_ABSRET_MEDIAN = 0.005
LIQ_WINDOW_D = 30

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTDIR = os.path.join(BASE_DIR, "output")
CACHEDIR = os.path.join(OUTDIR, "_cache_klines_1d")
UNIVERSE_CSV = os.path.join(DATA_DIR, "monthly_top10_median30d_spot_volume.csv")

os.makedirs(OUTDIR, exist_ok=True)
os.makedirs(CACHEDIR, exist_ok=True)

CONTRIB_YEAR = None
TOP_K_CONTRIB = 5

SESSION = requests.Session()


def norm_dt(x) -> pd.Timestamp:
    ts = pd.to_datetime(x)
    if getattr(ts, "tzinfo", None) is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts.normalize()


def to_ms(dt: pd.Timestamp) -> int:
    dt = pd.to_datetime(dt)
    if getattr(dt, "tzinfo", None) is not None:
        dt = dt.tz_convert("UTC").tz_localize(None)
    return int(dt.value // 1_000_000)


def annualize_vol(daily_std: float) -> float:
    return daily_std * math.sqrt(365.0)


def perf_stats(eq: pd.Series) -> dict:
    eq = eq.dropna()
    if len(eq) < 2:
        return {}

    rets = eq.pct_change().dropna()
    days = max((eq.index[-1] - eq.index[0]).days, 1)
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (365.0 / days) - 1.0
    vol = annualize_vol(rets.std())
    sharpe = (rets.mean() * 365.0) / (rets.std() * math.sqrt(365.0) + 1e-12)
    drawdown = eq / eq.cummax() - 1.0

    return {
        "Start": str(eq.index[0].date()),
        "End": str(eq.index[-1].date()),
        "Days": int(days),
        "CAGR": float(cagr),
        "Vol": float(vol),
        "Sharpe": float(sharpe),
        "MaxDD": float(drawdown.min()),
        "FinalEq": float(eq.iloc[-1]),
    }


def save_txt(path: str, text: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def parse_symbols_cell(x) -> list[str]:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return []

    s = str(x).strip().strip('"').strip("'")
    if not s:
        return []

    if (s.startswith("[") and s.endswith("]")) or (s.startswith("(") and s.endswith(")")):
        s2 = s.strip("[]()").strip()
        if not s2:
            return []
        parts = []
        for item in s2.split(","):
            item = item.strip().strip("'").strip('"')
            if item:
                parts.append(item)
        return parts

    if "|" in s:
        return [item.strip() for item in s.split("|") if item.strip()]

    if "," in s:
        return [item.strip() for item in s.split(",") if item.strip()]

    return [s]


def load_monthly_universe_lagged(csv_path: str, top_b: int) -> dict[pd.Period, list[str]]:
    df = pd.read_csv(csv_path)

    if "month_end" not in df.columns:
        raise ValueError(f"Missing 'month_end' column. Found: {list(df.columns)}")

    if "top_symbols" in df.columns:
        sym_col = "top_symbols"
    elif "symbols" in df.columns:
        sym_col = "symbols"
    else:
        raise ValueError(f"Missing symbols column. Found: {list(df.columns)}")

    df["month_end"] = (
        pd.to_datetime(df["month_end"], utc=True, errors="coerce")
        .dt.tz_convert(None)
        .dt.normalize()
    )
    df = df.dropna(subset=["month_end"]).sort_values("month_end").reset_index(drop=True)

    raw: dict[pd.Period, list[str]] = {}
    for _, row in df.iterrows():
        month = pd.Timestamp(row["month_end"]).to_period("M")
        raw[month] = parse_symbols_cell(row[sym_col])[:top_b]

    lagged: dict[pd.Period, list[str]] = {}
    for month, symbols in raw.items():
        lagged[month + 1] = symbols

    return lagged


def universe_union_symbols(trading_universe: dict[pd.Period, list[str]]) -> list[str]:
    symbols = set()
    for month_symbols in trading_universe.values():
        symbols.update(month_symbols)
    return sorted(symbols)


def select_universe_with_buffer(
    universe_lagged: dict[pd.Period, list[str]],
    month: pd.Period,
    prev_traded: list[str] | None,
    trade_b: int,
    buffer_k: int,
) -> list[str]:
    candidates = universe_lagged.get(month, [])
    if not candidates:
        return []

    top_now = candidates[:trade_b]
    candidate_set = set(candidates)

    kept = []
    if prev_traded:
        for symbol in prev_traded:
            if symbol in candidate_set:
                kept.append(symbol)

    kept_set = set(kept)
    fills = [symbol for symbol in top_now if symbol not in kept_set]
    out = kept + fills
    return out[:trade_b]


def binance_get(path: str, params: dict | None = None, retries: int = 6):
    url = BASE_URL + path
    last_err = None

    for attempt in range(retries):
        try:
            response = SESSION.get(url, params=params, timeout=30)
            if response.status_code in (418, 429):
                time.sleep(1.0 + 0.5 * attempt)
                continue
            response.raise_for_status()
            return response.json()
        except Exception as e:
            last_err = e
            time.sleep(0.8 + 0.5 * attempt)

    raise RuntimeError(f"Binance GET failed: {path} | params={params} | err={repr(last_err)}")


def fetch_klines_1d(symbol: str, start: pd.Timestamp, end: pd.Timestamp | None) -> pd.DataFrame:
    start = norm_dt(start)
    end = norm_dt(end) if end is not None else norm_dt(pd.Timestamp.utcnow())

    cache_path = os.path.join(CACHEDIR, f"{symbol}_1d.csv")

    if os.path.exists(cache_path):
        df = pd.read_csv(cache_path, parse_dates=["date"])
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
        df = df.set_index("date").sort_index()
    else:
        df = pd.DataFrame()

    if not df.empty and df.index.min() <= start and df.index.max() >= end:
        return df.loc[start:end]

    rows = []
    if not df.empty:
        current = max(start, df.index.max() - pd.Timedelta(days=REFRESH_LAST_DAYS))
    else:
        current = start

    end_ms = to_ms(end + pd.Timedelta(days=1)) - 1

    while True:
        params = {
            "symbol": symbol,
            "interval": "1d",
            "startTime": to_ms(current),
            "endTime": end_ms,
            "limit": 1000,
        }
        data = binance_get("/api/v3/klines", params=params)
        if not data:
            break

        for row in data:
            day = pd.to_datetime(int(row[0]), unit="ms", utc=True).tz_convert(None).normalize()
            rows.append(
                {
                    "date": day,
                    "open": float(row[1]),
                    "high": float(row[2]),
                    "low": float(row[3]),
                    "close": float(row[4]),
                    "volume": float(row[5]),
                    "quote_volume": float(row[7]),
                }
            )

        last_open = int(data[-1][0])
        last_day = pd.to_datetime(last_open, unit="ms", utc=True).tz_convert(None).normalize()
        current = last_day + pd.Timedelta(days=1)

        if len(data) < 1000 or current > end:
            break

        time.sleep(0.10)

    if rows:
        new_df = (
            pd.DataFrame(rows)
            .drop_duplicates(subset=["date"])
            .set_index("date")
            .sort_index()
        )
        df = pd.concat([df, new_df], axis=0).sort_index()
        df = df[~df.index.duplicated(keep="last")]
        df.to_csv(cache_path, index_label="date")

    return df.loc[start:end] if not df.empty else df


def compute_combo_paper_longonly(df: pd.DataFrame) -> pd.DataFrame:
    px = df["close"].astype(float).copy()
    ret = px.pct_change().fillna(0.0)

    sigma = ret.rolling(VOL_WINDOW_D, min_periods=max(20, VOL_WINDOW_D // 3)).std() * math.sqrt(365.0)

    don_up = {n: px.rolling(n).max().shift(1) for n in LOOKBACKS}
    don_dn = {n: px.rolling(n).min().shift(1) for n in LOOKBACKS}
    don_mid = {n: 0.5 * (don_up[n] + don_dn[n]) for n in LOOKBACKS}

    idx = px.index
    weights = pd.Series(0.0, index=idx)

    positions = {n: 0 for n in LOOKBACKS}
    trailing_stop = {n: np.nan for n in LOOKBACKS}
    frac_series = pd.Series(0.0, index=idx)

    for i in range(1, len(idx)):
        close_t = float(px.iloc[i])
        sigma_t = float(sigma.iloc[i]) if not np.isnan(sigma.iloc[i]) else np.nan

        for n in LOOKBACKS:
            up = float(don_up[n].iloc[i]) if not np.isnan(don_up[n].iloc[i]) else np.nan
            mid = float(don_mid[n].iloc[i]) if not np.isnan(don_mid[n].iloc[i]) else np.nan

            if np.isnan(up) or np.isnan(mid):
                continue

            if positions[n] == 0:
                if close_t >= up - 1e-12:
                    positions[n] = 1
                    trailing_stop[n] = mid
            else:
                if not np.isnan(trailing_stop[n]) and close_t <= float(trailing_stop[n]) + 1e-12:
                    positions[n] = 0
                    trailing_stop[n] = np.nan
                else:
                    trailing_stop[n] = max(float(trailing_stop[n]), mid) if not np.isnan(trailing_stop[n]) else mid

        prev_weight = float(weights.iloc[i - 1])
        prev_frac = float(frac_series.iloc[i - 1])

        frac = float(np.mean([1.0 if positions[n] == 1 else 0.0 for n in LOOKBACKS]))
        frac_series.iloc[i] = frac

        if np.isnan(sigma_t) or sigma_t <= 0:
            leverage = 0.0
        else:
            leverage = min(TARGET_VOL_ANNUAL / sigma_t, LEVERAGE_CAP)

        target_weight = frac * leverage
        new_weight = target_weight

        if prev_weight > 0 and target_weight > 0 and abs(frac - prev_frac) < 1e-12:
            if abs(target_weight - prev_weight) <= REBAL_THRESH * abs(prev_weight):
                new_weight = prev_weight

        weights.iloc[i] = new_weight

    turnover = (weights - weights.shift(1)).fillna(0.0).abs()
    ret_gross = weights.shift(1).fillna(0.0) * ret
    ret_net = ret_gross - TC * turnover

    out = pd.DataFrame(
        {
            "w": weights,
            "ret_net": ret_net,
            "turnover": turnover,
        },
        index=idx,
    ).dropna()

    return out


def run_program_section7(
    panel: dict[str, pd.DataFrame],
    combo: dict[str, pd.DataFrame],
    btc_df: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    trading_universe: dict[pd.Period, list[str]],
    trade_b: int,
    buffer_k: int,
) -> tuple[pd.Series, pd.Series, pd.DataFrame, pd.DataFrame]:
    dates = btc_df.loc[start:end].index
    dates = pd.DatetimeIndex([norm_dt(d) for d in dates]).unique().sort_values()

    all_symbols = sorted(panel.keys())
    weights = pd.DataFrame(0.0, index=dates, columns=all_symbols)

    equity = pd.Series(index=dates, dtype=float)
    equity.iloc[0] = 1.0

    by_month: dict[pd.Period, list[pd.Timestamp]] = {}
    for d in dates:
        by_month.setdefault(d.to_period("M"), []).append(d)

    months = sorted(by_month.keys())
    last_equity = 1.0
    prev_traded_symbols: list[str] | None = None

    for month in months:
        month_days = pd.DatetimeIndex(by_month[month]).sort_values()

        selected = select_universe_with_buffer(
            universe_lagged=trading_universe,
            month=month,
            prev_traded=prev_traded_symbols,
            trade_b=trade_b,
            buffer_k=buffer_k,
        )
        selected = [s for s in selected if s in panel and s in combo]
        prev_traded_symbols = list(selected)

        if len(selected) == 0:
            for d in month_days:
                equity.loc[d] = last_equity
            continue

        sleeve_aum = {s: last_equity / len(selected) for s in selected}
        removed = {s: False for s in selected}

        liquidity_ok: dict[str, pd.Series] = {}
        for symbol in selected:
            df = panel[symbol]
            close = df["close"].astype(float)
            quote_vol = df.get("quote_volume", pd.Series(index=df.index, data=np.nan)).astype(float)
            abs_ret = close.pct_change().fillna(0.0).abs()

            med_quote_vol = quote_vol.rolling(
                LIQ_WINDOW_D,
                min_periods=max(10, LIQ_WINDOW_D // 3),
            ).median().shift(1)

            med_abs_ret = abs_ret.rolling(
                LIQ_WINDOW_D,
                min_periods=max(10, LIQ_WINDOW_D // 3),
            ).median().shift(1)

            ok = (med_quote_vol >= LIQ_VOL_MEDIAN_USD) & (med_abs_ret >= LIQ_ABSRET_MEDIAN)
            liquidity_ok[symbol] = ok.reindex(month_days).fillna(False)

        for i, d in enumerate(month_days):
            if i == 0:
                equity.loc[d] = last_equity
                continue

            d_prev = month_days[i - 1]
            portfolio_prev = float(sum(sleeve_aum.values()))
            if portfolio_prev <= 0:
                portfolio_prev = 1e-12

            portfolio_now = 0.0

            for symbol in selected:
                if not removed[symbol] and not bool(liquidity_ok[symbol].loc[d]):
                    removed[symbol] = True

                if removed[symbol]:
                    portfolio_now += sleeve_aum[symbol]
                    continue

                if d not in combo[symbol].index or d_prev not in combo[symbol].index:
                    portfolio_now += sleeve_aum[symbol]
                    continue

                sleeve_weight = float(combo[symbol].loc[d_prev, "w"])
                portfolio_weight = (sleeve_aum[symbol] / portfolio_prev) * sleeve_weight
                weights.loc[d, symbol] = portfolio_weight

                sleeve_return = float(combo[symbol].loc[d, "ret_net"])
                sleeve_aum[symbol] *= (1.0 + sleeve_return)
                portfolio_now += sleeve_aum[symbol]

            equity.loc[d] = portfolio_now

        last_equity = float(equity.loc[month_days[-1]])

    returns = equity.pct_change().fillna(0.0)
    weights = weights.reindex(returns.index).fillna(0.0)

    portfolio_stats = pd.DataFrame(
        {
            "net": weights.sum(axis=1),
            "gross": weights.abs().sum(axis=1),
            "n_active": (weights.abs() > 0).sum(axis=1),
        },
        index=weights.index,
    )

    return equity, returns, weights, portfolio_stats


def btc_scaled_to_match(btc_px: pd.Series, target_ret: pd.Series) -> pd.Series:
    btc_ret = btc_px.pct_change().fillna(0.0)
    btc_vol = annualize_vol(btc_ret.std())
    target_vol = annualize_vol(target_ret.std())

    scale = 0.0 if btc_vol <= 0 else target_vol / btc_vol
    equity = (1.0 + btc_ret * scale).cumprod()
    equity.iloc[0] = 1.0
    return equity


def plot_equity(eq: pd.Series, btc_eq: pd.Series, path: str, log: bool = False, label: str = "Strategy") -> None:
    plt.figure(figsize=(12, 6))
    plt.plot(eq.index, eq.values, label=label)
    plt.plot(btc_eq.index, btc_eq.values, label="BTC B&H (scaled to strat vol)")
    plt.title("Equity Curve" + (" (log)" if log else ""))
    plt.xlabel("Date")
    plt.ylabel("Equity")
    if log:
        plt.yscale("log")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_rolling_vol(eq: pd.Series, btc_eq: pd.Series, path: str, window: int = 365) -> None:
    r1 = eq.pct_change().fillna(0.0)
    r2 = btc_eq.pct_change().fillna(0.0)
    v1 = r1.rolling(window).std() * math.sqrt(365.0)
    v2 = r2.rolling(window).std() * math.sqrt(365.0)

    plt.figure(figsize=(12, 5))
    plt.plot(v1.index, v1.values, label="Strategy")
    plt.plot(v2.index, v2.values, label="BTC")
    plt.title(f"Rolling {window}d Vol (ann.)")
    plt.ylabel("Vol")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_drawdowns(eq: pd.Series, btc_eq: pd.Series, path: str) -> None:
    dd1 = eq / eq.cummax() - 1.0
    dd2 = btc_eq / btc_eq.cummax() - 1.0

    plt.figure(figsize=(12, 5))
    plt.plot(dd1.index, dd1.values, label="Strategy DD")
    plt.plot(dd2.index, dd2.values, label="BTC DD")
    plt.title("Drawdowns")
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def bar_with_pct_labels(labels, values, title: str, path: str, figsize=(14, 5)) -> None:
    plt.figure(figsize=figsize)
    bars = plt.bar(range(len(values)), values)
    plt.title(title)
    plt.xticks(range(len(values)), labels, rotation=45, ha="right")
    plt.ylabel("Return")

    for bar, value in zip(bars, values):
        if np.isnan(value):
            continue
        y = bar.get_height()
        offset = 0.003 if y >= 0 else -0.01
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            y + offset,
            f"{value * 100:.1f}%",
            ha="center",
            va="bottom" if y >= 0 else "top",
            fontsize=9,
        )

    plt.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_monthly_returns(eq: pd.Series, path: str) -> None:
    monthly_eq = eq.resample("M").last()
    monthly_ret = monthly_eq.pct_change().dropna()

    labels = [d.strftime("%Y-%m") for d in monthly_ret.index]
    values = monthly_ret.values
    bar_with_pct_labels(labels, values, "Monthly Returns", path, figsize=(16, 5))


def plot_yearly_returns(eq: pd.Series, path: str) -> None:
    yearly_eq = eq.resample("Y").last()
    yearly_ret = yearly_eq.pct_change().dropna()

    labels = [d.strftime("%Y") for d in yearly_ret.index]
    values = yearly_ret.values
    bar_with_pct_labels(labels, values, "Yearly Returns", path, figsize=(12, 5))


def pick_last_complete_year(eq: pd.Series) -> int | None:
    if eq.empty:
        return None

    last_date = eq.index.max()
    if not (last_date.month == 12 and last_date.day == 31):
        return last_date.year - 1
    return last_date.year


def plot_top5_coin_contribution(
    year: int,
    weights: pd.DataFrame,
    panel: dict[str, pd.DataFrame],
    eq: pd.Series,
    btc_close: pd.Series,
    out_path: str,
    top_k: int = 5,
) -> None:
    start = pd.Timestamp(year=year, month=1, day=1)
    end = pd.Timestamp(year=year, month=12, day=31)

    weights_year = weights.loc[(weights.index >= start) & (weights.index <= end)].copy()
    if weights_year.empty:
        print(f"[WARN] No weights in year {year}; skipping contribution plot.")
        return

    weights_lag = weights_year.shift(1).fillna(0.0)

    contributions = {}
    for symbol in weights_year.columns:
        if symbol not in panel:
            continue
        px = panel[symbol]["close"].astype(float).reindex(weights_year.index)
        returns = px.pct_change().fillna(0.0)
        contributions[symbol] = float((weights_lag[symbol] * returns).sum())

    if not contributions:
        print(f"[WARN] No contribution data for year {year}; skipping.")
        return

    contrib_series = pd.Series(contributions).sort_values(ascending=False)
    top = contrib_series.head(top_k)

    eq_year = eq.loc[(eq.index >= start) & (eq.index <= end)]
    strat_ret = float(eq_year.iloc[-1] / eq_year.iloc[0] - 1.0) if len(eq_year) >= 2 else np.nan

    btc_year = btc_close.loc[(btc_close.index >= start) & (btc_close.index <= end)]
    btc_ret = float(btc_year.iloc[-1] / btc_year.iloc[0] - 1.0) if len(btc_year) >= 2 else np.nan

    labels = list(top.index) + ["BTC B&H"]
    values = list(top.values) + [btc_ret]

    plt.figure(figsize=(12, 4.8))
    bars = plt.bar(range(len(values)), values)
    plt.title(f"Top {top_k} Coin Contributions — {year} (plus BTC B&H)")
    plt.xticks(range(len(values)), labels, rotation=45, ha="right")
    plt.ylabel("Total Return / Contribution")

    for bar, value in zip(bars, values):
        if np.isnan(value):
            continue
        y = bar.get_height()
        offset = 0.005 if y >= 0 else -0.02
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            y + offset,
            f"{value * 100:.1f}%",
            ha="center",
            va="bottom" if y >= 0 else "top",
            fontsize=10,
        )

    if not np.isnan(strat_ret):
        plt.suptitle(f"Strategy total return in {year}: {strat_ret * 100:.1f}%", y=0.98, fontsize=10)

    plt.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def main() -> None:
    if LOAD_TOP < TOP_B:
        raise ValueError("LOAD_TOP must be >= TOP_B")

    start = norm_dt(START_DATE)
    end = norm_dt(END_DATE) if END_DATE else norm_dt(pd.Timestamp.utcnow())

    print("[INFO] Loading lagged monthly universe...")
    trading_universe = load_monthly_universe_lagged(UNIVERSE_CSV, top_b=LOAD_TOP)
    union_symbols = universe_union_symbols(trading_universe)

    if "BTCUSDT" not in union_symbols:
        union_symbols = ["BTCUSDT"] + union_symbols

    print(f"[INFO] Universe months loaded: {len(trading_universe)}")
    print(f"[INFO] Unique symbols across all months: {len(union_symbols)}")

    print("[INFO] Downloading daily spot data...")
    panel: dict[str, pd.DataFrame] = {}
    for i, symbol in enumerate(union_symbols, 1):
        try:
            df = fetch_klines_1d(symbol, start, end)
            if df is None or df.empty or df["close"].isna().all():
                continue
            panel[symbol] = df
        except Exception as e:
            print(f"[WARN] {symbol} failed: {repr(e)}")

        if i % 25 == 0:
            print(f"[DL] {i}/{len(union_symbols)}")

    if "BTCUSDT" not in panel:
        raise RuntimeError("BTCUSDT missing; cannot run benchmark/backtest.")

    btc_df = panel["BTCUSDT"]
    end_final = min(end, btc_df.index.max())

    print("[INFO] Building per-asset combo series...")
    combo: dict[str, pd.DataFrame] = {}
    for symbol, df in panel.items():
        try:
            combo[symbol] = compute_combo_paper_longonly(df)
        except Exception as e:
            print(f"[WARN] Combo failed for {symbol}: {repr(e)}")

    print("[RUN] Running Section 7 program...")
    eq, ret, weights, portfolio_stats = run_program_section7(
        panel=panel,
        combo=combo,
        btc_df=btc_df,
        start=start,
        end=end_final,
        trading_universe=trading_universe,
        trade_b=TOP_B,
        buffer_k=BUFFER_K,
    )

    weights.to_csv(os.path.join(OUTDIR, "weights_daily_portfolio_raw.csv"), index_label="date")
    portfolio_stats.to_csv(os.path.join(OUTDIR, "portfolio_daily_stats.csv"), index_label="date")
    print("[OUT] weights_daily_portfolio_raw.csv")
    print("[OUT] portfolio_daily_stats.csv")

    btc_close = btc_df["close"].loc[eq.index]
    btc_eq = btc_scaled_to_match(btc_close, ret)

    stats_strategy = perf_stats(eq)
    stats_btc = perf_stats(btc_eq)

    def block(name: str, stats: dict) -> str:
        return (
            f"{name}\n"
            f"  CAGR    : {stats['CAGR'] * 100:7.2f}%\n"
            f"  Vol     : {stats['Vol'] * 100:7.2f}%\n"
            f"  Sharpe  : {stats['Sharpe']:7.2f}\n"
            f"  MaxDD   : {stats['MaxDD'] * 100:7.2f}%\n"
            f"  FinalEq : {stats['FinalEq']:10.4f}\n"
        )

    lines = []
    lines.append("==== Catching Crypto Trends — Section 7 Diversified Program ====\n\n")
    lines.append(f"UNIVERSE_CSV       : {UNIVERSE_CSV}\n")
    lines.append("Universe rule      : Trading month M uses the universe from month M-1\n")
    lines.append(f"Universe select    : TOP_B={TOP_B}, BUFFER_K={BUFFER_K}, candidate size={LOAD_TOP}\n")
    lines.append(f"Period             : {stats_strategy.get('Start')} -> {stats_strategy.get('End')} ({stats_strategy.get('Days')} days)\n\n")
    lines.append("Per-asset sizing   : w_t = frac_t * min(TARGET_VOL_ANNUAL / sigma_t, LEVERAGE_CAP)\n")
    lines.append(f"TARGET_VOL_ANNUAL  : {TARGET_VOL_ANNUAL:.0%}\n")
    lines.append(f"LEVERAGE_CAP       : {LEVERAGE_CAP:.2f}x\n")
    lines.append(f"VOL_WINDOW_D       : {VOL_WINDOW_D}d\n")
    lines.append(f"Lookbacks          : {LOOKBACKS}\n")
    lines.append(f"Costs              : {TC_BPS} bps on turnover\n")
    lines.append(f"Rebal threshold    : {REBAL_THRESH:.0%}\n")
    lines.append(
        f"Liquidity exits    : median quote_volume >= ${LIQ_VOL_MEDIAN_USD:,.0f} "
        f"and median abs return >= {LIQ_ABSRET_MEDIAN * 100:.2f}%\n\n"
    )
    lines.append(block("Trend Program", stats_strategy))
    lines.append("\n")
    lines.append(block("BTC B&H (scaled to strategy vol)", stats_btc))
    lines.append("\n")
    lines.append(f"Alpha proxy (CAGR diff): {(stats_strategy['CAGR'] - stats_btc['CAGR']) * 100:7.2f}%\n")

    save_txt(os.path.join(OUTDIR, "results.txt"), "".join(lines))
    print("[OUT] results.txt")

    plot_equity(eq, btc_eq, os.path.join(OUTDIR, "equity_linear_vs_btc.png"), log=False, label="Trend Program")
    plot_equity(eq, btc_eq, os.path.join(OUTDIR, "equity_log_vs_btc.png"), log=True, label="Trend Program")
    plot_drawdowns(eq, btc_eq, os.path.join(OUTDIR, "drawdowns_vs_btc.png"))
    plot_rolling_vol(eq, btc_eq, os.path.join(OUTDIR, "rolling_vol_365d_vs_btc.png"), window=365)

    plot_monthly_returns(eq, os.path.join(OUTDIR, "monthly_returns.png"))
    print("[OUT] monthly_returns.png")

    plot_yearly_returns(eq, os.path.join(OUTDIR, "yearly_returns.png"))
    print("[OUT] yearly_returns.png")

    contrib_year = CONTRIB_YEAR if CONTRIB_YEAR is not None else pick_last_complete_year(eq)
    if contrib_year is not None:
        out_path = os.path.join(OUTDIR, f"top{TOP_K_CONTRIB}_coin_contrib_{contrib_year}.png")
        plot_top5_coin_contribution(
            year=contrib_year,
            weights=weights,
            panel=panel,
            eq=eq,
            btc_close=btc_close,
            out_path=out_path,
            top_k=TOP_K_CONTRIB,
        )
        print(f"[OUT] {os.path.basename(out_path)}")
    else:
        print("[WARN] Could not determine contribution year.")

    print("[DONE]")


if __name__ == "__main__":
    main()
