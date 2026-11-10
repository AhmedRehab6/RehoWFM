import io
import json
import math
import os
from datetime import datetime

import numpy as np
import pandas as pd
from langchain_core.tools import tool

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False

DATA_DIR = "data"      # put your Excel/CSV files here
OUTPUT_DIR = "output"  # forecasts, plans, charts, DAX/M code go here
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ----------------------------------------------------------------- helpers
def _find(file_name: str) -> str:
    for d in (DATA_DIR, OUTPUT_DIR):
        p = os.path.join(d, file_name)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"{file_name} not found in {DATA_DIR}/ or {OUTPUT_DIR}/")


def _load(file_name: str, sheet_name: str = "") -> pd.DataFrame:
    path = _find(file_name)
    if path.lower().endswith((".xlsx", ".xlsm", ".xls")):
        return pd.read_excel(path, sheet_name=sheet_name or 0)
    return pd.read_csv(path)


def _save_df(df: pd.DataFrame, output_name: str) -> str:
    path = os.path.join(OUTPUT_DIR, output_name)
    if output_name.lower().endswith(".csv"):
        df.to_csv(path, index=False)
    else:
        df.to_excel(path, index=False)
    return path


def _pct(x) -> float:
    """Accept 0.3 or 30 and return 0.3."""
    x = float(x)
    return x / 100.0 if x > 1 else x


# ------------------------------------------------------------ Erlang C core
def _erlang_c(agents: int, traffic: float) -> float:
    """Probability a contact has to wait. traffic is in Erlangs."""
    if agents <= traffic:
        return 1.0
    b = 1.0  # Erlang B, computed iteratively (numerically stable)
    for k in range(1, agents + 1):
        b = (traffic * b) / (k + traffic * b)
    return b / (1 - (traffic / agents) * (1 - b))


def _service_level(agents: int, traffic: float, aht: float, target_s: float) -> float:
    if agents <= traffic:
        return 0.0
    pw = _erlang_c(agents, traffic)
    return max(0.0, 1 - pw * math.exp(-(agents - traffic) * target_s / aht))


def _asa(agents: int, traffic: float, aht: float) -> float:
    if agents <= traffic:
        return float("inf")
    return _erlang_c(agents, traffic) * aht / (agents - traffic)


def _staff(volume, interval_min, aht, sl_target, sl_seconds, max_occ) -> dict:
    traffic = volume * aht / (interval_min * 60.0)
    if volume <= 0 or traffic <= 0:
        return {"traffic": 0.0, "agents": 0, "sl": 1.0, "occ": 0.0, "asa": 0.0}
    n = max(1, math.ceil(traffic))
    sl, occ = 0.0, 1.0
    while n < 5000:
        sl = _service_level(n, traffic, aht, sl_seconds)
        occ = traffic / n
        if sl >= sl_target and occ <= max_occ:
            break
        n += 1
    return {"traffic": traffic, "agents": n, "sl": sl, "occ": occ, "asa": _asa(n, traffic, aht)}


# --------------------------------------------------------- forecast methods
FREQ_ALIASES = {
    "d": "D", "day": "D", "daily": "D",
    "w": "W", "week": "W", "weekly": "W",
    "m": "MS", "month": "MS", "monthly": "MS",
    "h": "h", "hour": "h", "hourly": "h",
}
SEASON_DEFAULT = {"D": 7, "W": 52, "MS": 12, "h": 24, "30min": 48, "15min": 96}


def _fc_seasonal_naive(y, h, m):
    y = np.asarray(y, float)
    if m < 1 or len(y) < m:
        return np.repeat(y.mean(), h)
    return np.array([y[len(y) - m + (i % m)] for i in range(h)])


def _fc_moving_average(y, h, m):
    y = np.asarray(y, float)
    w = max(1, min(m, len(y)))
    return np.repeat(y[-w:].mean(), h)


def _fc_trend_seasonal(y, h, m):
    """Multiplicative seasonal indices + linear trend on the deseasonalised series."""
    y = np.asarray(y, float)
    n = len(y)
    x_future = np.arange(n, n + h)
    if m <= 1 or n < 2 * m:
        if n >= 2:
            slope, intercept = np.polyfit(np.arange(n), y, 1)
        else:
            slope, intercept = 0.0, y.mean()
        return np.maximum(intercept + slope * x_future, 0)
    cycles = n // m
    s0 = n - cycles * m
    mat = y[s0:].reshape(cycles, m)
    cm = mat.mean(axis=1, keepdims=True)
    cm[cm == 0] = 1
    idx = (mat / cm).mean(axis=0)
    idx = idx / idx.mean() if idx.mean() else np.ones(m)
    seas = idx[(np.arange(n) - s0) % m]
    seas[seas == 0] = 1
    des = y / seas
    w = min(n, 4 * m)  # fit the trend on recent cycles only
    slope, intercept = np.polyfit(np.arange(n - w, n), des[n - w:], 1)
    fpos = (x_future - s0) % m
    return np.maximum((intercept + slope * x_future) * idx[fpos], 0)


def _hw_run(y, m, a, b, g):
    level = y[:m].mean()
    trend = (y[m:2 * m].mean() - y[:m].mean()) / m
    season = (y[:m] / level).tolist()
    sse = 0.0
    for t in range(m, len(y)):
        s = season[t % m] or 1.0
        pred = (level + trend) * s
        sse += (y[t] - pred) ** 2
        new_level = a * (y[t] / s) + (1 - a) * (level + trend)
        trend = b * (new_level - level) + (1 - b) * trend
        season[t % m] = g * (y[t] / new_level if new_level else 1.0) + (1 - g) * s
        level = new_level
    return sse, level, trend, season


def _fc_holt_winters(y, h, m):
    """Multiplicative Holt-Winters with a small parameter grid search."""
    y = np.asarray(y, float)
    if m < 2 or len(y) < 2 * m or (y <= 0).any():
        return _fc_trend_seasonal(y, h, m)
    best = None
    for a in (0.1, 0.3, 0.5):
        for b in (0.01, 0.05, 0.1):
            for g in (0.1, 0.3, 0.5):
                res = _hw_run(y, m, a, b, g)
                if best is None or res[0] < best[0]:
                    best = res
    _, level, trend, season = best
    n = len(y)
    fc = [(level + (k + 1) * trend) * season[(n + k) % m] for k in range(h)]
    return np.maximum(np.array(fc), 0)


FORECAST_METHODS = {
    "seasonal_naive": _fc_seasonal_naive,
    "moving_average": _fc_moving_average,
    "trend_seasonal": _fc_trend_seasonal,
    "holt_winters": _fc_holt_winters,
}


def _metrics(actual, fc) -> dict:
    actual = np.asarray(actual, float)
    fc = np.asarray(fc, float)
    denom = np.abs(actual).sum()
    nz = actual != 0
    return {
        "WAPE_%": round(np.abs(actual - fc).sum() / denom * 100, 2) if denom else np.nan,
        "MAPE_%": round(np.abs((actual[nz] - fc[nz]) / actual[nz]).mean() * 100, 2) if nz.any() else np.nan,
        "Bias_%": round((fc.sum() - actual.sum()) / denom * 100, 2) if denom else np.nan,
    }


# ------------------------------------------------------------- basic tools
@tool
def list_data_files() -> str:
    """List the Excel and CSV files available in the data and output folders."""
    out = []
    for d in (DATA_DIR, OUTPUT_DIR):
        for f in os.listdir(d):
            if f.lower().endswith((".xlsx", ".xlsm", ".xls", ".csv")):
                out.append(f"{d}/{f}")
    return "\n".join(out) if out else "No data files found."


@tool
def inspect_data(file_name: str, sheet_name: str = "") -> str:
    """Show sheets, shape, column types, null counts, and the first rows of a file.
    Always call this before analyzing a file."""
    try:
        out = []
        if _find(file_name).lower().endswith((".xlsx", ".xlsm", ".xls")):
            out.append(f"Sheets: {pd.ExcelFile(_find(file_name)).sheet_names}")
        df = _load(file_name, sheet_name)
        buf = io.StringIO()
        df.info(buf=buf)
        out.append(f"Shape: {df.shape}")
        out.append(buf.getvalue())
        out.append(f"Nulls per column:\n{df.isna().sum().to_string()}")
        out.append(f"First rows:\n{df.head(5).to_string()}")
        return "\n".join(out)
    except Exception as e:
        return f"Error: {e}"


@tool
def run_pandas_code(file_name: str, code: str, sheet_name: str = "", output_name: str = "") -> str:
    """Run pandas code on a file. The data is available as `df`; pandas is `pd`, numpy is `np`.
    Assign what you want returned to a variable named `result`.
    If output_name is given (e.g. 'clean_sales.xlsx'), `result` is saved as a
    DataFrame to the output folder, ready to import into Power BI."""
    try:
        df = _load(file_name, sheet_name)
        scope = {"df": df, "pd": pd, "np": np}
        exec(code, scope)  # NOTE: runs arbitrary code, use only on your own machine
        result = scope.get("result", "No `result` variable was set.")
        if output_name and isinstance(result, pd.DataFrame):
            path = _save_df(result, output_name)
            return f"Saved {len(result)} rows to {path}\n{result.head(10).to_string()}"
        return result.to_string() if hasattr(result, "to_string") else str(result)
    except Exception as e:
        return f"Error: {e}"


@tool
def create_chart(file_name: str, chart_type: str, x: str, y: str, title: str = "", sheet_name: str = "") -> str:
    """Create a chart (chart_type: bar, line, scatter, hist) from columns x and y
    and save it as a PNG in the output folder."""
    if not HAS_MPL:
        return "matplotlib is not installed, so charts are unavailable."
    try:
        df = _load(file_name, sheet_name)
        fig, ax = plt.subplots(figsize=(9, 5))
        if chart_type == "hist":
            ax.hist(df[x].dropna(), bins=20)
        elif chart_type == "scatter":
            ax.scatter(df[x], df[y])
        elif chart_type == "line":
            ax.plot(df[x], df[y])
        else:
            df.plot(kind="bar", x=x, y=y, ax=ax)
        ax.set_title(title or f"{y} by {x}")
        fig.tight_layout()
        path = os.path.join(OUTPUT_DIR, f"chart_{datetime.now():%Y%m%d_%H%M%S}.png")
        fig.savefig(path)
        plt.close(fig)
        return f"Chart saved to {path}"
    except Exception as e:
        return f"Error: {e}"


@tool
def save_power_bi_code(code: str, kind: str, name: str) -> str:
    """Save generated Power BI code to a file. kind is 'dax' (measures/columns)
    or 'm' (Power Query). name is the file name without extension."""
    ext = "dax" if kind.lower() == "dax" else "pq"
    path = os.path.join(OUTPUT_DIR, f"{name}.{ext}")
    with open(path, "w", encoding="utf-8") as f:
        f.write(code)
    return f"Saved to {path}. Paste into Power BI (DAX: New measure; M: Advanced Editor)."


@tool
def save_report(text: str, filename: str = "analysis_report.txt") -> str:
    """Save an analysis summary to a text file in the output folder."""
    path = os.path.join(OUTPUT_DIR, filename)
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"--- {datetime.now():%Y-%m-%d %H:%M:%S} ---\n{text}\n\n")
    return f"Saved to {path}"


# --------------------------------------------------------------- WFM tools
@tool
def forecast_volume(
    file_name: str,
    date_col: str,
    value_col: str,
    periods: int,
    freq: str = "D",
    method: str = "auto",
    season_length: int = 0,
    sheet_name: str = "",
    output_name: str = "forecast.xlsx",
) -> str:
    """Forecast contact volume (or any time series) for the next `periods` periods.
    freq: D (daily), W (weekly), MS (monthly), h (hourly), 30min, 15min.
    method: auto (backtests all methods and picks the lowest WAPE), seasonal_naive,
    moving_average, trend_seasonal, holt_winters. season_length 0 = automatic
    (7 for daily, 52 weekly, 12 monthly). Returns data-quality notes, backtest
    accuracy, the forecast with 80% range, and saves it to the output folder."""
    try:
        df = _load(file_name, sheet_name)
        df[date_col] = pd.to_datetime(df[date_col])
        df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
        f_use = FREQ_ALIASES.get(freq.strip().lower(), freq.strip())
        s = df.set_index(date_col)[value_col].sort_index().resample(f_use).sum(min_count=1)
        gaps = int(s.isna().sum())
        s = s.interpolate(limit_direction="both")
        y = s.values.astype(float)
        n = len(y)
        if n < 6:
            return "Error: need at least 6 periods of history to forecast."
        m = int(season_length) if season_length else SEASON_DEFAULT.get(f_use, 7)

        # data quality: outliers vs rolling median (flagged, not removed)
        med = pd.Series(y).rolling(max(3, min(m, n)), center=True, min_periods=1).median().values
        resid = y - med
        mad = np.median(np.abs(resid - np.median(resid))) * 1.4826
        n_out = int((np.abs(resid) > 3.5 * mad).sum()) if mad > 0 else 0

        # backtest on the last bt periods
        bt = min(int(periods), max(1, n // 4))
        train, test = y[:-bt], y[-bt:]
        results = {}
        for name, fn in FORECAST_METHODS.items():
            try:
                if name == "holt_winters" and (len(train) < 2 * m or (train <= 0).any()):
                    continue
                if name == "seasonal_naive" and len(train) < m:
                    continue
                f = fn(train, bt, m)
                results[name] = (f, _metrics(test, f))
            except Exception:
                continue
        if not results:
            return "Error: not enough data to backtest any method."

        def score(k):
            return np.nan_to_num(results[k][1]["WAPE_%"], nan=1e9)

        chosen = method if method in results else min(results, key=score)
        note = "" if method in ("auto", chosen) else f" (requested '{method}' not usable, used best available)"

        fc = FORECAST_METHODS[chosen](y, int(periods), m)
        errs = test - results[chosen][0]
        sigma = float(errs.std()) if len(errs) > 1 else 0.1 * float(y.mean())
        z = 1.2816  # 80% interval
        idx = pd.date_range(start=s.index[-1], periods=int(periods) + 1, freq=f_use)[1:]
        out = pd.DataFrame({
            "period": idx,
            "forecast": np.round(fc, 1),
            "lower_80": np.round(np.maximum(fc - z * sigma, 0), 1),
            "upper_80": np.round(fc + z * sigma, 1),
            "method": chosen,
        })
        path = _save_df(out, output_name)

        chart_msg = ""
        if HAS_MPL:
            try:
                k = min(n, 3 * m)
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.plot(s.index[-k:], y[-k:], label="history")
                ax.plot(out["period"], out["forecast"], label=f"forecast ({chosen})")
                ax.fill_between(out["period"], out["lower_80"], out["upper_80"], alpha=0.2, label="80% range")
                ax.legend()
                ax.set_title(f"{value_col} forecast")
                fig.tight_layout()
                cpath = os.path.join(OUTPUT_DIR, f"chart_forecast_{datetime.now():%Y%m%d_%H%M%S}.png")
                fig.savefig(cpath)
                plt.close(fig)
                chart_msg = f"\nChart: {cpath}"
            except Exception:
                pass

        bt_table = pd.DataFrame({k: v[1] for k, v in results.items()}).T
        return (
            f"History: {s.index[0].date()} to {s.index[-1].date()} ({n} periods, season length {m}). "
            f"Gaps filled by interpolation: {gaps}. Possible outliers flagged: {n_out}.\n"
            f"Backtest on last {bt} periods:\n{bt_table.to_string()}\n"
            f"Chosen method: {chosen}{note}\n"
            f"Forecast ({periods} periods):\n{out.head(40).to_string(index=False)}\n"
            f"Saved to {path}{chart_msg}"
        )
    except Exception as e:
        return f"Error: {e}"


@tool
def forecast_accuracy(file_name: str, actual_col: str, forecast_col: str, sheet_name: str = "") -> str:
    """Measure forecast accuracy from a file with actual and forecast columns.
    Returns WAPE, MAPE, bias, MAE and RMSE."""
    try:
        df = _load(file_name, sheet_name)
        a = pd.to_numeric(df[actual_col], errors="coerce")
        f = pd.to_numeric(df[forecast_col], errors="coerce")
        ok = a.notna() & f.notna()
        a, f = a[ok].values, f[ok].values
        m = _metrics(a, f)
        m["MAE"] = round(float(np.abs(a - f).mean()), 2)
        m["RMSE"] = round(float(np.sqrt(((a - f) ** 2).mean())), 2)
        m["rows"] = int(ok.sum())
        return json.dumps(m, default=str)
    except Exception as e:
        return f"Error: {e}"


@tool
def interval_profile(
    file_name: str,
    datetime_col: str,
    volume_col: str,
    sheet_name: str = "",
    output_name: str = "interval_profile.xlsx",
) -> str:
    """Build the intraday distribution (share of daily volume per time-of-day interval,
    by day of week) and a day-of-week index from interval-level history.
    Saves two sheets: interval_share and day_of_week."""
    try:
        df = _load(file_name, sheet_name)
        df[datetime_col] = pd.to_datetime(df[datetime_col])
        df[volume_col] = pd.to_numeric(df[volume_col], errors="coerce").fillna(0)
        df["_date"] = df[datetime_col].dt.date
        df["_dow"] = df[datetime_col].dt.day_name()
        df["_time"] = df[datetime_col].dt.strftime("%H:%M")
        agg = df.groupby(["_date", "_dow", "_time"], as_index=False)[volume_col].sum()
        agg["_daily"] = agg.groupby("_date")[volume_col].transform("sum")
        agg = agg[agg["_daily"] > 0].copy()
        agg["_share"] = agg[volume_col] / agg["_daily"]

        share = agg.groupby(["_dow", "_time"])["_share"].mean().unstack("_dow").fillna(0)
        share = share.div(share.sum().replace(0, 1))  # each day column sums to 100%
        order = [d for d in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"] if d in share.columns]
        share = share[order].reset_index().rename(columns={"_time": "time"})

        daily = agg.drop_duplicates("_date")[["_date", "_dow", "_daily"]]
        dow = daily.groupby("_dow")["_daily"].mean()
        dow_tbl = pd.DataFrame({
            "day_of_week": dow.index,
            "avg_daily_volume": dow.values.round(1),
            "index_vs_avg": (dow / dow.mean()).values.round(3),
        })

        path = os.path.join(OUTPUT_DIR, output_name)
        with pd.ExcelWriter(path, engine="openpyxl") as w:
            share.to_excel(w, sheet_name="interval_share", index=False)
            dow_tbl.to_excel(w, sheet_name="day_of_week", index=False)

        peak = share.set_index("time").max().round(3).to_dict()
        return (
            f"Days analysed: {daily.shape[0]}. Intervals per day: {share.shape[0]}.\n"
            f"Day-of-week index:\n{dow_tbl.to_string(index=False)}\n"
            f"Peak interval share by day: {peak}\nSaved to {path}"
        )
    except Exception as e:
        return f"Error: {e}"


@tool
def erlang_staffing(
    volume: float,
    aht_seconds: float = 300,
    interval_minutes: int = 30,
    sl_target: float = 0.80,
    sl_seconds: float = 20,
    max_occupancy: float = 0.85,
    shrinkage: float = 0.30,
) -> str:
    """Erlang C staffing for ONE interval. Given contacts in the interval and AHT, find the
    minimum agents meeting the service level target (sl_target of contacts answered within
    sl_seconds) and the occupancy cap. Also returns scheduled FTE after shrinkage and a
    sensitivity table of nearby staffing levels. Percent inputs can be 0.8 or 80."""
    try:
        sl_target, max_occupancy, shrinkage = _pct(sl_target), _pct(max_occupancy), _pct(shrinkage)
        r = _staff(volume, interval_minutes, aht_seconds, sl_target, sl_seconds, max_occupancy)
        n = r["agents"]
        rows = []
        for k in range(max(1, n - 2), n + 4):
            rows.append({
                "agents": k,
                "service_level_%": round(100 * _service_level(k, r["traffic"], aht_seconds, sl_seconds), 1),
                "occupancy_%": round(100 * r["traffic"] / k, 1),
                "asa_sec": round(_asa(k, r["traffic"], aht_seconds), 1),
            })
        return (
            f"Traffic: {r['traffic']:.2f} Erlangs\n"
            f"Agents required (on the phones): {n}\n"
            f"Service level: {r['sl']*100:.1f}% in {sl_seconds:.0f}s | Occupancy: {r['occ']*100:.1f}% | ASA: {r['asa']:.1f}s\n"
            f"Scheduled agents with {shrinkage*100:.0f}% shrinkage: {n / (1 - shrinkage):.1f}\n"
            f"Sensitivity:\n{pd.DataFrame(rows).to_string(index=False)}"
        )
    except Exception as e:
        return f"Error: {e}"


@tool
def staffing_from_file(
    file_name: str,
    volume_col: str,
    aht_col: str = "",
    aht_seconds: float = 300,
    interval_minutes: int = 30,
    sl_target: float = 0.80,
    sl_seconds: float = 20,
    max_occupancy: float = 0.85,
    shrinkage: float = 0.30,
    sheet_name: str = "",
    output_name: str = "staffing_requirements.xlsx",
) -> str:
    """Apply Erlang C staffing to every row of a file (each row = one interval).
    Use aht_col if AHT is in the file, otherwise aht_seconds. Saves the file with
    traffic, agents required, service level, occupancy, ASA and scheduled agents."""
    try:
        sl_target, max_occupancy, shrinkage = _pct(sl_target), _pct(max_occupancy), _pct(shrinkage)
        df = _load(file_name, sheet_name)
        recs = []
        for _, r in df.iterrows():
            vol = float(pd.to_numeric(r[volume_col], errors="coerce") or 0)
            aht = float(r[aht_col]) if aht_col else float(aht_seconds)
            recs.append(_staff(vol, interval_minutes, aht, sl_target, sl_seconds, max_occupancy))
        out = df.copy()
        out["traffic_erlangs"] = [round(x["traffic"], 2) for x in recs]
        out["agents_required"] = [x["agents"] for x in recs]
        out["service_level_%"] = [round(x["sl"] * 100, 1) for x in recs]
        out["occupancy_%"] = [round(x["occ"] * 100, 1) for x in recs]
        out["asa_sec"] = [round(x["asa"], 1) for x in recs]
        out["scheduled_agents"] = [math.ceil(x["agents"] / (1 - shrinkage)) for x in recs]
        path = _save_df(out, output_name)
        hours = out["agents_required"].sum() * interval_minutes / 60
        return (
            f"Rows: {len(out)} | Peak agents: {out['agents_required'].max()} | "
            f"Average agents: {out['agents_required'].mean():.1f}\n"
            f"Total agent hours: {hours:.1f} | Scheduled hours with shrinkage: {hours / (1 - shrinkage):.1f}\n"
            f"Saved to {path}"
        )
    except Exception as e:
        return f"Error: {e}"


@tool
def daily_to_interval_staffing(
    forecast_file: str,
    date_col: str,
    volume_col: str,
    profile_file: str = "interval_profile.xlsx",
    aht_seconds: float = 300,
    interval_minutes: int = 30,
    sl_target: float = 0.80,
    sl_seconds: float = 20,
    max_occupancy: float = 0.85,
    shrinkage: float = 0.30,
    output_name: str = "interval_staffing.xlsx",
) -> str:
    """Turn a DAILY volume forecast into interval volumes using the profile from
    interval_profile, then calculate agents required per interval with Erlang C.
    interval_minutes must match the profile's interval length. Saves intervals and
    a daily summary (peak agents, agent hours, scheduled hours)."""
    try:
        sl_target, max_occupancy, shrinkage = _pct(sl_target), _pct(max_occupancy), _pct(shrinkage)
        fc = _load(forecast_file)
        fc[date_col] = pd.to_datetime(fc[date_col])
        share = pd.read_excel(_find(profile_file), sheet_name="interval_share")
        rows = []
        for _, r in fc.iterrows():
            dname = r[date_col].day_name()
            if dname not in share.columns:
                continue
            for _, p in share.iterrows():
                vol = float(r[volume_col]) * float(p[dname])
                res = _staff(vol, interval_minutes, aht_seconds, sl_target, sl_seconds, max_occupancy)
                rows.append({
                    "date": r[date_col].date(), "day": dname, "time": p["time"],
                    "volume": round(vol, 1), "agents_required": res["agents"],
                    "service_level_%": round(res["sl"] * 100, 1),
                    "occupancy_%": round(res["occ"] * 100, 1),
                    "scheduled_agents": math.ceil(res["agents"] / (1 - shrinkage)),
                })
        iv = pd.DataFrame(rows)
        if iv.empty:
            return "Error: no matching days between the forecast and the profile."
        daily = iv.groupby(["date", "day"], as_index=False).agg(
            volume=("volume", "sum"),
            peak_agents=("agents_required", "max"),
            agent_intervals=("agents_required", "sum"),
        )
        daily["agent_hours"] = (daily["agent_intervals"] * interval_minutes / 60).round(1)
        daily["scheduled_hours"] = (daily["agent_hours"] / (1 - shrinkage)).round(1)
        daily["volume"] = daily["volume"].round(0)
        daily = daily.drop(columns="agent_intervals")
        path = os.path.join(OUTPUT_DIR, output_name)
        with pd.ExcelWriter(path, engine="openpyxl") as w:
            iv.to_excel(w, sheet_name="intervals", index=False)
            daily.to_excel(w, sheet_name="daily_summary", index=False)
        return f"Daily summary:\n{daily.head(31).to_string(index=False)}\nSaved to {path}"
    except Exception as e:
        return f"Error: {e}"


@tool
def capacity_plan(
    file_name: str,
    period_col: str,
    volume_col: str,
    aht_col: str = "",
    aht_seconds: float = 300,
    period_type: str = "week",
    starting_headcount: float = 0,
    weekly_hours_per_fte: float = 40,
    shrinkage: float = 0.30,
    occupancy: float = 0.85,
    buffer_pct: float = 0.0,
    attrition_pct_annual: float = 0.30,
    hires_col: str = "",
    planned_hires_per_period: float = 0,
    headcount_col: str = "",
    sheet_name: str = "",
    output_name: str = "capacity_plan.xlsx",
) -> str:
    """Long-term capacity plan. For each period (day, week or month): workload hours =
    volume x AHT, required FTE = workload / (hours per FTE x (1-shrinkage) x occupancy),
    plus an optional buffer. Compares with projected headcount (starting_headcount reduced
    by attrition and increased by hires, or headcount_col if the file has it) and reports
    the gap and hires needed. Percent inputs can be 0.3 or 30."""
    try:
        shrinkage, occupancy = _pct(shrinkage), _pct(occupancy)
        attr_annual, buffer_pct = _pct(attrition_pct_annual), _pct(buffer_pct) if buffer_pct else 0.0
        pt = period_type.strip().lower()
        per_year = {"day": 365, "week": 52, "month": 12}
        hrs_factor = {"day": 1 / 5, "week": 1, "month": 52 / 12}
        if pt not in per_year:
            return "Error: period_type must be day, week or month."
        hours_per_fte = weekly_hours_per_fte * hrs_factor[pt]

        df = _load(file_name, sheet_name)
        vol = pd.to_numeric(df[volume_col], errors="coerce").fillna(0)
        aht = pd.to_numeric(df[aht_col], errors="coerce").fillna(aht_seconds) if aht_col else aht_seconds
        workload = vol * aht / 3600.0
        req = workload / (hours_per_fte * (1 - shrinkage) * occupancy) * (1 + buffer_pct)

        attr_p = 1 - (1 - attr_annual) ** (1 / per_year[pt])
        if headcount_col:
            hc = pd.to_numeric(df[headcount_col], errors="coerce").fillna(0).tolist()
        else:
            hires = (pd.to_numeric(df[hires_col], errors="coerce").fillna(0).tolist()
                     if hires_col else [planned_hires_per_period] * len(df))
            hc, prev = [], float(starting_headcount)
            for h in hires:
                prev = prev * (1 - attr_p) + h
                hc.append(prev)

        out = pd.DataFrame({
            "period": df[period_col],
            "volume": vol.round(0),
            "workload_hours": workload.round(1),
            "required_fte": req.round(1),
            "projected_headcount": np.round(hc, 1),
        })
        out["gap_fte"] = (out["projected_headcount"] - out["required_fte"]).round(1)
        out["status"] = np.where(out["gap_fte"] < 0, "Shortfall", "Surplus")
        out["hires_needed"] = np.ceil(np.maximum(-out["gap_fte"], 0)).astype(int)
        path = _save_df(out, output_name)

        short = out[out["status"] == "Shortfall"]
        return (
            f"Hours per FTE per {pt}: {hours_per_fte:.1f} | shrinkage {shrinkage*100:.0f}% | occupancy {occupancy*100:.0f}% | "
            f"attrition per {pt}: {attr_p*100:.2f}%\n"
            f"Periods with shortfall: {len(short)} of {len(out)} | "
            f"Largest shortfall: {out['gap_fte'].min():.1f} FTE | Peak required FTE: {out['required_fte'].max():.1f}\n"
            f"{out.head(40).to_string(index=False)}\nSaved to {path}\n"
            f"Note: this is volume-based capacity; interval-level Erlang staffing is more precise for service level."
        )
    except Exception as e:
        return f"Error: {e}"


@tool
def calc_shrinkage(components_json: str) -> str:
    """Combine shrinkage components into a total. components_json example:
    '{"absence": 0.08, "breaks": 0.10, "training": 0.04, "meetings": 0.03, "annual_leave": 0.10}'
    (fractions or percents). Returns the additive total (standard WFM practice) and the
    compounded total."""
    try:
        vals = {k: _pct(v) for k, v in json.loads(components_json).items()}
        additive = sum(vals.values())
        compounded = 1 - float(np.prod([1 - v for v in vals.values()]))
        return (
            f"Components: { {k: f'{v*100:.1f}%' for k, v in vals.items()} }\n"
            f"Additive shrinkage: {additive*100:.1f}%\nCompounded shrinkage: {compounded*100:.1f}%"
        )
    except Exception as e:
        return f"Error: {e}"


analysis_tools = [
    list_data_files,
    inspect_data,
    run_pandas_code,
    create_chart,
    save_power_bi_code,
    save_report,
    forecast_volume,
    forecast_accuracy,
    interval_profile,
    erlang_staffing,
    staffing_from_file,
    daily_to_interval_staffing,
    capacity_plan,
    calc_shrinkage,
]