
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import pandas as pd
import numpy as np
from io import BytesIO
import matplotlib.pyplot as plt
import base64
import math
from datetime import datetime
import pytz

# --------- Helpers ---------

TZ = "Europe/Kyiv"

def _read_table(file: UploadFile) -> pd.DataFrame:
    name = (file.filename or "").lower()
    content = file.file.read()
    if name.endswith(".csv"):
        # Try common encodings
        for enc in [None, "utf-8", "utf-16", "cp1251"]:
            try:
                return pd.read_csv(BytesIO(content), encoding=enc, sep=None, engine="python")
            except Exception:
                continue
        # Fallback strict
        return pd.read_csv(BytesIO(content))
    elif name.endswith(".xlsx") or name.endswith(".xls"):
        try:
            # .xlsx via openpyxl (default), .xls via xlrd
            return pd.read_excel(BytesIO(content))
        except Exception:
            try:
                return pd.read_excel(BytesIO(content), engine="openpyxl")
            except Exception:
                try:
                    return pd.read_excel(BytesIO(content), engine="xlrd")
                except Exception as e:
                    raise HTTPException(status_code=400, detail=f"Failed to read Excel: {e}")
    else:
        # Try CSV as a fallback
        try:
            return pd.read_csv(BytesIO(content), sep=None, engine="python")
        except Exception:
            raise HTTPException(status_code=400, detail="Unsupported file type. Please upload CSV or Excel.")

def _to_dt(s: pd.Series) -> pd.Series:
    # Parse Ukrainian DD.MM.YYYY HH:MM[:SS] with dayfirst
    dt = pd.to_datetime(s, dayfirst=True, errors="coerce", utc=False)
    # Localize to Europe/Kyiv if naive
    try:
        tz = pytz.timezone(TZ)
        # If any tz-aware, keep as-is; else localize
        if dt.dt.tz is None:
            dt = dt.dt.tz_localize(tz, nonexistent='shift_forward', ambiguous='NaT')
        else:
            dt = dt.dt.tz_convert(tz)
    except Exception:
        pass
    return dt

def _encode_png(fig) -> str:
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")

def _duration_stats(s: pd.Series) -> Dict[str, Any]:
    s = s.dropna()
    try:
        s = s.dt.total_seconds()
    except Exception:
        s = pd.to_numeric(s, errors="coerce")
    s = s.dropna().astype(float)
    if s.empty:
        return {
            "count": 0, "mean_s": None, "median_s": None, "p90_s": None,
            "p95_s": None, "std_s": None, "iqr_s": None, "mad_s": None
        }
    q = s.quantile([0.25, 0.5, 0.9, 0.95, 0.75])
    mad = (s - s.median()).abs().median()
    return {
        "count": int(s.count()),
        "mean_s": float(s.mean()),
        "median_s": float(q.loc[0.5]),
        "p90_s": float(q.loc[0.9]),
        "p95_s": float(q.loc[0.95]),
        "std_s": float(s.std(ddof=1)) if s.count() > 1 else 0.0,
        "iqr_s": float(q.loc[0.75] - q.loc[0.25]),
        "mad_s": float(mad),
    }


def _format_seconds(x: Optional[float]) -> Optional[str]:
    if isinstance(x, pd.Timedelta):
        x = x.total_seconds()
    try:
        import numpy as _np
        if isinstance(x, _np.timedelta64):
            x = _np.timedelta64(x, 's') / _np.timedelta64(1, 's')
    except Exception:
        pass

    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    x = int(x)
    h, r = divmod(x, 3600)
    m, s = divmod(r, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def _pick_bottleneck(stage_stats: pd.DataFrame) -> Dict[str, Any]:
    # Use p95 duration as primary, median as secondary
    if stage_stats.empty:
        return {}
    s_p95 = stage_stats.sort_values(["p95_s","median_s","mean_s"], ascending=False).iloc[0].to_dict()
    return {
        "stage": s_p95.get("stage"),
        "p95": _format_seconds(s_p95.get("p95_s")),
        "median": _format_seconds(s_p95.get("median_s")),
        "mean": _format_seconds(s_p95.get("mean_s")),
        "count": int(s_p95.get("count", 0)),
    }


def _boxplot_chart(stage_df: pd.DataFrame) -> str:
    # Boxplot durations by stage
    data = []
    labels = []
    for stage, g in stage_df.groupby("stage"):
        d = g["duration"].dropna().dt.total_seconds() / 60.0  # minutes
        if not d.empty:
            data.append(d.values)
            labels.append(stage)
    if not data:
        fig = plt.figure(figsize=(8,4))
        plt.title("No duration data")
        return _encode_png(fig)
    fig = plt.figure(figsize=(10, 6))
    plt.boxplot(data, showfliers=False)
    plt.xticks(range(1, len(labels)+1), labels, rotation=30, ha="right")
    plt.ylabel("Тривалість, хв")
    plt.title("Розподіл тривалості по етапах (без викидів)")
    return _encode_png(fig)

def _bottleneck_barplot(stage_df: pd.DataFrame) -> str:
    # Візуалізація тривалостей по етапах у хвилинах
    df_stats = (
        stage_df.groupby("stage")["duration"]
        .apply(lambda s: s.dt.total_seconds().mean() / 60.0)
        .reset_index(name="mean_min")
        .sort_values("mean_min", ascending=False)
    )
    fig = plt.figure(figsize=(10, 6))
    plt.barh(df_stats["stage"], df_stats["mean_min"], color="tomato")
    plt.xlabel("Середня тривалість, хв")
    plt.ylabel("Етап")
    plt.title("Середня тривалість по етапах (чим довше, тим вузьке місце)")
    plt.gca().invert_yaxis()
    return _encode_png(fig)


def _cumulative_stage_plot(stage_df: pd.DataFrame) -> str:
    # Кумулятивна сума середніх тривалостей — покаже накопичення часу по процесу
    df_stats = (
        stage_df.groupby("stage")["duration"]
        .apply(lambda s: s.dt.total_seconds().mean() / 60.0)
        .reset_index(name="mean_min")
    )
    df_stats["cum_sum"] = df_stats["mean_min"].cumsum()
    fig = plt.figure(figsize=(10, 5))
    plt.plot(df_stats["stage"], df_stats["cum_sum"], marker="o", color="royalblue")
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Накопичений час, хв")
    plt.title("Кумулятивний час проходження етапів")
    plt.grid(True, alpha=0.3)
    return _encode_png(fig)

# --------- FastAPI ---------

app = FastAPI(title="Bottleneck Analysis API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AnalyzeResponse(BaseModel):
    summary: Dict[str, Any]
    stage_table: List[Dict[str, Any]]
    bottleneck: Dict[str, Any]
    charts: Dict[str, str]  # base64 PNGs

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(file: UploadFile = File(...)) -> AnalyzeResponse:
    try:
        df = _read_table(file)
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read file: {e}")

    required_cols = ["ТТН", "НомерРядка", "ТочкаБП", "Час реєстрації"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing required columns: {missing}")

    # Types & sorting
    df = df.copy()
    df["НомерРядка"] = pd.to_numeric(df["НомерРядка"], errors="coerce")
    df["Час реєстрації"] = _to_dt(df["Час реєстрації"])
    df = df.dropna(subset=["ТТН", "НомерРядка", "Час реєстрації"]).sort_values(["ТТН", "НомерРядка"])

    # Duration = next start - current start within each TTN
    df["next_start"] = df.groupby("ТТН")["Час реєстрації"].shift(-1)
    df["duration"] = df["next_start"] - df["Час реєстрації"]

    # Remove final stages with no next_start (e.g. "Диспетчер виїзду")
    df = df[~df["next_start"].isna()].copy()

    # Remove negative or zero durations (data issues)
    df.loc[(df["duration"] <= pd.Timedelta(0)) | (df["duration"].isna()), "duration"] = pd.NaT

    # Stage-level stats
    stage_df = df.rename(columns={"ТочкаБП": "stage"})
    stats_rows = []
    for stage, g in stage_df.groupby("stage"):
        st = _duration_stats(g["duration"])
        row = {"stage": stage, **st}
        row["mean"] = _format_seconds(row["mean_s"])
        row["median"] = _format_seconds(row["median_s"])
        row["p90"] = _format_seconds(row["p90_s"])
        row["p95"] = _format_seconds(row["p95_s"])
        row["std"] = _format_seconds(row["std_s"])
        row["iqr"] = _format_seconds(row["iqr_s"])
        row["mad"] = _format_seconds(row["mad_s"])
        stats_rows.append(row)

    stage_stats = pd.DataFrame(stats_rows).sort_values(
        ["p95_s", "median_s", "mean_s"], ascending=False, na_position="last"
    )

    # Cycle time per TTN (first start to last start)
    cyc = df.groupby("ТТН", as_index=False).agg(
        first_start=("Час реєстрації", "min"),
        last_start=("Час реєстрації", "max"),
        n_events=("ТочкаБП", "count"),
    )
    cyc["cycle_time"] = cyc["last_start"] - cyc["first_start"]
    cyc_valid = cyc.dropna(subset=["cycle_time"])

    # Summary
    summ = {
        "ttn_count": int(cyc.shape[0]),
        "events_count": int(df.shape[0]),
        "period_start": str(df["Час реєстрації"].min()),
        "period_end": str(df["Час реєстрації"].max()),
        "cycle_time_mean": _format_seconds(
            cyc_valid["cycle_time"].dt.total_seconds().mean() if not cyc_valid.empty else None
        ),
        "cycle_time_median": _format_seconds(
            cyc_valid["cycle_time"].dt.total_seconds().median() if not cyc_valid.empty else None
        ),
    }

    # Bottleneck
    bottleneck = _pick_bottleneck(stage_stats)

    # Charts
    try:
        box_b64 = _boxplot_chart(stage_df)
    except Exception:
        fig = plt.figure(figsize=(6, 3))
        plt.title("Boxplot unavailable")
        box_b64 = _encode_png(fig)

    try:
        bar_b64 = _bottleneck_barplot(stage_df)
    except Exception:
        fig = plt.figure(figsize=(6, 3))
        plt.title("Barplot unavailable")
        bar_b64 = _encode_png(fig)

    try:
        cum_b64 = _cumulative_stage_plot(stage_df)
    except Exception:
        fig = plt.figure(figsize=(6, 3))
        plt.title("Cumulative plot unavailable")
        cum_b64 = _encode_png(fig)

    # Prepare stage_table
    display_cols = ["stage", "count", "mean", "median", "p90", "p95", "std", "iqr", "mad"]
    stage_table = stage_stats.fillna("").reindex(columns=display_cols).to_dict(orient="records")

    return AnalyzeResponse(
        summary=summ,
        stage_table=stage_table,
        bottleneck=bottleneck,
        charts={
            "bottleneck_barplot": bar_b64,
            "stage_boxplot": box_b64,
            "cumulative_time": cum_b64,
        },
    )
