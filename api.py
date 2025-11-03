from __future__ import annotations
import os, json
from pathlib import Path
from typing import List, Union, Dict, Any, Iterable, Optional
import pandas as pd

def ingest_astarta(
    ttn_path: Union[str, Path],
    station_paths: Union[str, Path, List[Union[str, Path]]],
    mapping_json: Optional[Union[str, Path]] = None,
    chunksize: int = 200_000,
    tz: str = "Europe/Kyiv",
) -> Dict[str, pd.DataFrame]:
    """
    Recieves:
      - ttn_path: path to file with TTN in format of rows like:
        [idx] [ТТН ввоз ... SEMEN000519 от 08.07.2025 16:07:51] [step_order] [step_name] [ts] [operator]
      - station_paths: list of files or folder with files in format:
        Post, DT  (e.g., 'КПП 1', '10/23/25 14:56')
      - mapping_json (optional): {"map": {raw_name -> norm_name}, "order": [...]}

    Returns:
      {
        "ttn": DataFrame(ttn_id, step_order, step_name_raw, step_norm, ts, operator),
        "stations": DataFrame(station_raw, station_norm, dt)
      }
    """
    mapping = _load_mapping(mapping_json) if mapping_json else {"map": {}, "order": []}

    df_ttn = _read_ttn_file(ttn_path, chunksize=chunksize, tz=tz)

    if mapping["map"]:
        miss = df_ttn["step_norm"].isna() | (df_ttn["step_norm"] == "")
        df_ttn.loc[miss, "step_norm"] = (
            df_ttn.loc[miss, "step_name_raw"].map(mapping["map"]).astype("string")
        )

    df_ttn = df_ttn[["ttn_id","step_order","step_name_raw","step_norm","ts","operator"]]

    station_files = _expand_paths(station_paths)
    st_frames = []
    for p in station_files:
        st = _read_station_file(p, tz=tz)
        if mapping["map"]:
            miss = st["station_norm"].isna() | (st["station_norm"] == "")
            st.loc[miss, "station_norm"] = st.loc[miss, "station_raw"].map(mapping["map"]).astype("string")
        st_frames.append(st)

    df_st = pd.concat(st_frames, ignore_index=True) if st_frames else pd.DataFrame(
        columns=["station_raw","station_norm","dt"]
    )

    return {"ttn": df_ttn, "stations": df_st}

def _read_ttn_file(path: Union[str, Path], chunksize: int, tz: str) -> pd.DataFrame:
    """
    Expects columns without strict names; autodetects separator.
    Minimum required columns: in the order as in the example below:
      idx | ttn_full | step_order | step_name_raw | ts | operator
    """
    cols = ["row_no","ttn_full","step_order","step_name_raw","ts","operator"]
    frames = []

    path_obj = Path(path)
    ext = path_obj.suffix.lower()

    if ext in {".xls", ".xlsx"}:
        df = pd.read_excel(path_obj, header=None, names=cols, dtype=str)
        chunks = [df]
    else:
        chunks = pd.read_csv(
            path_obj, header=None, names=cols, sep=None, engine="python",
            chunksize=chunksize, dtype=str, keep_default_na=False
        )

    for chunk in chunks:
        c = chunk.copy()

        c["step_order"] = pd.to_numeric(c["step_order"], errors="coerce").astype("Int64")
        c["step_name_raw"] = c["step_name_raw"].astype("string")
        c["operator"] = c["operator"].astype("string")

        # Extract ttn_id from ttn_full: take the part before ' от ', then the last word
        # example: "ТТН ввоз (элеватор) SEMEN000519 от 08.07.2025 16:07:51"
        ttn_prefix = c["ttn_full"].astype(str).str.split(" от ", n=1, expand=True)[0]
        c["ttn_id"] = ttn_prefix.str.split().str[-1].astype("string")

        c["ts"] = _to_tz(c["ts"], tz=tz, dayfirst=True)

        c["step_norm"] = pd.Series([pd.NA] * len(c), dtype="string")

        c = c.dropna(subset=["ttn_id","ts"])
        frames.append(c[["ttn_id","step_order","step_name_raw","step_norm","ts","operator"]])

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["ttn_id","step_order","step_name_raw","step_norm","ts","operator"]
    )



def _read_station_file(path: Union[str, Path], tz: str) -> pd.DataFrame:
    """
    Expects file in format:
      Post, DT
      'КПП 1', '10/23/25 14:56'
    """
    path_obj = Path(path)
    ext = path_obj.suffix.lower()
    if ext in {".xls", ".xlsx"}:
        df = pd.read_excel(path_obj, dtype=str)
    else:
        df = pd.read_csv(path_obj, sep=None, engine="python", dtype=str, keep_default_na=False)

    # Normalize headers to handle possible localized or case variants
    rename_map = {}
    station_aliases = {"post", "station"}
    dt_aliases = {"dt", "datetime", "date", "timestamp"}
    for col in list(df.columns):
        low = str(col).strip().lower()
        if low in station_aliases:
            rename_map[col] = "station_raw"
        elif low in dt_aliases:
            rename_map[col] = "dt"
    if not rename_map:
        # Fallback to default English headers if present
        df = df.rename(columns={"Post":"station_raw", "DT":"dt"})
    else:
        df = df.rename(columns=rename_map)
    if "station_raw" not in df or "dt" not in df:
        cols_l = {c.lower(): c for c in df.columns}
        df = df.rename(columns={cols_l.get("post","post"):"station_raw",
                                cols_l.get("dt","dt"):"dt"})
    df["station_raw"] = df["station_raw"].astype("string").str.strip()

    df["dt"] = _to_tz(df["dt"], tz=tz, dayfirst=False)

    df["station_norm"] = pd.Series([pd.NA] * len(df), dtype="string")

    df = df.dropna(subset=["station_raw","dt"])
    return df[["station_raw","station_norm","dt"]]


def _expand_paths(paths: Union[str, Path, List[Union[str, Path]]]) -> List[Path]:
    supported_exts = {".csv", ".txt", ".xlsx", ".xls"}
    def _ok(f: Path) -> bool:
        return f.is_file() and f.suffix.lower() in supported_exts

    if isinstance(paths, (str, Path)):
        p = Path(paths)
        if p.is_dir():
            return sorted([x for x in p.iterdir() if _ok(x)])
        return [p] if _ok(p) else []
    return [Path(x) for x in paths if Path(x).suffix.lower() in supported_exts]

def _to_tz(series: pd.Series, tz: str, dayfirst: bool) -> pd.Series:
    dt = pd.to_datetime(series, errors="coerce", dayfirst=dayfirst, utc=False)
    dt = dt.dt.tz_localize(tz, nonexistent="NaT", ambiguous="NaT")
    return dt

def _load_mapping(path: Union[str, Path]) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        m = json.load(f)
    if "map" not in m: m["map"] = {}
    if "order" not in m: m["order"] = []
    return m


def compute_stage_durations(
    df_ttn: pd.DataFrame,
    stages: Dict[str, tuple[str, str]],
    use_norm: bool = True,
) -> pd.DataFrame:
    """
    Compute per-stage durations as the time between the start of a step and the
    start of the next step for each TTN.

    Params:
      - df_ttn: output of _read_ttn_file/ingest_astarta["ttn"].
      - stages: mapping of stage_name -> (start_step_name, next_start_step_name).
        Example:
          {
            'Вїзд': ('Диспетчер вїзду', 'Початок відбору проб'),
            'Відбір_проб': ('Початок відбору проб', 'Кінець відбору проб'),
            ...
          }
      - use_norm: if True, match on 'step_norm' when available, otherwise use 'step_name_raw'.

    Returns DataFrame with columns:
      [ttn_id, stage, start_ts, end_ts, duration_s, duration_min]
    """
    if df_ttn.empty:
        return pd.DataFrame(columns=["ttn_id","stage","start_ts","end_ts","duration_s","duration_min"])

    step_col = "step_norm" if use_norm and ("step_norm" in df_ttn.columns) else "step_name_raw"

    # For each ttn_id + step name, take the earliest timestamp
    base = (
        df_ttn[["ttn_id", step_col, "ts"]]
        .dropna(subset=["ttn_id", step_col, "ts"])
        .groupby(["ttn_id", step_col], as_index=False)["ts"].min()
    )

    # Pivot to wide for quick column lookups
    try:
        wide = base.pivot(index="ttn_id", columns=step_col, values="ts")
    except Exception:
        # Fallback to empty if pivot fails (e.g., too many duplicate keys after aggregation)
        return pd.DataFrame(columns=["ttn_id","stage","start_ts","end_ts","duration_s","duration_min"])

    rows = []
    for stage_name, (start_name, next_start_name) in stages.items():
        if start_name not in wide.columns or next_start_name not in wide.columns:
            continue
        start_series = wide[start_name]
        end_series = wide[next_start_name]
        # Compute durations; keep only positive deltas
        duration = (end_series - start_series)
        mask = duration.notna() & start_series.notna() & end_series.notna() & (duration.dt.total_seconds() > 0)
        if not mask.any():
            continue
        part = pd.DataFrame({
            "ttn_id": start_series.index[mask],
            "stage": stage_name,
            "start_ts": start_series[mask].values,
            "end_ts": end_series[mask].values,
        })
        part["duration_s"] = (part["end_ts"] - part["start_ts"]).dt.total_seconds().astype("int64")
        part["duration_min"] = (part["duration_s"] / 60.0)
        rows.append(part)

    if not rows:
        return pd.DataFrame(columns=["ttn_id","stage","start_ts","end_ts","duration_s","duration_min"])

    out = pd.concat(rows, ignore_index=True)
    return out[["ttn_id","stage","start_ts","end_ts","duration_s","duration_min"]]


def summarize_stage_durations(df_durations: pd.DataFrame) -> pd.DataFrame:
    """
    Produce per-stage statistics similar to notebook summaries.
    Returns columns:
      [stage, n, mean_min, median_min, p90_min, total_hours]
    """
    if df_durations.empty:
        return pd.DataFrame(columns=["stage","n","mean_min","median_min","p90_min","total_hours"])

    g = df_durations.groupby("stage")
    stats = g["duration_min"].agg(
        n="count",
        mean_min="mean",
        median_min="median",
    ).reset_index()
    # p90 separately to avoid including it in the same agg signature (pandas older versions)
    p90 = g["duration_min"].quantile(0.90).rename("p90_min").reset_index()
    stats = stats.merge(p90, on="stage", how="left")
    # total hours for rough load measure
    total_sec = g["duration_s"].sum().rename("total_seconds").reset_index()
    stats = stats.merge(total_sec, on="stage", how="left")
    stats["total_hours"] = stats["total_seconds"] / 3600.0
    stats = stats.drop(columns=["total_seconds"]).sort_values("mean_min", ascending=False)
    return stats
