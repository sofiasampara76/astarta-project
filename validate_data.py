from __future__ import annotations
from pathlib import Path
import argparse
import sys
import pandas as pd

from api import ingest_astarta, compute_stage_durations, summarize_stage_durations


def summarize_ttn(df: pd.DataFrame) -> str:
    if df.empty:
        return "TTN: 0 rows"
    parts = []
    parts.append(f"TTN: {len(df):,} rows")
    parts.append(f"  unique ttn_id: {df['ttn_id'].nunique():,}")
    parts.append(f"  steps with missing step_norm: {df['step_norm'].isna().sum():,}")
    if 'ts' in df and not df['ts'].isna().all():
        parts.append(f"  ts range: {df['ts'].min()} -> {df['ts'].max()}")
        parts.append(f"  tz: {df['ts'].dt.tz}")
    return "\n".join(parts)


def summarize_stations(df: pd.DataFrame) -> str:
    if df.empty:
        return "Stations: 0 rows"
    parts = []
    parts.append(f"Stations: {len(df):,} rows")
    parts.append(f"  unique station_raw: {df['station_raw'].nunique():,}")
    parts.append(f"  rows with missing station_norm: {df['station_norm'].isna().sum():,}")
    if 'dt' in df and not df['dt'].isna().all():
        parts.append(f"  dt range: {df['dt'].min()} -> {df['dt'].max()}")
        parts.append(f"  tz: {df['dt'].dt.tz}")
    return "\n".join(parts)


def _df_head_jsonable(df: pd.DataFrame, n: int) -> list[dict]:
    head = df.head(n).copy()
    # Convert datetimes to ISO strings for API/JSON friendliness
    for col in head.columns:
        if pd.api.types.is_datetime64_any_dtype(head[col]):
            head[col] = head[col].dt.strftime("%Y-%m-%dT%H:%M:%S%z")
    return head.to_dict(orient="records")


DEFAULT_STAGES = {
    'Вїзд': ('Диспетчер вїзду', 'Початок відбору проб'),
    'Відбір_проб': ('Початок відбору проб', 'Кінець відбору проб'),
    'Зважування': ('Зважування брутто', 'Лабораторний аналіз'),
    'Лабораторія': ('Лабораторний аналіз', 'Пункт розгрузки'),
    'Розвантаження': ('Пункт розгрузки', 'Зважування тари'),
    'Виїзд': ('Зважування тари', 'Диспетчер виїзду'),
}


def validate_ingestion(ttn: Path, stations: Path, mapping: Path | None, chunksize: int = 200_000, tz: str = "Europe/Kyiv", head: int = 5, stages: dict | None = None, use_norm: bool = True) -> dict:
    """
    Runs ingestion and returns a JSON-serializable summary suitable for API responses.
    """
    result = ingest_astarta(
        ttn_path=ttn,
        station_paths=stations,
        mapping_json=mapping,
        chunksize=chunksize,
        tz=tz,
    )
    df_ttn = result.get("ttn", pd.DataFrame())
    df_st = result.get("stations", pd.DataFrame())

    # Compute durations using provided or default stages mapping
    stages_map = stages or DEFAULT_STAGES
    try:
        df_dur = compute_stage_durations(df_ttn, stages_map, use_norm=use_norm)
    except Exception:
        df_dur = pd.DataFrame()

    ttn_summary = {
        "rows": int(len(df_ttn)),
        "unique_ttn_id": int(df_ttn["ttn_id"].nunique()) if not df_ttn.empty else 0,
        "missing_step_norm": int(df_ttn["step_norm"].isna().sum()) if "step_norm" in df_ttn else 0,
        "ts_min": (df_ttn["ts"].min().isoformat() if not df_ttn.empty and "ts" in df_ttn and not df_ttn["ts"].isna().all() else None),
        "ts_max": (df_ttn["ts"].max().isoformat() if not df_ttn.empty and "ts" in df_ttn and not df_ttn["ts"].isna().all() else None),
        "tz": (str(df_ttn["ts"].dt.tz) if not df_ttn.empty and "ts" in df_ttn else None),
        "head": _df_head_jsonable(df_ttn, head) if head > 0 else [],
        "columns_ok": set(["ttn_id","step_order","step_name_raw","step_norm","ts","operator"]).issubset(df_ttn.columns),
    }

    st_summary = {
        "rows": int(len(df_st)),
        "unique_station_raw": int(df_st["station_raw"].nunique()) if not df_st.empty else 0,
        "missing_station_norm": int(df_st["station_norm"].isna().sum()) if "station_norm" in df_st else 0,
        "dt_min": (df_st["dt"].min().isoformat() if not df_st.empty and "dt" in df_st and not df_st["dt"].isna().all() else None),
        "dt_max": (df_st["dt"].max().isoformat() if not df_st.empty and "dt" in df_st and not df_st["dt"].isna().all() else None),
        "tz": (str(df_st["dt"].dt.tz) if not df_st.empty and "dt" in df_st else None),
        "head": _df_head_jsonable(df_st, head) if head > 0 else [],
        "columns_ok": set(["station_raw","station_norm","dt"]).issubset(df_st.columns),
    }

    # Durations summary
    dur_summary = {
        "rows": int(len(df_dur)),
        "stages": (sorted(df_dur["stage"].unique().tolist()) if not df_dur.empty else []),
        "duration_s_min": (float(df_dur["duration_s"].min()) if not df_dur.empty else None),
        "duration_s_max": (float(df_dur["duration_s"].max()) if not df_dur.empty else None),
        "head": _df_head_jsonable(df_dur, head) if head > 0 and not df_dur.empty else [],
    }

    overall_ok = (ttn_summary["rows"] > 0 and st_summary["rows"] > 0 and ttn_summary["columns_ok"] and st_summary["columns_ok"]) 

    return {
        "ok": overall_ok,
        "ttn": ttn_summary,
        "stations": st_summary,
        "durations": dur_summary,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Astarta data ingestion")
    parser.add_argument("--ttn", type=Path, required=True, help="Path to TTN file (.xls/.xlsx/.csv)")
    parser.add_argument("--stations", type=Path, required=True, help="Path to stations file or directory")
    parser.add_argument("--mapping", type=Path, default=None, help="Optional JSON mapping file")
    parser.add_argument("--chunksize", type=int, default=200_000)
    parser.add_argument("--tz", type=str, default="Europe/Kyiv")
    parser.add_argument("--head", type=int, default=5, help="Print first N rows for quick glance")
    parser.add_argument("--use-norm", action="store_true", help="Use normalized step names (step_norm) for durations if available")
    args = parser.parse_args()

    if not args.ttn.exists():
        print(f"ERROR: TTN path not found: {args.ttn}")
        return 2
    if not args.stations.exists():
        print(f"ERROR: Stations path not found: {args.stations}")
        return 2
    if args.mapping is not None and not args.mapping.exists():
        print(f"WARNING: Mapping file not found: {args.mapping}; proceeding without mapping")
        args.mapping = None

    try:
        result = ingest_astarta(
            ttn_path=args.ttn,
            station_paths=args.stations,
            mapping_json=args.mapping,
            chunksize=args.chunksize,
            tz=args.tz,
        )
    except Exception as e:
        print("ERROR: Ingestion failed:")
        print(repr(e))
        return 1

    df_ttn = result.get("ttn", pd.DataFrame())
    df_st = result.get("stations", pd.DataFrame())

    print(summarize_ttn(df_ttn))
    if not df_ttn.empty and args.head > 0:
        print("\nTTN head:")
        print(df_ttn.head(args.head).to_string(index=False))

    print("\n" + summarize_stations(df_st))
    if not df_st.empty and args.head > 0:
        print("\nStations head:")
        print(df_st.head(args.head).to_string(index=False))

    # Durations computation and summary
    try:
        df_dur = compute_stage_durations(df_ttn, DEFAULT_STAGES, use_norm=args.use_norm)
    except Exception as e:
        df_dur = pd.DataFrame()
        print("\nDurations: ERROR during computation:")
        print(repr(e))
    if not df_dur.empty:
        print("\nDurations:")
        print(f"  rows: {len(df_dur):,}")
        print(f"  stages: {', '.join(sorted(df_dur['stage'].unique().tolist()))}")
        print(f"  duration_s: min={df_dur['duration_s'].min():.0f}, max={df_dur['duration_s'].max():.0f}")
        if args.head > 0:
            print("\nDurations head:")
            print(df_dur.head(args.head).to_string(index=False))
        # Per-stage summary similar to notebook
        df_sum = summarize_stage_durations(df_dur)
        if not df_sum.empty:
            print("\nPer-stage summary (minutes):")
            print(df_sum.to_string(index=False, formatters={
                'mean_min': '{:.2f}'.format,
                'median_min': '{:.2f}'.format,
                'p90_min': '{:.2f}'.format,
                'total_hours': '{:.2f}'.format,
            }))
    else:
        print("\nDurations: 0 rows (no valid start→next-start pairs found)")

    # Basic integrity checks -> non-zero rows and required columns present
    ttn_ok = set(["ttn_id","step_order","step_name_raw","step_norm","ts","operator"]).issubset(df_ttn.columns)
    st_ok = set(["station_raw","station_norm","dt"]).issubset(df_st.columns)
    if df_ttn.empty or df_st.empty or not ttn_ok or not st_ok:
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())


