# app.py (Streamlit UI -> FastAPI /analyze)
import base64
import requests
import streamlit as st
import pandas as pd  # for rendering the stage table

st.set_page_config(page_title="Astarta Data Analyzer", layout="wide")
st.title("Astarta Data Analyzer 🌽")

st.markdown(
    "Upload a CSV/Excel and click **Start analysis**. "
    "The file is sent to the API; results (summary, bottleneck, tables, charts) are rendered here."
)

# --- Primary button color (blue) ---
st.markdown("""
<style>
.stButton > button[kind="primary"]{
  background-color:#3B82F6;border-color:#3B82F6;color:white;
}
.stButton > button[kind="primary"]:hover{filter:brightness(0.95);}
</style>
""", unsafe_allow_html=True)

# --- API base URL in sidebar ---
with st.sidebar:
    api_base = st.text_input("API base URL", value="http://localhost:8000").rstrip("/")
    st.caption("FastAPI expected at `/analyze`")

# --- File upload ---
file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx", "xls"])

# Keep last analysis result in session
if "analysis" not in st.session_state:
    st.session_state.analysis = None

def call_api_analyze(api_base_url: str, file_obj) -> dict:
    files = {
        # FastAPI endpoint expects field name "file"
        "file": (file_obj.name, file_obj.getvalue(), getattr(file_obj, "type", "application/octet-stream"))
    }
    resp = requests.post(f"{api_base_url}/analyze", files=files, timeout=300)
    if resp.status_code != 200:
        try:
            detail = resp.json().get("detail")
        except Exception:
            detail = resp.text
        raise RuntimeError(f"API error {resp.status_code}: {detail}")
    return resp.json()

# --- Start button ---
start = st.button("Start analysis", type="primary", disabled=(file is None or not api_base))

if start:
    if file is None:
        st.warning("Please upload a file first.")
        st.stop()
    with st.spinner("Analyzing via API..."):
        try:
            result = call_api_analyze(api_base, file)
            st.session_state.analysis = result
            st.success("Analysis complete.")
        except Exception as e:
            st.session_state.analysis = None
            st.error(str(e))

# --- Render results (from API) ---
res = st.session_state.analysis
if not res:
    st.stop()

# Summary
st.header("Bottleneck analysis")
summary = res.get("summary", {})
cols = st.columns(5)
cols[0].metric("TTN count", f"{summary.get('ttn_count', 0):,}")
cols[1].metric("Events", f"{summary.get('events_count', 0):,}")
cols[2].metric("Period start", str(summary.get("period_start", "—")))
cols[3].metric("Period end", str(summary.get("period_end", "—")))
cols[4].metric("Median cycle", summary.get("cycle_time_median") or "—")

# Bottleneck
bn = res.get("bottleneck", {})
st.subheader("Top bottleneck")
if bn:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Stage", bn.get("stage", "—"))
    c2.metric("P95", bn.get("p95", "—"))
    c3.metric("Median", bn.get("median", "—"))
    c4.metric("Mean", bn.get("mean", "—"))
else:
    st.info("No bottleneck identified.")

# Stage table
stage_rows = res.get("stage_table", [])
if stage_rows:
    st.subheader("Stage statistics")
    st.dataframe(pd.DataFrame(stage_rows), use_container_width=True, height=360)
else:
    st.caption("No stage statistics returned.")

# Charts
st.header("Overall analysis")
charts = res.get("charts", {})

box = charts.get("stage_boxplot")
bar = charts.get("bottleneck_barplot")
cum = charts.get("cumulative_time")

def show_chart(b64img: str, caption: str):
    st.image(base64.b64decode(b64img), caption=caption, width=1000)

if box:
    show_chart(box, "Stage duration distribution (boxplot)")
if bar:
    show_chart(bar, "Average duration by stage (bottleneck visualization)")
if cum:
    show_chart(cum, "Cumulative process time (total growth by stage)")
