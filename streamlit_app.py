import streamlit as st
import pandas as pd
import json
from huggingface_hub import HfFileSystem
import config
from us_calendar import next_trading_day

st.set_page_config(page_title="Kernel Mean Embedding Engine", layout="wide")
st.markdown("""
<style>
    .main-header { font-size: 2.5rem; font-weight: 700; color: #1f77b4; margin-bottom: 0.5rem; }
    .sub-header { font-size: 1.2rem; color: #555; margin-bottom: 2rem; }
    .universe-title { font-size: 1.5rem; font-weight: 600; margin-top: 1rem; margin-bottom: 1rem; padding-left: 0.5rem; border-left: 5px solid #1f77b4; }
    .etf-card { background: linear-gradient(135deg, #1f77b4 0%, #2c3e50 100%); color: white; border-radius: 15px; padding: 1rem; margin: 0.5rem; text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.2); }
    .etf-ticker { font-size: 1.3rem; font-weight: bold; }
    .etf-score { font-size: 0.9rem; margin-top: 0.3rem; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🎯 Kernel Mean Embedding Engine</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">RKHS embeddings | Maximum Mean Discrepancy (MMD) | Compare ETF distributions to reference | Score = -MMD | Best window per ETF</div>', unsafe_allow_html=True)

st.sidebar.markdown("## 🎯 Kernel Embedding")
st.sidebar.markdown(f"**Run Date:** `{st.session_state.get('run_date', 'Not loaded')}`")
st.sidebar.markdown(f"**Next Trading Day:** `{next_trading_day()}`")
st.sidebar.markdown(f"**Kernel:** {config.KERNEL} | **Gamma:** {config.GAMMA}")
st.sidebar.markdown(f"**Reference:** {config.REFERENCE_TYPE} ({config.BENCHMARK_TICKER if config.REFERENCE_TYPE=='benchmark' else f'low-vol {config.LOW_VOL_PERCENTILE}%'})")
st.sidebar.markdown("**Windows evaluated:** 63, 252, 504, 1008, 2016 days (best per ETF)")

OUTPUT_REPO = config.OUTPUT_REPO
HF_TOKEN = config.HF_TOKEN

@st.cache_data(ttl=3600)
def list_repo_files():
    fs = HfFileSystem(token=HF_TOKEN)
    try:
        files = [f['name'] for f in fs.ls(f"datasets/{OUTPUT_REPO}", detail=True, recursive=True) if f['type'] == 'file']
        return files
    except Exception as e:
        return [f"Error: {e}"]

def find_latest_json(files):
    json_files = [f for f in files if f.endswith('.json') and 'kernel_embedding_' in f]
    if not json_files:
        return None
    json_files.sort(reverse=True)
    return json_files[0]

@st.cache_data(ttl=3600)
def load_json(path):
    fs = HfFileSystem(token=HF_TOKEN)
    try:
        with fs.open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        return {"error": str(e)}

files = list_repo_files()
latest = find_latest_json(files)
if not latest:
    st.error("No results found. Run trainer first.")
    st.stop()

data = load_json(latest)
if "error" in data:
    st.error(f"Error: {data['error']}")
    st.stop()

st.session_state['run_date'] = data['run_date']
universes = data["universes"]

st.header("🏆 Top ETFs by MMD Score (Closest to Reference Distribution)")

with st.expander("📖 Interpretation", expanded=True):
    st.markdown("""
    - **Kernel Mean Embedding** maps probability distributions into a reproducing kernel Hilbert space (RKHS).
    - **Maximum Mean Discrepancy (MMD)** measures the distance between two distributions in that space.
    - The **score** is **-MMD**, so a higher score means the ETF's return distribution is closer to the reference.
    - **Reference** can be a benchmark ETF (e.g., SPY) or a low‑volatility regime (days with lowest volatility).
    - **Higher score → ETF distribution resembles the reference → overweight signal** (e.g., if reference is a desirable low‑vol regime).
    - For each ETF, the rolling window that gives the **highest score** is selected.
    """)

for universe_name, uni_data in universes.items():
    top_etfs = uni_data.get("top_etfs", [])
    if not top_etfs:
        continue
    st.markdown(f'<div class="universe-title">{universe_name.replace("_", " ").title()}</div>', unsafe_allow_html=True)
    cols = st.columns(3)
    for idx, etf in enumerate(top_etfs):
        with cols[idx]:
            st.markdown(f"""
            <div class="etf-card">
                <div class="etf-ticker">{etf['ticker']}</div>
                <div class="etf-score">MMD score = {etf['mmd_score']:.6f}</div>
                <div class="etf-score">best window = {etf.get('best_window', 'N/A')}d</div>
            </div>
            """, unsafe_allow_html=True)
    with st.expander("📋 Full ranking (all ETFs, best window per ETF)"):
        full = uni_data.get("full_scores", {})
        if full:
            rows = []
            for ticker, info in full.items():
                if isinstance(info, dict):
                    score = info.get("score", 0.0)
                    win = info.get("best_window", "N/A")
                else:
                    score = info
                    win = "N/A"
                rows.append({"ETF": ticker, "MMD Score (closer to ref)": score, "Best Window": win})
            df = pd.DataFrame(rows)
            df["MMD Score (closer to ref)"] = pd.to_numeric(df["MMD Score (closer to ref)"], errors='coerce')
            df = df.dropna(subset=["MMD Score (closer to ref)"]).sort_values("MMD Score (closer to ref)", ascending=False)
            st.dataframe(df, use_container_width=True, hide_index=True)
    st.divider()

st.caption("Kernel mean embedding and MMD (Gretton et al., 2012). The engine compares the distribution of each ETF's returns to a reference (benchmark or low‑vol regime) using an RBF or Matérn kernel. Higher score = more similar to reference → overweight signal.")
