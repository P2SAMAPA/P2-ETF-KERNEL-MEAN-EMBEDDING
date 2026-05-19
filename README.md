# Kernel Mean Embedding Engine

Embeds ETF return distributions into a reproducing kernel Hilbert space (RKHS) using kernel mean embeddings. Compares distributions with the reference (benchmark ETF or low‑volatility regime) via Maximum Mean Discrepancy (MMD). Score = -MMD (higher = closer to reference). Multi‑window evaluation selects the best window per ETF.

- **Kernel:** RBF or Matérn (configurable)
- **Reference:** benchmark ETF (e.g., SPY) or low‑volatility days (lowest X% volatility)
- **MMD:** unbiased estimator
- **Windows:** 63, 252, 504, 1008, 2016 days (best per ETF)
- **Output:** top 3 ETFs per universe by MMD score (closest to reference)

Runs daily on GitHub Actions.

## Local execution

```bash
pip install -r requirements.txt
export HF_TOKEN=<your_token>
python trainer.py
streamlit run streamlit_app.py
