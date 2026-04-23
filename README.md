# Macro Regime Timing — Live Dashboard

Streamlit app for the Python for Finance group project. Shows the live macro
regime timing strategy (Markov-chain allocation between Ken French risk-on and
risk-off portfolios), driven by FRED macro indicators.

## Local

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Push this `webapp/` folder to a GitHub repo.
2. Go to https://share.streamlit.io → *New app*.
3. Point it at the repo and `app.py`. Done.

Data is cached for 24h; Ken French & FRED release monthly.

## Files

- `strategy.py` — data loading, risk score, Markov-chain allocation, metrics
- `app.py` — Streamlit UI (KPI cards, Plotly charts, sidebar controls)
- `requirements.txt` — minimal dependency pin
