"""
Streamlit dashboard for the Macro Regime Timing Strategy.

Run locally:
    streamlit run app.py

Deploy free on Streamlit Community Cloud: https://streamlit.io/cloud
"""
from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import strategy as strat

# --------------------------------------------------------------------------- #
# Page config
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="Macro Regime Timing — Live",
    page_icon="📈",
    layout="wide",
)

st.title("Macro Regime Timing Strategy - Dashboard")
st.caption(
    "As a part of our Python for Finance Class at Católica Lisbon School of Business and Economics we " \
    "defined the following Macro Regime Stratgey. It contains of a Forward-looking macro risk score and Markov-chain allocation between "
    "risk-on (small/growth) and risk-off (big/value) Ken French portfolios. "
    "Data auto-refreshes daily from FRED + Ken French."
)

# --------------------------------------------------------------------------- #
# Cached data loaders — TTL = 1 day (FF/FRED update monthly)
# --------------------------------------------------------------------------- #
@st.cache_data(ttl=60 * 60 * 24, show_spinner="Loading Ken French 25 portfolios…")
def cached_ff25():
    return strat.load_ff25(start="1999-01")

@st.cache_data(ttl=60 * 60 * 24, show_spinner="Loading Fama-French 3 factors…")
def cached_ff3():
    return strat.load_ff3(start="1999-01")

@st.cache_data(ttl=60 * 60 * 24, show_spinner="Loading FRED macro series…")
def cached_fred():
    return strat.load_fred_macro(start="1998-01-01")

@st.cache_data(ttl=60 * 60 * 24)
def cached_master(_ff25, _ff, _macro):
    return strat.build_master(_ff25, _ff, _macro)


# --------------------------------------------------------------------------- #
# Sidebar controls
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("Parameters")
    fee_bp = st.slider("Transaction fee (bp per unit turnover)", 0, 100, 30, step=5)
    st.caption("Applied to monthly turnover of the risk-off weight.")
    st.divider()
    if st.button("🔄 Force refresh data"):
        st.cache_data.clear()
        st.rerun()
    st.caption("Data caches for 24h. Ken French + FRED update ~monthly.")

fee_rate = fee_bp / 10_000

# --------------------------------------------------------------------------- #
# Load + run
# --------------------------------------------------------------------------- #
try:
    ff25  = cached_ff25()
    ff    = cached_ff3()
    macro = cached_fred()
    master = cached_master(ff25, ff, macro)
except Exception as e:
    st.error(f"Data load failed: {e}")
    st.stop()

result = strat.run_strategy(master, fee_rate=fee_rate)
full   = result["full"]
test   = result["test"]
train  = result["train"]
m_test  = result["metrics_test"]
m_bench = result["metrics_bench_test"]

# --------------------------------------------------------------------------- #
# Header: latest allocation + last data date
# --------------------------------------------------------------------------- #
latest = full.iloc[-1]
latest_date = full.index[-1].strftime("%Y-%m")

st.subheader(f"Current signal  —  latest data: {latest_date}")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Risk-Off weight", f"{latest['w_off']*100:.1f}%")
c2.metric("Risk-On weight",  f"{latest['w_on']*100:.1f}%")
c3.metric("Risk score",      f"{latest['risk_score']:.2f}")
c4.metric("State (1 = calm, 5 = stressed)", f"{int(latest['state'])}")

st.divider()

# --------------------------------------------------------------------------- #
# KPI row — TEST period (out-of-sample)
# --------------------------------------------------------------------------- #
st.subheader(f"Out-of-sample performance ({strat.TEST_START} → {latest_date})")

def _delta(strat_val, bench_val, pct=False):
    diff = strat_val - bench_val
    return f"{diff*100:+.2f} pp" if pct else f"{diff:+.2f}"

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("CAGR",        f"{m_test['cagr']*100:.2f}%",
          _delta(m_test["cagr"], m_bench["cagr"], pct=True))
k2.metric("Ann. vol",    f"{m_test['ann_vol']*100:.2f}%",
          _delta(m_test["ann_vol"], m_bench["ann_vol"], pct=True), delta_color="inverse")
k3.metric("Sharpe",      f"{m_test['sharpe']:.2f}",
          _delta(m_test["sharpe"], m_bench["sharpe"]))
k4.metric("Max drawdown",f"{m_test['max_dd']*100:.2f}%",
          _delta(m_test["max_dd"], m_bench["max_dd"], pct=True))
k5.metric("Hit ratio",   f"{m_test['hit_ratio']*100:.1f}%",
          _delta(m_test["hit_ratio"], m_bench["hit_ratio"], pct=True))

st.divider()

# --------------------------------------------------------------------------- #
# Charts
# --------------------------------------------------------------------------- #
tab1, tab2, tab3, tab4 = st.tabs(
    ["Cumulative returns", "Risk score & allocation", "Rolling Sharpe & drawdown", "Transition matrix"]
)

# --- tab 1: cumulative returns ---
with tab1:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=full.index, y=full["cumret_strat_net"],
        name="Strategy (net)", line=dict(color="steelblue", width=2.5),
    ))
    fig.add_trace(go.Scatter(
        x=full.index, y=full["cumret_strat"],
        name="Strategy (gross)", line=dict(color="steelblue", width=1, dash="dash"),
    ))
    fig.add_trace(go.Scatter(
        x=full.index, y=full["cumret_bench"],
        name="Market benchmark", line=dict(color="gray", width=2),
    ))
    split_date = pd.Timestamp(strat.TRAIN_END)
    fig.add_shape(
        type="line", x0=split_date, x1=split_date, y0=0, y1=1, yref="paper",
        line=dict(color="black", dash="dot", width=1),
    )
    fig.add_annotation(
        x=split_date, y=1, yref="paper", text="Train / Test",
        showarrow=False, yanchor="bottom", font=dict(size=11),
    )
    fig.update_layout(
        height=500, hovermode="x unified",
        yaxis_title="Growth of $1", xaxis_title=None,
        legend=dict(orientation="h", y=1.08),
        margin=dict(l=20, r=20, t=40, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)

# --- tab 2: risk score + allocation ---
with tab2:
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, row_heights=[0.45, 0.55],
        vertical_spacing=0.08,
        subplot_titles=("Composite macro risk score (z-score)", "Risk-off allocation weight"),
    )
    fig.add_trace(go.Scatter(
        x=full.index, y=full["risk_score"],
        name="Risk score", line=dict(color="navy"),
    ), row=1, col=1)
    fig.add_hline(y=result["threshold"], line_dash="dash", line_color="red", row=1, col=1,
                  annotation_text=f"Threshold = {result['threshold']:.2f}")

    fig.add_trace(go.Scatter(
        x=full.index, y=full["w_off"],
        name="Risk-off weight", fill="tozeroy", line=dict(color="coral"),
    ), row=2, col=1)
    fig.update_yaxes(range=[0, 1], row=2, col=1)
    fig.update_layout(
        height=600, hovermode="x unified",
        showlegend=False, margin=dict(l=20, r=20, t=40, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)

# --- tab 3: rolling Sharpe + drawdown ---
with tab3:
    roll = strat.rolling_sharpe(full["strategy_ret_net"], full["RF"], window=12)
    roll_b = strat.rolling_sharpe(full["r_benchmark"],     full["RF"], window=12)
    dd    = strat.drawdown_series(full["strategy_ret_net"])
    dd_b  = strat.drawdown_series(full["r_benchmark"])

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=("Rolling 12M Sharpe ratio", "Drawdown"),
    )
    fig.add_trace(go.Scatter(x=roll.index,   y=roll,   name="Strategy", line=dict(color="steelblue")), row=1, col=1)
    fig.add_trace(go.Scatter(x=roll_b.index, y=roll_b, name="Market",   line=dict(color="gray")),      row=1, col=1)
    fig.add_hline(y=0, line_color="black", line_width=0.5, row=1, col=1)

    fig.add_trace(go.Scatter(x=dd.index,   y=dd,   name="Strategy DD", fill="tozeroy",
                             line=dict(color="steelblue")), row=2, col=1)
    fig.add_trace(go.Scatter(x=dd_b.index, y=dd_b, name="Market DD",   fill="tozeroy",
                             line=dict(color="gray")), row=2, col=1)
    fig.update_layout(height=600, hovermode="x unified",
                      margin=dict(l=20, r=20, t=40, b=20),
                      legend=dict(orientation="h", y=1.08))
    st.plotly_chart(fig, use_container_width=True)

# --- tab 4: Markov transition matrix ---
with tab4:
    # full state-to-state matrix on train
    tr = master.loc[strat.TRAIN_START:strat.TRAIN_END].copy()
    tr["state"] = pd.cut(tr["risk_score_lag"], bins=result["bin_edges"],
                         labels=range(1, strat.N_BINS + 1))
    tr["next_state"] = tr["state"].shift(-1)
    tr = tr.dropna(subset=["state", "next_state"])
    matrix = pd.crosstab(tr["state"], tr["next_state"], normalize="index")

    fig = go.Figure(data=go.Heatmap(
        z=matrix.values, x=[f"State {c}" for c in matrix.columns],
        y=[f"State {i}" for i in matrix.index],
        colorscale="Blues", zmin=0, zmax=1,
        text=np.round(matrix.values, 3), texttemplate="%{text}",
    ))
    fig.update_layout(
        height=500, title="P(state at t+1 | state at t) — estimated on train",
        xaxis_title="To state (next month)", yaxis_title="From state (current)",
        margin=dict(l=20, r=20, t=60, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(result["trans_table"].rename(
        columns={"prob_high_next": "P(next month high-risk)", "n_months": "# months in bin"}
    ).style.format({"P(next month high-risk)": "{:.3f}", "# months in bin": "{:.0f}"}))

st.divider()
st.caption(
    f"Last data point: {latest_date}  ·  Fee: {fee_bp} bp  ·  "
    f"Rendered: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}"
)
