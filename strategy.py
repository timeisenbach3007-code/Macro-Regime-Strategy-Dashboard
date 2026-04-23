"""
Macro Regime Timing Strategy — reusable module.
Extracted from Code_Python_for_Finance.ipynb.

All data fetching is cached by the caller (e.g., Streamlit) to avoid
hitting Ken French / FRED on every page load.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pandas_datareader.data as web

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
TRAIN_START = "2000-01"
TRAIN_END   = "2015-12"
TEST_START  = "2016-01"
DEFAULT_FEE = 0.003      # 30 bp per unit of turnover
N_BINS      = 5
THRESHOLD_PCT = 70       # percentile for high-risk threshold (picked on train)


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def load_ff25(start="1999-01", end=None) -> pd.DataFrame:
    ff25 = web.DataReader("25_Portfolios_5x5", "famafrench", start=start, end=end)[0].copy()
    ff25 = ff25 / 100
    ff25.index = ff25.index.to_timestamp("M")
    return ff25


def load_ff3(start="1999-01", end=None) -> pd.DataFrame:
    ff = web.DataReader("F-F_Research_Data_Factors", "famafrench", start=start, end=end)[0].copy()
    ff = ff / 100
    ff.index = ff.index.to_timestamp("M")
    ff["Mkt"] = ff["RF"] + ff["Mkt-RF"]
    ff = ff.rename(columns={"Mkt-RF": "mkt_rf"})
    return ff


def load_fred_macro(start="1998-01-01", end=None) -> pd.DataFrame:
    term   = web.DataReader("T10Y2Y",  "fred", start, end)
    credit = web.DataReader("BAA10YM", "fred", start, end)
    unrate = web.DataReader("UNRATE",  "fred", start, end)
    cpi    = web.DataReader("CPIAUCSL","fred", start, end)

    term_m   = term.resample("ME").last().rename(columns={"T10Y2Y": "term_spread"})
    credit_m = credit.resample("ME").last().rename(columns={"BAA10YM": "credit_spread"})
    unrate_m = unrate.resample("ME").last().rename(columns={"UNRATE": "unrate"})
    cpi_m    = cpi.resample("ME").last()

    unrate_m["unrate_chg12"] = unrate_m["unrate"] - unrate_m["unrate"].shift(12)
    cpi_m["inflation_yoy"]   = cpi_m["CPIAUCSL"].pct_change(12)

    macro = pd.DataFrame({
        "term_spread":   term_m["term_spread"],
        "credit_spread": credit_m["credit_spread"],
        "unrate_chg12":  unrate_m["unrate_chg12"],
        "inflation_yoy": cpi_m["inflation_yoy"],
    }).dropna()
    return macro


# --------------------------------------------------------------------------- #
# Score + portfolio construction
# --------------------------------------------------------------------------- #
def _expanding_zscore(s: pd.Series) -> pd.Series:
    mu  = s.expanding(min_periods=12).mean()
    sig = s.expanding(min_periods=12).std()
    return (s - mu) / sig


def build_risk_score(macro: pd.DataFrame) -> pd.DataFrame:
    m = macro.copy()
    m["z_term"]   = _expanding_zscore(-m["term_spread"])     # inverted: low spread = more risk
    m["z_credit"] = _expanding_zscore(m["credit_spread"])
    m["z_unemp"]  = _expanding_zscore(m["unrate_chg12"])
    m["z_infl"]   = _expanding_zscore(m["inflation_yoy"])
    m["risk_score"]     = m[["z_term", "z_credit", "z_unemp", "z_infl"]].mean(axis=1)
    m["risk_score_lag"] = m["risk_score"].shift(1)
    return m


def build_risk_portfolios(ff25: pd.DataFrame) -> pd.DataFrame:
    cols = list(ff25.columns)
    grid = np.array(cols).reshape(5, 5)
    risk_on_cols  = [grid[0, 0], grid[0, 1], grid[1, 0], grid[1, 1]]   # small + growth
    risk_off_cols = [grid[3, 3], grid[3, 4], grid[4, 3], grid[4, 4]]   # big   + value
    out = ff25.copy()
    out["r_on"]  = out[risk_on_cols].mean(axis=1)
    out["r_off"] = out[risk_off_cols].mean(axis=1)
    return out[["r_on", "r_off"]]


def build_master(
    ff25: pd.DataFrame, ff: pd.DataFrame, macro: pd.DataFrame,
    start=TRAIN_START, end=None,
) -> pd.DataFrame:
    r = build_risk_portfolios(ff25).copy(); r["date"] = r.index
    f = ff[["mkt_rf", "SMB", "HML", "RF", "Mkt"]].copy(); f["date"] = f.index
    m = build_risk_score(macro)[["risk_score", "risk_score_lag"]].copy(); m["date"] = m.index

    master = pd.merge(r, f, on="date", how="inner")
    master = pd.merge(master, m, on="date", how="inner").set_index("date")
    master["r_benchmark"] = master["Mkt"]
    master = master.dropna(subset=["r_on", "r_off", "r_benchmark", "risk_score_lag"])
    return master.loc[start:end]


# --------------------------------------------------------------------------- #
# Markov-chain transition table + allocation
# --------------------------------------------------------------------------- #
def select_threshold(train: pd.DataFrame, pct: int = THRESHOLD_PCT) -> float:
    return float(np.percentile(train["risk_score_lag"].dropna(), pct))


def build_transition_table(train: pd.DataFrame, threshold: float, n_bins: int = N_BINS):
    scores = train["risk_score_lag"].dropna()
    edges = np.percentile(scores, np.linspace(0, 100, n_bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf

    df = train.copy()
    df["state"] = pd.cut(df["risk_score_lag"], bins=edges, labels=range(1, n_bins + 1))
    df["next_high"] = np.where(df["risk_score"] > threshold, 1, 0)
    table = df.dropna(subset=["risk_score"]).groupby("state", observed=False)["next_high"].agg(["mean", "count"])
    table.columns = ["prob_high_next", "n_months"]
    return edges, table


def apply_allocation(df: pd.DataFrame, bin_edges, trans_table) -> pd.DataFrame:
    n_bins = len(bin_edges) - 1
    out = df.copy()
    out["state"] = pd.cut(out["risk_score_lag"], bins=bin_edges, labels=range(1, n_bins + 1))
    lookup = trans_table["prob_high_next"].to_dict()
    out["w_off"] = out["state"].map(lookup).astype(float)
    out["w_on"]  = 1 - out["w_off"]
    return out


def compute_strategy(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["strategy_ret"] = out["w_on"] * out["r_on"] + out["w_off"] * out["r_off"]
    out["pos"] = out["w_off"]
    return out


def add_fees(df: pd.DataFrame, fee_rate: float = DEFAULT_FEE) -> pd.DataFrame:
    out = df.copy()
    out["turnover"] = out["pos"].diff().abs().fillna(0)
    out["fee"] = out["turnover"] * fee_rate
    out["strategy_ret_net"] = out["strategy_ret"] - out["fee"]
    out["cumret_strat"]     = (1 + out["strategy_ret"]).cumprod()
    out["cumret_strat_net"] = (1 + out["strategy_ret_net"]).cumprod()
    out["cumret_bench"]     = (1 + out["r_benchmark"]).cumprod()
    return out


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def compute_metrics(monthly_returns: pd.Series, rf: pd.Series) -> dict:
    excess = monthly_returns - rf
    ann_ret = excess.mean() * 12
    ann_vol = excess.std() * np.sqrt(12)
    sharpe  = ann_ret / ann_vol if ann_vol > 0 else 0.0
    cum     = (1 + monthly_returns).cumprod()
    mdd     = ((cum - cum.cummax()) / cum.cummax()).min()
    cagr    = cum.iloc[-1] ** (12 / len(monthly_returns)) - 1
    hit     = (monthly_returns > 0).mean()
    return {
        "cagr": cagr, "ann_excess": ann_ret, "ann_vol": ann_vol,
        "sharpe": sharpe, "max_dd": mdd, "hit_ratio": hit,
    }


def drawdown_series(monthly_returns: pd.Series) -> pd.Series:
    cum = (1 + monthly_returns).cumprod()
    return (cum - cum.cummax()) / cum.cummax()


def rolling_sharpe(monthly_returns: pd.Series, rf: pd.Series, window: int = 12) -> pd.Series:
    excess = monthly_returns - rf
    mu  = excess.rolling(window).mean() * 12
    sig = excess.rolling(window).std() * np.sqrt(12)
    return mu / sig


# --------------------------------------------------------------------------- #
# End-to-end helper
# --------------------------------------------------------------------------- #
def run_strategy(master: pd.DataFrame, fee_rate: float = DEFAULT_FEE) -> dict:
    """
    Full pipeline: fit on train, apply on train & test, return a dict
    with all series + metrics the UI needs.
    """
    train = master.loc[TRAIN_START:TRAIN_END].copy()
    test  = master.loc[TEST_START:].copy()

    thr = select_threshold(train)
    edges, table = build_transition_table(train, thr, N_BINS)

    full = add_fees(compute_strategy(apply_allocation(master, edges, table)), fee_rate)
    tr   = full.loc[TRAIN_START:TRAIN_END]
    te   = full.loc[TEST_START:]

    return {
        "threshold": thr,
        "bin_edges": edges,
        "trans_table": table,
        "full": full,
        "train": tr,
        "test": te,
        "metrics_train": compute_metrics(tr["strategy_ret_net"], tr["RF"]),
        "metrics_test":  compute_metrics(te["strategy_ret_net"], te["RF"]),
        "metrics_bench_test": compute_metrics(te["r_benchmark"], te["RF"]),
    }
