# =============================================================================
# make_features_rule_based.py
# Language: Python 3
#
# S&P 500 Feature Engineering + Simple Rule-Based Anomaly Flags (2014–2024)
# -----------------------------------------------------------------------------
# Purpose
#   Build rolling technical features and basic rule-based anomaly flags from a
#   daily S&P 500 panel (one CSV), then export a features file and yearly stats.
#
# Ownership / Rights
#   © 2025 (All rights reserved). This code is authored for an academic thesis.
#   You may read and review this script; redistribution of any third-party data
#   (e.g., WRDS/CRSP-derived CSVs) is not permitted. Publish the code, not data.
#
# Inputs (same folder)
#   - sp500_prices_with_names_2014_2024_FULL.csv
#       Columns used: permno, date, close, volume, crsp_ret, ticker, comnam,
#                     siccd, in_index_flag
#
# Outputs (same folder)
#   - sp500_anomaly_features_2015_2024.csv
#   - B1_rule_based_anomaly_rates_by_year.csv
#
# Notes
#   - Features start from 2015 to avoid partial lookbacks biasing rolling stats.
#   - If crsp_ret is missing, a simple close-to-close pct change is used.
#   - Do NOT commit input/output CSVs to a public repo; commit the code only.
# =============================================================================

import os
import numpy as np
import pandas as pd

# ---------------------- Parameters ----------------------
IN_PATH  = "sp500_prices_with_names_2014_2024_FULL.csv"

# Rolling window lengths (in trading days)
W_RET_SHORT = 5      # ~1 week cumulative return
W_RET_MED   = 20     # ~1 month cumulative return
W_VOL       = 20     # rolling stdev of daily returns
W_Z         = 20     # mean/std window for z-score of daily return
W_RSI       = 14     # RSI window (standard)
W_BB        = 20     # Bollinger Band mid/std window
W_MOM_S     = 21     # short momentum (approx. 1 month)
W_MOM_L     = 63     # long momentum  (approx. 3 months)

# Rule-based thresholds (simple, interpretable)
ZSCORE_THRESH  = 3.0          # |z| > 3 => extreme daily return
VOL_MULTIPLIER = 5.0          # volume anomaly: vol > 5 × avg_vol_20
RSI_LOW, RSI_HIGH = 20.0, 80.0

# Optional per-ticker, two-sided winsorization of daily returns
# Set to e.g. 0.01 for 1%/99% winsor, or leave as None (off)
WINSOR_PCT = None

# ---------------------- Helper functions ----------------
def _safe_pct_change(s: pd.Series) -> pd.Series:
    """Fallback: simple percentage change. Assumes input is numeric and ordered."""
    return s.pct_change()

def compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """Classic RSI using EMA of gains/losses; returns 0..100. NaNs filled with 50."""
    delta = series.diff()
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    gain_ema = pd.Series(gain, index=series.index).ewm(alpha=1/window, adjust=False).mean()
    loss_ema = pd.Series(loss, index=series.index).ewm(alpha=1/window, adjust=False).mean()
    rs = gain_ema / (loss_ema.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)

def bollinger_bandwidth(series: pd.Series, window: int = 20) -> pd.Series:
    """(Upper-Lower)/Middle where Middle=rolling mean and Upper/Lower=±2σ."""
    mid = series.rolling(window, min_periods=window).mean()
    std = series.rolling(window, min_periods=window).std(ddof=0)
    upper = mid + 2 * std
    lower = mid - 2 * std
    width = (upper - lower) / mid
    return width

def winsorize_series(s: pd.Series, p: float = 0.01) -> pd.Series:
    """Clip series to [p, 1-p] quantiles. Use p=None or <=0 to disable."""
    if p is None or p <= 0:
        return s
    lower = s.quantile(p)
    upper = s.quantile(1 - p)
    return s.clip(lower, upper)

# ---------------------- Load data -----------------------
usecols = [
    "permno","date","close","volume","crsp_ret",
    "ticker","comnam","siccd","in_index_flag"
]
df = pd.read_csv(IN_PATH, usecols=usecols, parse_dates=["date"])
df = df.sort_values(["permno","date"]).reset_index(drop=True)

# Ensure numeric types for math operations
for col in ["close","volume","crsp_ret"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# If CRSP daily return is missing, fall back to close-to-close return
df["ret_fallback"] = df.groupby("permno", group_keys=False)["close"].apply(_safe_pct_change)
df["ret_use"] = df["crsp_ret"].where(~df["crsp_ret"].isna(), df["ret_fallback"])

# ---------------- Per-PERMNO rolling features -----------
def per_permno_features(g: pd.DataFrame) -> pd.DataFrame:
    """Compute rolling features per security; assumes g is a single PERMNO."""
    g = g.sort_values("date").copy()

    # Optional robustness: winsorize returns to limit extreme influence
    if WINSOR_PCT is not None:
        g["ret_use"] = winsorize_series(g["ret_use"], WINSOR_PCT)

    # Rolling cumulative returns
    g[f"roll_ret_{W_RET_SHORT}"] = (1 + g["ret_use"]).rolling(W_RET_SHORT, min_periods=W_RET_SHORT).apply(np.prod, raw=True) - 1
    g[f"roll_ret_{W_RET_MED}"]   = (1 + g["ret_use"]).rolling(W_RET_MED,   min_periods=W_RET_MED).apply(np.prod, raw=True) - 1

    # Rolling volatility (std) of daily returns
    g[f"roll_vol_{W_VOL}"] = g["ret_use"].rolling(W_VOL, min_periods=W_VOL).std(ddof=0)

    # Daily-return z-score vs rolling mean/std
    mean_w = g["ret_use"].rolling(W_Z, min_periods=W_Z).mean()
    std_w  = g["ret_use"].rolling(W_Z, min_periods=W_Z).std(ddof=0)
    g[f"zscore_{W_Z}"] = (g["ret_use"] - mean_w) / std_w.replace(0, np.nan)

    # Momentum windows (cumulative)
    g[f"momentum_{W_MOM_S}"] = (1 + g["ret_use"]).rolling(W_MOM_S, min_periods=W_MOM_S).apply(np.prod, raw=True) - 1
    g[f"momentum_{W_MOM_L}"] = (1 + g["ret_use"]).rolling(W_MOM_L, min_periods=W_MOM_L).apply(np.prod, raw=True) - 1

    # RSI on close
    g["rsi_14"] = compute_rsi(g["close"], window=W_RSI)

    # Bollinger bandwidth on price
    g["bb_width"] = bollinger_bandwidth(g["close"], window=W_BB)

    # Volume anomaly vs 20-day average
    avg_vol20 = g["volume"].rolling(20, min_periods=20).mean()
    g["vol_anomaly"] = (g["volume"] > VOL_MULTIPLIER * avg_vol20).astype(int)

    # Robust Bollinger expansion flag (median + MAD-like threshold)
    bw = g["bb_width"]
    med = bw.rolling(252, min_periods=60).median()
    mad = (bw - med).abs().rolling(252, min_periods=60).median()
    g["bb_expansion_anom"] = (bw > (med + 2.0 * mad.fillna(0))).astype(int)

    return g

# Apply per security (grouped by PERMNO)
features = df.groupby("permno", group_keys=False).apply(per_permno_features)

# ---------------- Rule-based anomaly flags ---------------
# Simple, interpretable boolean flags
features["anomaly_flag"] = (features[f"zscore_{W_Z}"].abs() > ZSCORE_THRESH).astype(int)
features["rsi_anomaly"]  = ((features["rsi_14"] < RSI_LOW) | (features["rsi_14"] > RSI_HIGH)).astype(int)

# ---------------- Housekeeping / Export ------------------
# Keep only years with full lookbacks (avoid partial-window bias)
features["year"] = features["date"].dt.year
features_out = features[features["year"] >= 2015].copy()

# Column ordering for a tidy export
cols_first = [
    "permno","date","ticker","comnam","siccd","in_index_flag","year",
    "close","volume","crsp_ret","ret_use"
]
feat_cols = [
    f"roll_ret_{W_RET_SHORT}", f"roll_ret_{W_RET_MED}",
    f"roll_vol_{W_VOL}",
    f"zscore_{W_Z}",
    f"momentum_{W_MOM_S}", f"momentum_{W_MOM_L}",
    "rsi_14", "bb_width",
    "vol_anomaly", "rsi_anomaly", "bb_expansion_anom",
    "anomaly_flag"
]
keep_cols = [c for c in cols_first + feat_cols if c in features_out.columns]
features_out = features_out[keep_cols].reset_index(drop=True)

# Save main features file
out_main = "sp500_anomaly_features_2015_2024.csv"
features_out.to_csv(out_main, index=False)

# Quick diagnostics by year: count of observations and share with any rule flag
diag = (
    features_out
      .assign(any_rule_anom=lambda x: (
          (x.get("anomaly_flag", 0) == 1) |
          (x.get("vol_anomaly", 0) == 1) |
          (x.get("rsi_anomaly", 0) == 1) |
          (x.get("bb_expansion_anom", 0) == 1)
      ).astype(int))
      .groupby("year")
      .agg(
          n_obs=("permno", "count"),
          n_any_rule_anom=("any_rule_anom", "sum"),
          pct_any_rule_anom=("any_rule_anom", lambda s: 100 * s.mean())
      )
      .reset_index()
)
diag.to_csv("B1_rule_based_anomaly_rates_by_year.csv", index=False)

print("✅ Done.")
print(f"Main features: {out_main}")
print("Diagnostics:   B1_rule_based_anomaly_rates_by_year.csv")
