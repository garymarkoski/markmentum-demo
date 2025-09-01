
# app_mr_clean.py — stat box left (1/3), Graph 1 right (2/3)
# Calibri font everywhere. Default ticker = SPY. All loads are lazy.
# Graph 1 uses the pill buttons below the stat box and applies a 5‑day gutter.

import base64
from pathlib import Path
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import rcParams

# ==============================
# SETTINGS / CONSTANTS
# ==============================

APP_DIR  = Path(__file__).parent
DATA_DIR = APP_DIR / "data"
ASSETS   = APP_DIR / "assets"

# Logo path (make sure the file exists at Markmentum-demo/assets/markmentum_logo.png)
LOGO_PATH = ASSETS / "markmentum_logo.png"

FILE_STATS = DATA_DIR / "qry_graph_data_25.csv"   # stat box
FILE_G1    = DATA_DIR / "qry_graph_data_01.csv"   # Graph 1: Probable Ranges

DEFAULT_TICKER = "SPY"
EXCEL_BLUE   = "#4472C4"
EXCEL_ORANGE = "#FFC000"
EXCEL_GRAY   = "#A6A6A6"

st.set_page_config(page_title="Markmentum Research", layout="wide")
rcParams["font.family"] = ["Calibri", "Segoe UI", "Tahoma", "Arial", "sans-serif"]

# ==============================
# GLOBAL CSS (Calibri + stat box styling)
# ==============================
st.markdown(f"""
<style>
  html, body, [class^="css"], .stMarkdown, .stButton, .stSelectbox, .stTable, .stCaption, .stText {{
    font-family: Calibri, "Segoe UI", Tahoma, Arial, sans-serif !important;
  }}
  .title-text {{ font-weight:700; font-size:18px; text-align:center; margin:0; color:#111; }}

  /* Stat box: locked width and table style */
  .stat-wrapper {{ max-width: 275px; margin: 0 auto; }}
  .tbl {{ border-collapse: collapse; margin-top:2px; width:auto; }}
  .tbl colgroup col {{ width:68px; }}
  .tbl th {{ background:{EXCEL_BLUE}; color:#fff; font-weight:700; padding:2px 4px; border:1px solid #000; font-size:10px; text-align:center; }}
  .tbl td {{ padding:2px 4px; border:1px solid #000; font-size:12px; background:#fff; }}
  .val {{ font-weight:700; font-size:12px; }}
  .right {{ text-align:right; }} .center {{ text-align:center; }}
</style>
""", unsafe_allow_html=True)
st.markdown(
    """
    <style>
    .calibri-text {
        font-family: Calibri, Arial, sans-serif;
        font-size: 18px;
        line-height: 1.6;
    }
    </style>
    """,
    unsafe_allow_html=True
)
# ==============================
# HELPERS
# ==============================
def _image_to_base64(path: Path) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def fmt_px(v):
    try: return f"{float(v):,.2f}"
    except: return "—"

def fmt_pct(v):
    try:
        x = float(v)
        if abs(x) <= 1.0: x *= 100.0
        return f"{x:,.1f}%"
    except: return "—"

def fmt_rr(v):
    try: return f"({abs(float(v)):,.2f})"
    except: return "—"

def fmt_score(v):
    try: return f"{float(v):,.2f}"
    except: return "—"

def fmt_date_long(dt_val) -> str:
    try: return pd.to_datetime(dt_val).strftime("%m/%d/%Y")
    except: return ""

def range_start(end_date: pd.Timestamp, label: str) -> pd.Timestamp | None:
    if label == "3M":
        return end_date - pd.DateOffset(months=3)
    if label == "6M":
        return end_date - pd.DateOffset(months=6)
    if label == "YTD":
        return pd.Timestamp(end_date.year, 1, 1)
    if label == "1Y":
        return end_date - pd.DateOffset(years=1)
    return None  # All

def apply_window_with_gutter(df: pd.DataFrame, label: str, date_col: str = "date", gutter_days: int = 5) -> pd.DataFrame:
    if df.empty: 
        return df
    end = pd.to_datetime(df[date_col]).max()
    start_raw = range_start(end, label)
    if start_raw is None:
        start = df[date_col].min() - pd.Timedelta(days=gutter_days)
        end = end + pd.Timedelta(days=gutter_days)
    else:
        start = max(df[date_col].min(), start_raw - pd.Timedelta(days=gutter_days))
        end   = end + pd.Timedelta(days=gutter_days)
    m = (df[date_col] >= start) & (df[date_col] <= end)
    return df.loc[m].copy()

# ==============================
# LAZY LOADERS (ticker-only, CSV sorted by ticker/date)
# ==============================
@st.cache_data(show_spinner=False)
def load_stats_for_ticker(path: Path, ticker: str) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    out = []
    for chunk in pd.read_csv(path, chunksize=200000):
        cols = {c.lower(): c for c in chunk.columns}
        tcol = cols.get("ticker")
        if tcol is None:
            df = chunk.copy()
            df.columns = [c.strip().lower() for c in df.columns]
            return df
        m = chunk[tcol] == ticker
        if m.any():
            part = chunk.loc[m].copy()
            part.columns = [c.strip().lower() for c in part.columns]
            out.append(part)
        elif out:
            break
    if not out:
        return pd.DataFrame()
    df = pd.concat(out, ignore_index=True)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.sort_values("date")
    return df

@st.cache_data(show_spinner=False)
def load_g1_ticker(path: Path, ticker: str) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    out = []
    for chunk in pd.read_csv(path, chunksize=200000):
        cols = {c.lower(): c for c in chunk.columns}
        tcol = cols.get("ticker")
        m = chunk[tcol] == ticker if tcol else pd.Series([True]*len(chunk))
        if m.any():
            part = chunk.loc[m].copy()
            part.columns = [c.strip().lower() for c in part.columns]
            out.append(part)
        elif out:
            break
    if not out:
        return pd.DataFrame()
    df = pd.concat(out, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df.sort_values("date").reset_index(drop=True)

# ==============================
# HEADER (logo centered)
# ==============================
if LOGO_PATH.exists():
    st.markdown(
        f"""
        <div style="text-align:center; margin-bottom:12px;">
            <img src="data:image/png;base64,{_image_to_base64(LOGO_PATH)}" width="500">
        </div>
        """,
        unsafe_allow_html=True
    )

# ==============================
# ===== STAT BOX (do not modify) START =====
# ==============================
def _row_block(lbl, lo, hi, rr):
    return (
        f"<tr>"
        f"<td style='background:{EXCEL_BLUE};color:#fff;font-weight:700;text-align:center;'>{lbl}</td>"
        f"<td class='right val'>{lo}</td>"
        f"<td class='right val'>{hi}</td>"
        f"<td class='right val'>{rr}</td>"
        f"</tr>"
    )

def render_stat_box(df_stats: pd.DataFrame, ticker: str):
    if df_stats.empty:
        st.info("No stat data found.")
        return
    row = df_stats.iloc[-1]
    latest_date = fmt_date_long(row.get("date"))
    title = row.get("ticker_name", ticker)

    html = f"<div class='title-text'>{title} - {latest_date}</div><div class='stat-wrapper'>"

    html += "<table class='tbl'><colgroup><col><col><col><col></colgroup>"
    html += "<tr><th>Ticker</th><th>Close</th><th>Anchor</th><th>Chng %</th></tr>"
    html += (
        "<tr>"
        f"<td class='center val'>{ticker}</td>"
        f"<td class='right val'>{fmt_px(row.get('close'))}</td>"
        f"<td class='right val'>{fmt_px(row.get('lt_pt_sm'))}</td>"
        f"<td class='right val'>{fmt_pct(row.get('change_pct'))}</td>"
        "</tr></table>"
    )

    html += "<table class='tbl'><colgroup><col><col><col><col></colgroup>"
    html += "<tr><th>Period</th><th>PR Low</th><th>PR High</th><th>R/R Ratio</th></tr>"
    for lbl, pfx in [("Day","day"),("Week","week"),("Month","month")]:
        lo  = fmt_px(row.get(f"{pfx}_pr_low"))
        hi  = fmt_px(row.get(f"{pfx}_pr_high"))
        rr  = fmt_rr(row.get(f"{pfx}_rr_ratio"))
        html += _row_block(lbl, lo, hi, rr)
    html += "</table>"

    ivol_pd_value = row.get("ivol_pd", row.get("prem_disc"))
    html += "<table class='tbl'><colgroup><col><col><col><col></colgroup>"
    html += "<tr><th>Ivol</th><th>Rvol</th><th>Ivol P/D</th><th>Model Score</th></tr>"
    html += (
        "<tr>"
        f"<td class='right val'>{fmt_pct(row.get('ivol'))}</td>"
        f"<td class='right val'>{fmt_pct(row.get('rvol'))}</td>"
        f"<td class='right val'>{fmt_pct(ivol_pd_value)}</td>"
        f"<td class='right val'>{fmt_score(row.get('model_score'))}</td>"
        "</tr></table></div>"
    )

    st.markdown(html, unsafe_allow_html=True)
# ==============================
# ===== STAT BOX (do not modify) END =====
# ==============================

# ==============================
# GRAPH 1 — Probable Ranges
# ==============================
def plot_probable_ranges(dfv: pd.DataFrame, ticker: str):
    # Ensure order and expected columns
    dfv = dfv.sort_values("date")

    fig, ax = plt.subplots(figsize=(13, 5))
    fig.subplots_adjust(left=0.07, right=0.985, top=0.88, bottom=0.33)

    # Series
    ax.plot(dfv["date"], dfv["close"],         color=EXCEL_BLUE,   linewidth=1.6)
    ax.plot(dfv["date"], dfv["day_pr_low"],    color=EXCEL_GRAY,   linewidth=1.0)
    ax.plot(dfv["date"], dfv["day_pr_high"],   color=EXCEL_GRAY,   linewidth=1.0)
    ax.plot(dfv["date"], dfv["week_pr_low"],   color=EXCEL_ORANGE, linewidth=1.4)
    ax.plot(dfv["date"], dfv["week_pr_high"],  color=EXCEL_ORANGE, linewidth=1.4)
    ax.plot(dfv["date"], dfv["month_pr_low"],  color="black",      linewidth=2.0)
    ax.plot(dfv["date"], dfv["month_pr_high"], color="black",      linewidth=2.0)

    # X-range gutter
    xmin = dfv["date"].min() - pd.Timedelta(days=5)
    xmax = dfv["date"].max() + pd.Timedelta(days=5)
    ax.set_xlim(xmin, xmax)

    # --- Exact bi-weekly Monday ticks that line up with grid ---
    start = dfv["date"].min().normalize()
    end   = dfv["date"].max().normalize()
    # align start to the Monday on/before the first date
    start_monday = start - pd.Timedelta(days=start.weekday())  # Monday=0
    ticks = pd.date_range(start=start_monday, end=end, freq="2W-MON")
    ax.set_xticks([t.to_pydatetime() for t in ticks])

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d/%y"))
    for lbl in ax.get_xticklabels():
        lbl.set_rotation(90)
        lbl.set_ha("center")
        lbl.set_va("top")
        lbl.set_fontsize(9)
    ax.tick_params(axis="x", pad=6)
    # -----------------------------------------------------------

    ax.set_title(f"{ticker} – Probable Ranges", fontsize=14, loc="center")

    legend_handles = [
        plt.Line2D([0], [0], color=EXCEL_BLUE,   linewidth=1.6),
        plt.Line2D([0], [0], color=EXCEL_GRAY,   linewidth=1.0),
        plt.Line2D([0], [0], color=EXCEL_ORANGE, linewidth=1.4),
        plt.Line2D([0], [0], color="black",      linewidth=2.0),
    ]
    ax.legend(
        legend_handles, ["Close", "Day PR", "Week PR", "Month PR"],
        loc="upper center", bbox_to_anchor=(0.5, -0.20),
        ncol=4, frameon=False, fontsize=9
    )

    ax.grid(True, alpha=0.3)
    plt.close(fig)
    return fig

# ==============================
# LAYOUT — Left (1/3) stat box + controls; Right (2/3) Graph 1
# ==============================
left, mid, right_gutter = st.columns([1, 2, 1], gap="large")

with left:
    # vertical spacer to align stat box vertically with Graph 1
    st.markdown("<div style='height:0px'></div>", unsafe_allow_html=True)

    
    # ----- Range pill buttons ABOVE the stat box (inline) -----
    RANGE_OPTIONS = ["3M", "6M", "YTD", "1Y", "All"]
    if "range_sel" not in st.session_state:
        st.session_state["range_sel"] = "All"
    try:
        bcol1, bcol2, bcol3 = st.columns([1.2, 3, 0.8])
        with bcol2:

            st.segmented_control(
                "    Range",
                options=RANGE_OPTIONS,
                key="range_sel",
                label_visibility="collapsed",
            )
    except AttributeError:
        bcol1, bcol2, bcol3 = st.columns([1, 3, 1])
        with bcol2:
            st.radio(
                "Range",
                options=RANGE_OPTIONS,
                key="range_sel",
                horizontal=True,
                label_visibility="collapsed",
            )

    # ----- Stat box -----
    current_ticker = st.session_state.get("ticker_select", DEFAULT_TICKER)
    stats_df = load_stats_for_ticker(FILE_STATS, current_ticker)
    render_stat_box(stats_df, current_ticker)

    # ----- Dropdown under stat box (fit ~9 chars) -----
    tickers = [DEFAULT_TICKER]
    try:
        for chunk in pd.read_csv(FILE_STATS, chunksize=100000, usecols=["Ticker"]):
            tickers.extend(chunk["Ticker"].dropna().astype(str).unique().tolist())
    except Exception:
        pass
    tickers = sorted(set(tickers))
    try:
        default_index = tickers.index(current_ticker)
    except ValueError:
        default_index = 0

    dcol1, dcol2, dcol3 = st.columns([2, 1, 2])
    with dcol2:
        st.selectbox(
            "Ticker",
            options=tickers,
            index=default_index,
            key="ticker_select",
            label_visibility="collapsed",
        )
with mid:
    ticker = st.session_state.get("ticker_select", DEFAULT_TICKER)
    df_all = load_g1_ticker(FILE_G1, ticker)
    df_view = apply_window_with_gutter(df_all, st.session_state.get("range_sel", "All"), date_col="date", gutter_days=5)
    if df_view.empty:
        st.info("No data for selected range.")
    else:
        st.pyplot(plot_probable_ranges(df_view, ticker), clear_figure=True)
# ==============================
# ===== STAT BOX & Graph1 (do not modify) END =====
# ==============================

# ==============================
# BOTTOM ROW — Graphs 2, 3 & 4
# - All inherit Calibri styling already set.
# - All lazy-load by ticker (CSV assumed sorted by ticker/date).
# - All use the same pill-button range (st.session_state["range_sel"]).
# - All apply a 5-day gutter to the window.
# ==============================

# ---- CSV paths (same directory convention you already use) ----
FILE_G2 = DATA_DIR / "qry_graph_data_02.csv"   # Trend Lines
FILE_G3 = DATA_DIR / "qry_graph_data_03.csv"   # Price + Probable Anchors
FILE_G4 = DATA_DIR / "qry_graph_data_04.csv"   # Gap to LT Anchor (with bands)

# ---- tiny helper (reuses your apply_window_with_gutter) ----
def _window_by_label_with_gutter(df: pd.DataFrame, label: str, date_col: str) -> pd.DataFrame:
    return apply_window_with_gutter(df, label, date_col=date_col, gutter_days=5)

# ---- Lazy readers (ticker-only) ----
@st.cache_data(show_spinner=False)
def load_g2_ticker(path: Path, ticker: str) -> pd.DataFrame:
    """
    Graph 2: Trend Lines
      expected cols (case-insensitive):
        date, st_trend, mt_trend, lt_trend, [ticker]
      values may be fractions (0.12) or percentages (12); we normalize to percent.
    """
    if not Path(path).exists():
        return pd.DataFrame()
    out = []
    for chunk in pd.read_csv(path, chunksize=200000):
        cols = {c.strip().lower(): c for c in chunk.columns}
        tcol = cols.get("ticker")
        mask = (chunk[tcol] == ticker) if tcol else pd.Series(True, index=chunk.index)
        if mask.any():
            part = chunk.loc[mask].copy()
            part.columns = [c.strip().lower() for c in part.columns]
            out.append(part)
        elif out:
            break
    if not out:
        return pd.DataFrame()

    df = pd.concat(out, ignore_index=True)
    # normalize names
    rename = {
        "st_trend": "st",
        "mt_trend": "mt",
        "lt_trend": "lt",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    if "date" not in df.columns or not {"st","mt","lt"}.issubset(df.columns):
        return pd.DataFrame()

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for c in ["st","mt","lt"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # if values look like fractions (<=1), convert to percent
    mx = pd.concat([df["st"], df["mt"], df["lt"]], axis=0).abs().max()
    if pd.notna(mx) and mx <= 1.0:
        df[["st","mt","lt"]] = df[["st","mt","lt"]] * 100.0

    return df.sort_values("date").reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_g3_ticker(path: Path, ticker: str) -> pd.DataFrame:
    """
    Graph 3: Price + mid/long probable anchors
      expected cols (case-insensitive):
        date, close, mt_pb_anchor, lt_pb_anchor, [ticker]
    """
    if not Path(path).exists():
        return pd.DataFrame()
    out = []
    for chunk in pd.read_csv(path, chunksize=200000):
        cols = {c.strip().lower(): c for c in chunk.columns}
        tcol = cols.get("ticker")
        mask = (chunk[tcol] == ticker) if tcol else pd.Series(True, index=chunk.index)
        if mask.any():
            part = chunk.loc[mask].copy()
            part.columns = [c.strip().lower() for c in part.columns]
            out.append(part)
        elif out:
            break
    if not out:
        return pd.DataFrame()

    df = pd.concat(out, ignore_index=True)
    need = {"date","close","mt_pb_anchor","lt_pb_anchor"}
    if not need.issubset(set(df.columns)):
        return pd.DataFrame()

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for c in ["close","mt_pb_anchor","lt_pb_anchor"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df.sort_values("date").reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_g4_ticker(path: Path, ticker: str) -> pd.DataFrame:
    """
    Graph 4: Gap to LT probable anchor + bands
      expected cols (case-insensitive):
        date, gap_lt, gap_lt_avg, gap_lt_hi, gap_lt_lo, [ticker]
    """
    if not Path(path).exists():
        return pd.DataFrame()
    out = []
    for chunk in pd.read_csv(path, chunksize=200000):
        cols = {c.strip().lower(): c for c in chunk.columns}
        tcol = cols.get("ticker")
        mask = (chunk[tcol] == ticker) if tcol else pd.Series(True, index=chunk.index)
        if mask.any():
            part = chunk.loc[mask].copy()
            part.columns = [c.strip().lower() for c in part.columns]
            out.append(part)
        elif out:
            break
    if not out:
        return pd.DataFrame()

    df = pd.concat(out, ignore_index=True)
    need = {"date","gap_lt","gap_lt_avg","gap_lt_hi","gap_lt_lo"}
    if not need.issubset(set(df.columns)):
        return pd.DataFrame()

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for c in ["gap_lt","gap_lt_avg","gap_lt_hi","gap_lt_lo"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df.sort_values("date").reset_index(drop=True)


# ---- Plotters (Calibri already global) ----
def plot_g2_trend(df: pd.DataFrame, ticker: str):
    fig, ax = plt.subplots(figsize=(9.5, 3.9), dpi=150)

    ax.plot(df["date"], df["st"], label="ST Term",  linewidth=1.8, color=EXCEL_BLUE)
    ax.plot(df["date"], df["mt"], label="Mid Term", linewidth=1.8, color=EXCEL_ORANGE)
    ax.plot(df["date"], df["lt"], label="Long Term",linewidth=2.2, color="black")

    ax.set_title(f"{ticker} – Trend Lines", fontsize=12, pad=6)
    ax.set_ylabel("Percent")
    from matplotlib.ticker import PercentFormatter
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100))
    ax.grid(True, linewidth=0.4, alpha=0.4)

    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO, interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d/%y"))
    plt.setp(ax.get_xticklabels(), rotation=90, ha="center", fontsize=8)

    pad = pd.Timedelta(days=5)
    ax.set_xlim(df["date"].min() - pad, df["date"].max() + pad)

    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22),
              ncol=3, frameon=False, handlelength=2.8, fontsize=9)
    fig.subplots_adjust(bottom=0.30)
    plt.close(fig)  # 🔑 Prevents too many open figures
    return fig


def plot_g3_anchors(df: pd.DataFrame, ticker: str):
    fig, ax = plt.subplots(figsize=(9.5, 3.9), dpi=150)

    ax.plot(df["date"], df["close"],          label="Close",                     linewidth=1.8, color=EXCEL_BLUE)
    ax.plot(df["date"], df["mt_pb_anchor"],   label="Mid Term Probable Anchor",  linewidth=1.8, color=EXCEL_ORANGE)
    ax.plot(df["date"], df["lt_pb_anchor"],   label="Long Term Probable Anchor", linewidth=2.2, color="black")

    ax.set_title(f"{ticker} – Probable Anchors", fontsize=12, pad=6)
    ax.set_ylabel("Price")
    from matplotlib.ticker import StrMethodFormatter
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    ax.grid(True, linewidth=0.4, alpha=0.4)

    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO, interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d/%y"))
    plt.setp(ax.get_xticklabels(), rotation=90, ha="center", fontsize=8)

    pad = pd.Timedelta(days=5)
    ax.set_xlim(df["date"].min() - pad, df["date"].max() + pad)

    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22),
              ncol=3, frameon=False, handlelength=2.8, fontsize=9)
    fig.subplots_adjust(bottom=0.30)
    plt.close(fig)  # 🔑 Prevents too many open figures
    return fig


def plot_g4_gap(df: pd.DataFrame, ticker: str):
    fig, ax = plt.subplots(figsize=(9.5, 3.9), dpi=150)

    ax.plot(df["date"], df["gap_lt"], color=EXCEL_BLUE, linewidth=1.6, label="Gap to LT Anchor")

    # Flat reference lines from the first row (mirrors your Excel behavior)
    avg_val = df["gap_lt_avg"].iloc[0]
    hi_val  = df["gap_lt_hi"].iloc[0]
    lo_val  = df["gap_lt_lo"].iloc[0]
    ax.axhline(y=avg_val, color="black", linewidth=1.6, label="Avg")
    ax.axhline(y=hi_val,  color="red",   linewidth=1.2, label="High")
    ax.axhline(y=lo_val,  color="green", linewidth=1.2, label="Low")

    ax.set_title(f"{ticker} – Price to Long Term Probable Anchor", fontsize=12, pad=6)
    ax.set_ylabel("Gap")
    from matplotlib.ticker import StrMethodFormatter
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.2f}"))
    ax.grid(True, linewidth=0.4, alpha=0.4)

    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO, interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d/%y"))
    plt.setp(ax.get_xticklabels(), rotation=90, ha="center", fontsize=8)

    pad = pd.Timedelta(days=5)
    ax.set_xlim(df["date"].min() - pad, df["date"].max() + pad)

    # y-limits: include bands
    yvals = pd.concat([
        df["gap_lt"], df["gap_lt_avg"], df["gap_lt_hi"], df["gap_lt_lo"]
    ], axis=0)
    y_min, y_max = float(yvals.min()), float(yvals.max())
    if y_min == y_max:
        y_min -= 1.0; y_max += 1.0
    y_pad = 0.08 * (y_max - y_min)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)

    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22),
              ncol=4, frameon=False, handlelength=2.8, fontsize=9)
    fig.subplots_adjust(bottom=0.30)
    plt.close(fig)  # 🔑 Prevents too many open figures
    return fig


# ---- Render: three columns on one row ----
col2, col3, col4 = st.columns(3, gap="large")

with col2:
    _ticker = st.session_state.get("ticker_select", DEFAULT_TICKER)
    _rng    = st.session_state.get("range_sel", "All")
    df2_all = load_g2_ticker(FILE_G2, _ticker)
    if df2_all.empty:
        st.info("No trend data.")
    else:
        df2v = _window_by_label_with_gutter(df2_all, _rng, date_col="date")
        st.pyplot(plot_g2_trend(df2v, _ticker), clear_figure=True)

with col3:
    _ticker = st.session_state.get("ticker_select", DEFAULT_TICKER)
    _rng    = st.session_state.get("range_sel", "All")
    df3_all = load_g3_ticker(FILE_G3, _ticker)
    if df3_all.empty:
        st.info("No anchor data.")
    else:
        df3v = _window_by_label_with_gutter(df3_all, _rng, date_col="date")
        st.pyplot(plot_g3_anchors(df3v, _ticker), clear_figure=True)

with col4:
    _ticker = st.session_state.get("ticker_select", DEFAULT_TICKER)
    _rng    = st.session_state.get("range_sel", "All")
    df4_all = load_g4_ticker(FILE_G4, _ticker)
    if df4_all.empty:
        st.info("No gap data.")
    else:
        df4v = _window_by_label_with_gutter(df4_all, _rng, date_col="date")
        st.pyplot(plot_g4_gap(df4v, _ticker), clear_figure=True)

# ==============================
# ===== Graphs 2,3 & 4 (do not modify) END =====
# ==============================

# ==============================
# BOTTOM ROW #2 — Graphs 5, 6 & 7
# - Reuses: apply_window_with_gutter (5-day gutter), Calibri CSS, colors
# - Shares the pill-button range in st.session_state["range_sel"]
# ==============================

# ---- CSV paths ----
FILE_G5 = DATA_DIR / "qry_graph_data_05.csv"   # Z-Score + bands
FILE_G6 = DATA_DIR / "qry_graph_data_06.csv"   # Z-Score Percentile Rank
FILE_G7 = DATA_DIR / "qry_graph_data_07.csv"   # Rvol + bands

# ---- Lazy readers (ticker-only) ----
@st.cache_data(show_spinner=False)
def load_g5_ticker(path: Path, ticker: str) -> pd.DataFrame:
    """
    Graph 5: Z-Score 30d with bands
      expected (case-insensitive): date, [ticker], z-score, z-score_avg, z-score_hi, z-score_lo
      also supports: zscore, zscore_avg, zscore_hi, zscore_lo
    """
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    out = []
    for chunk in pd.read_csv(p, chunksize=200000):
        cols = {c.strip().lower(): c for c in chunk.columns}
        tcol = cols.get("ticker")
        mask = (chunk[tcol] == ticker) if tcol else pd.Series(True, index=chunk.index)
        if mask.any():
            part = chunk.loc[mask].copy()
            part.columns = [c.strip().lower() for c in part.columns]
            out.append(part)
        elif out:
            break
    if not out:
        return pd.DataFrame()
    df = pd.concat(out, ignore_index=True)

    # normalize names
    rename = {
        "z-score": "z",
        "zscore": "z",
        "z-score_avg": "avg",
        "zscore_avg": "avg",
        "z-score_hi": "hi",
        "zscore_hi": "hi",
        "z-score_lo": "lo",
        "zscore_lo": "lo",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    need = {"date", "z", "avg", "hi", "lo"}
    if not need.issubset(df.columns):
        return pd.DataFrame()

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for c in ["z", "avg", "hi", "lo"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.sort_values("date").reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_g6_ticker(path: Path, ticker: str) -> pd.DataFrame:
    """
    Graph 6: Z-Score Percentile Rank (0-100)
      expected: date, [ticker], z-score rank
      supports: z-score rank, zscore rank, zscore_rank, z_rank -> normalized to 'rank'
    """
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    out = []
    for chunk in pd.read_csv(p, chunksize=200000):
        cols = {c.strip().lower(): c for c in chunk.columns}
        tcol = cols.get("ticker")
        mask = (chunk[tcol] == ticker) if tcol else pd.Series(True, index=chunk.index)
        if mask.any():
            part = chunk.loc[mask].copy()
            part.columns = [c.strip().lower() for c in part.columns]
            out.append(part)
        elif out:
            break
    if not out:
        return pd.DataFrame()
    df = pd.concat(out, ignore_index=True)

    # normalize names
    for cand in ["z-score rank", "zscore rank", "zscore_rank", "z_rank", "rank"]:
        if cand in df.columns:
            df = df.rename(columns={cand: "rank"})
            break
    need = {"date", "rank"}
    if not need.issubset(df.columns):
        return pd.DataFrame()

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["rank"] = pd.to_numeric(df["rank"], errors="coerce")

    # If values look 0..1, convert to 0..100
    mx = df["rank"].abs().max()
    if pd.notna(mx) and mx <= 1.0:
        df["rank"] = df["rank"] * 100.0

    return df.sort_values("date").reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_g7_ticker(path: Path, ticker: str) -> pd.DataFrame:
    """
    Graph 7: Rvol 30d with bands
      expected: date, [ticker], rvol, rvol_avg, rvol_hi, rvol_low
    """
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    out = []
    for chunk in pd.read_csv(p, chunksize=200000):
        cols = {c.strip().lower(): c for c in chunk.columns}
        tcol = cols.get("ticker")
        mask = (chunk[tcol] == ticker) if tcol else pd.Series(True, index=chunk.index)
        if mask.any():
            part = chunk.loc[mask].copy()
            part.columns = [c.strip().lower() for c in part.columns]
            out.append(part)
        elif out:
            break
    if not out:
        return pd.DataFrame()
    df = pd.concat(out, ignore_index=True)

    need = {"date", "rvol", "rvol_avg", "rvol_hi", "rvol_low"}
    if not need.issubset(df.columns):
        return pd.DataFrame()

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for c in ["rvol", "rvol_avg", "rvol_hi", "rvol_low"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # If values look like fractions (<=1), convert to percent scale
    mx = pd.concat([df["rvol"], df["rvol_avg"], df["rvol_hi"], df["rvol_low"]], axis=0).abs().max()
    if pd.notna(mx) and mx <= 1.0:
        df[["rvol", "rvol_avg", "rvol_hi", "rvol_low"]] = df[["rvol", "rvol_avg", "rvol_hi", "rvol_low"]] * 100.0

    return df.sort_values("date").reset_index(drop=True)


# ---- Plotters ----
def plot_g5_zscore(df: pd.DataFrame, ticker: str):
    fig, ax = plt.subplots(figsize=(9.5, 3.9), dpi=150)

    ax.plot(df["date"], df["z"], color=EXCEL_BLUE, linewidth=1.6, label="Z-Score 30d")
    ax.axhline(y=df["avg"].iloc[0], color="black", linewidth=1.6, label="Avg")
    ax.axhline(y=df["hi"].iloc[0],  color="red",   linewidth=1.2, label="High")
    ax.axhline(y=df["lo"].iloc[0],  color="green", linewidth=1.2, label="Low")

    ax.set_title(f"{ticker} – Z-Score 30d", fontsize=12, pad=6)
    from matplotlib.ticker import StrMethodFormatter
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.2f}"))
    ax.grid(True, linewidth=0.4, alpha=0.4)

    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO, interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d/%y"))
    plt.setp(ax.get_xticklabels(), rotation=90, ha="center", fontsize=8)

    pad = pd.Timedelta(days=5)
    ax.set_xlim(df["date"].min() - pad, df["date"].max() + pad)

    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22),
              ncol=4, frameon=False, handlelength=2.8, fontsize=9)
    fig.subplots_adjust(bottom=0.30)
    plt.close(fig)  # 🔑 Prevents too many open figures
    return fig


def plot_g6_rank(df: pd.DataFrame, ticker: str):
    fig, ax = plt.subplots(figsize=(9.5, 3.9), dpi=150)

    ax.plot(df["date"], df["rank"], color=EXCEL_BLUE, linewidth=1.6)

    ax.set_title(f"{ticker} – Z-Score Percentile Rank", fontsize=12, pad=6)
    ax.set_ylim(0, 100)
    ax.grid(True, linewidth=0.4, alpha=0.4)

    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO, interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d/%y"))
    plt.setp(ax.get_xticklabels(), rotation=90, ha="center", fontsize=8)

    pad = pd.Timedelta(days=5)
    ax.set_xlim(df["date"].min() - pad, df["date"].max() + pad)

    fig.subplots_adjust(bottom=0.22)
    plt.close(fig)  # 🔑 Prevents too many open figures
    return fig


def plot_g7_rvol(df: pd.DataFrame, ticker: str):
    fig, ax = plt.subplots(figsize=(9.5, 3.9), dpi=150)

    ax.plot(df["date"], df["rvol"], color=EXCEL_BLUE, linewidth=1.6, label="Rvol 30d")
    ax.axhline(y=df["rvol_avg"].iloc[0], color="black", linewidth=1.6, label="Avg")
    ax.axhline(y=df["rvol_hi"].iloc[0],  color="red",   linewidth=1.2, label="High")
    ax.axhline(y=df["rvol_low"].iloc[0], color="green", linewidth=1.2, label="Low")

    ax.set_title(f"{ticker} – Rvol 30d", fontsize=12, pad=6)
    ax.set_ylabel("Percent")
    from matplotlib.ticker import PercentFormatter
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100))
    ax.grid(True, linewidth=0.4, alpha=0.4)

    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO, interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d/%y"))
    plt.setp(ax.get_xticklabels(), rotation=90, ha="center", fontsize=8)

    pad = pd.Timedelta(days=5)
    ax.set_xlim(df["date"].min() - pad, df["date"].max() + pad)

    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22),
              ncol=4, frameon=False, handlelength=2.8, fontsize=9)
    fig.subplots_adjust(bottom=0.30)
    plt.close(fig)  # 🔑 Prevents too many open figures
    return fig


# ---- Render: three columns on one row (5, 6, 7) ----
col5, col6, col7 = st.columns(3, gap="large")

with col5:
    _ticker = st.session_state.get("ticker_select", DEFAULT_TICKER)
    _rng    = st.session_state.get("range_sel", "All")
    df5_all = load_g5_ticker(FILE_G5, _ticker)
    if df5_all.empty:
        st.info("No Z-Score data.")
    else:
        df5v = apply_window_with_gutter(df5_all, _rng, date_col="date", gutter_days=5)  # reuse windowing :contentReference[oaicite:2]{index=2}
        st.pyplot(plot_g5_zscore(df5v, _ticker), clear_figure=True)

with col6:
    _ticker = st.session_state.get("ticker_select", DEFAULT_TICKER)
    _rng    = st.session_state.get("range_sel", "All")
    df6_all = load_g6_ticker(FILE_G6, _ticker)
    if df6_all.empty:
        st.info("No percentile rank data.")
    else:
        df6v = apply_window_with_gutter(df6_all, _rng, date_col="date", gutter_days=5)
        st.pyplot(plot_g6_rank(df6v, _ticker), clear_figure=True)

with col7:
    _ticker = st.session_state.get("ticker_select", DEFAULT_TICKER)
    _rng    = st.session_state.get("range_sel", "All")
    df7_all = load_g7_ticker(FILE_G7, _ticker)
    if df7_all.empty:
        st.info("No rVol data.")
    else:
        df7v = apply_window_with_gutter(df7_all, _rng, date_col="date", gutter_days=5)
        st.pyplot(plot_g7_rvol(df7v, _ticker), clear_figure=True)
# ==============================
# ===== Graphs 5,6 & 7 (do not modify) END =====
# ==============================
# ==============================
# BOTTOM ROW #3 — Graphs 8, 9 & 10
# - Reuses: apply_window_with_gutter (5-day gutter), Calibri CSS, Excel palette
# - Shares the pill-button range in st.session_state["range_sel"]
# ==============================

# ---- CSV paths ----
FILE_G8  = DATA_DIR / "qry_graph_data_08.csv"   # Sharpe Ratio 30d + bands
FILE_G9  = DATA_DIR / "qry_graph_data_09.csv"   # Sharpe Ratio Percentile Rank
FILE_G10 = DATA_DIR / "qry_graph_data_10.csv"   # Ivol Prem/Disc 30d + bands

# ---- Lazy readers (ticker-only) ----
@st.cache_data(show_spinner=False)
def load_g8_ticker(path: Path, ticker: str) -> pd.DataFrame:
    """
    Graph 8: Sharpe Ratio 30d with bands
      Accepts (case-insensitive):
        date, [ticker], sharpe OR sharpe_ratio,
        sharpe_avg, sharpe_hi, sharpe_lo/sharpe_low
    """
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()

    out = []
    for chunk in pd.read_csv(p, chunksize=200000):
        cols = {c.strip().lower(): c for c in chunk.columns}
        tcol = cols.get("ticker")
        m = (chunk[tcol] == ticker) if tcol else pd.Series(True, index=chunk.index)
        if m.any():
            part = chunk.loc[m].copy()
            part.columns = [c.strip().lower() for c in part.columns]
            out.append(part)
        elif out:
            break
    if not out:
        return pd.DataFrame()
    df = pd.concat(out, ignore_index=True)

    # normalize names
    rename = {
        "sharpe_ratio": "sharpe",
        "sharpe": "sharpe",
        "sharpe_avg": "avg",
        "sharpe_hi": "hi",
        "sharpe_lo": "lo",
        "sharpe_low": "lo",   # alias from your CSV
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    need = {"date", "sharpe", "avg", "hi", "lo"}
    if not need.issubset(df.columns):
        return pd.DataFrame()

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for c in ["sharpe", "avg", "hi", "lo"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df.sort_values("date").reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_g9_ticker(path: Path, ticker: str) -> pd.DataFrame:
    """
    Graph 9: Sharpe Ratio Percentile Rank (0–100)
      Accepts (case-insensitive):
        date, [ticker], sharpe_rank / sharpe percentile / percentile / rank
      If values are 0..1, auto-scale to 0..100.
    """
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()

    out = []
    for chunk in pd.read_csv(p, chunksize=200000):
        cols = {c.strip().lower(): c for c in chunk.columns}
        tcol = cols.get("ticker")
        m = (chunk[tcol] == ticker) if tcol else pd.Series(True, index=chunk.index)
        if m.any():
            part = chunk.loc[m].copy()
            part.columns = [c.strip().lower() for c in part.columns]
            out.append(part)
        elif out:
            break
    if not out:
        return pd.DataFrame()
    df = pd.concat(out, ignore_index=True)

    for cand in ["sharpe_rank", "sharpe percentile", "percentile", "rank"]:
        if cand in df.columns:
            df = df.rename(columns={cand: "rank"})
            break
    need = {"date", "rank"}
    if not need.issubset(df.columns):
        return pd.DataFrame()

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["rank"] = pd.to_numeric(df["rank"], errors="coerce")

    mx = df["rank"].abs().max()
    if pd.notna(mx) and mx <= 1.0:
        df["rank"] = df["rank"] * 100.0

    return df.sort_values("date").reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_g10_ticker(path: Path, ticker: str) -> pd.DataFrame:
    """
    Graph 10: Ivol Prem/Disc 30d with bands (percent)
      Accepts (case-insensitive):
        date, [ticker],
        prem_disc, prem_disc_avg, prem_disc_hi, prem_disc_lo
        OR legacy ivol names: ivol_pd, ivol_avg, ivol_hi, ivol_lo/ivol_low
      If values are 0..1, auto-scale to percent (×100).
    """
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()

    out = []
    for chunk in pd.read_csv(p, chunksize=200000):
        cols = {c.strip().lower(): c for c in chunk.columns}
        tcol = cols.get("ticker")
        m = (chunk[tcol] == ticker) if tcol else pd.Series(True, index=chunk.index)
        if m.any():
            part = chunk.loc[m].copy()
            part.columns = [c.strip().lower() for c in part.columns]
            out.append(part)
        elif out:
            break
    if not out:
        return pd.DataFrame()
    df = pd.concat(out, ignore_index=True)

    # normalize to a single schema: ivol_pd, avg, hi, lo
    rename = {
        # legacy ivol names
        "ivol_p/d": "ivol_pd",
        "ivol_pd": "ivol_pd",
        "ivol_avg": "avg",
        "ivol_hi": "hi",
        "ivol_lo": "lo",
        "ivol_low": "lo",
        # prem/discount names (your CSV)
        "prem_disc": "ivol_pd",
        "prem_disc_avg": "avg",
        "prem_disc_hi": "hi",
        "prem_disc_lo": "lo",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    need = {"date", "ivol_pd", "avg", "hi", "lo"}
    if not need.issubset(df.columns):
        return pd.DataFrame()

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for c in ["ivol_pd", "avg", "hi", "lo"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # If looks like fractions (<=1), convert to percent scale
    mx = pd.concat([df["ivol_pd"], df["avg"], df["hi"], df["lo"]], axis=0).abs().max()
    if pd.notna(mx) and mx <= 1.0:
        df[["ivol_pd", "avg", "hi", "lo"]] = df[["ivol_pd", "avg", "hi", "lo"]] * 100.0

    return df.sort_values("date").reset_index(drop=True)


# ---- Plotters ----
def plot_g8_sharpe(df: pd.DataFrame, ticker: str):
    fig, ax = plt.subplots(figsize=(9.5, 3.9), dpi=150)
    ax.plot(df["date"], df["sharpe"], color=EXCEL_BLUE, linewidth=1.6, label="Sharpe Ratio 30d")
    ax.axhline(y=df["avg"].iloc[0], color="black", linewidth=1.6, label="Avg")
    ax.axhline(y=df["hi"].iloc[0],  color="red",   linewidth=1.2, label="High")
    ax.axhline(y=df["lo"].iloc[0],  color="green", linewidth=1.2, label="Low")

    ax.set_title(f"{ticker} – Sharpe Ratio 30d", fontsize=12, pad=6)
    from matplotlib.ticker import StrMethodFormatter
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.2f}"))
    ax.grid(True, linewidth=0.4, alpha=0.4)

    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO, interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d/%y"))
    plt.setp(ax.get_xticklabels(), rotation=90, ha="center", fontsize=8)

    pad = pd.Timedelta(days=5)
    ax.set_xlim(df["date"].min() - pad, df["date"].max() + pad)

    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22),
              ncol=4, frameon=False, handlelength=2.8, fontsize=9)
    fig.subplots_adjust(bottom=0.30)
    plt.close(fig)  # 🔑 Prevents too many open figures
    return fig


def plot_g9_sharpe_rank(df: pd.DataFrame, ticker: str):
    fig, ax = plt.subplots(figsize=(9.5, 3.9), dpi=150)
    ax.plot(df["date"], df["rank"], color=EXCEL_BLUE, linewidth=1.6)

    ax.set_title(f"{ticker} – Sharpe Ratio Percentile Rank", fontsize=12, pad=6)
    ax.set_ylim(0, 100)
    ax.grid(True, linewidth=0.4, alpha=0.4)

    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO, interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d/%y"))
    plt.setp(ax.get_xticklabels(), rotation=90, ha="center", fontsize=8)

    pad = pd.Timedelta(days=5)
    ax.set_xlim(df["date"].min() - pad, df["date"].max() + pad)

    fig.subplots_adjust(bottom=0.22)
    plt.close(fig)  # 🔑 Prevents too many open figures
    return fig


def plot_g10_ivol_pd(df: pd.DataFrame, ticker: str):
    fig, ax = plt.subplots(figsize=(9.5, 3.9), dpi=150)
    ax.plot(df["date"], df["ivol_pd"], color=EXCEL_BLUE, linewidth=1.6, label="Prem/Disc")
    ax.axhline(y=df["avg"].iloc[0], color="black", linewidth=1.6, label="Avg")
    ax.axhline(y=df["hi"].iloc[0],  color="red",   linewidth=1.2, label="High")
    ax.axhline(y=df["lo"].iloc[0],  color="green", linewidth=1.2, label="Low")

    ax.set_title(f"{ticker} – Ivol Prem/Disc", fontsize=12, pad=6)
    ax.set_ylabel("Percent")
    from matplotlib.ticker import PercentFormatter
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100))
    ax.grid(True, linewidth=0.4, alpha=0.4)

    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO, interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d/%y"))
    plt.setp(ax.get_xticklabels(), rotation=90, ha="center", fontsize=8)

    pad = pd.Timedelta(days=5)
    ax.set_xlim(df["date"].min() - pad, df["date"].max() + pad)

    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22),
              ncol=4, frameon=False, handlelength=2.8, fontsize=9)
    fig.subplots_adjust(bottom=0.30)
    plt.close(fig)  # 🔑 Prevents too many open figures
    return fig


# ---- Render: three columns on one row (8, 9, 10) ----
col8, col9, col10 = st.columns(3, gap="large")

with col8:
    _ticker = st.session_state.get("ticker_select", DEFAULT_TICKER)
    _rng    = st.session_state.get("range_sel", "All")
    df8_all = load_g8_ticker(FILE_G8, _ticker)
    if df8_all.empty:
        st.info("No Sharpe data.")
    else:
        df8v = apply_window_with_gutter(df8_all, _rng, date_col="date", gutter_days=5)
        st.pyplot(plot_g8_sharpe(df8v, _ticker), clear_figure=True)

with col9:
    _ticker = st.session_state.get("ticker_select", DEFAULT_TICKER)
    _rng    = st.session_state.get("range_sel", "All")
    df9_all = load_g9_ticker(FILE_G9, _ticker)
    if df9_all.empty:
        st.info("No Sharpe rank data.")
    else:
        df9v = apply_window_with_gutter(df9_all, _rng, date_col="date", gutter_days=5)
        st.pyplot(plot_g9_sharpe_rank(df9v, _ticker), clear_figure=True)

with col10:
    _ticker = st.session_state.get("ticker_select", DEFAULT_TICKER)
    _rng    = st.session_state.get("range_sel", "All")
    df10_all = load_g10_ticker(FILE_G10, _ticker)
    if df10_all.empty:
        st.info("No Prem/Disc data.")
    else:
        df10v = apply_window_with_gutter(df10_all, _rng, date_col="date", gutter_days=5)
        st.pyplot(plot_g10_ivol_pd(df10v, _ticker), clear_figure=True)

# ==============================
# ===== Graphs 8, 9 & 10 (do not modify) END =====
# ==============================

# ==============================
# ROW — Notes (left), Graph 11 (middle), Graph 12 (right)
# - Reuses: apply_window_with_gutter (5-day gutter), Calibri CSS, Excel palette
# - Shares the pill-button range in st.session_state["range_sel"]
# ==============================

# ---- CSV paths ----
FILE_G11 = DATA_DIR / "qry_graph_data_11.csv"   # Signal Score + Close
FILE_G12 = DATA_DIR / "qry_graph_data_12.csv"   # Scatter: Z-Score vs Ivol Prem/Disc (two dates per ticker)

# ---- Lazy readers (ticker-only) ----
@st.cache_data(show_spinner=False)
def load_g11_ticker(path: Path, ticker: str) -> pd.DataFrame:
    """
    Graph 11: Signal Score (left axis) with Close (right axis)
      expected (case-insensitive): date, [ticker], ticker_name, exposure, category, close, model_score
    """
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()

    out = []
    for chunk in pd.read_csv(p, chunksize=200000):
        cols = {c.strip().lower(): c for c in chunk.columns}
        tcol = cols.get("ticker")
        m = (chunk[tcol] == ticker) if tcol else pd.Series(True, index=chunk.index)
        if m.any():
            part = chunk.loc[m].copy()
            part.columns = [c.strip().lower() for c in part.columns]
            out.append(part)
        elif out:
            break
    if not out:
        return pd.DataFrame()
    df = pd.concat(out, ignore_index=True)

    df = df.rename(columns={"model_score": "score"})
    need = {"date", "close", "score"}
    if not need.issubset(df.columns):
        return pd.DataFrame()

    df["date"]  = pd.to_datetime(df["date"], errors="coerce")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df["score"] = pd.to_numeric(df["score"], errors="coerce")

    return df.sort_values("date").reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_g12_ticker(path: Path, ticker: str) -> pd.DataFrame:
    """
    Graph 12: Scatter — Z-Score (x) vs Ivol Prem/Disc (y, %)
      expected (case-insensitive): date, [ticker], zscore, prem_disc
      If prem_disc is 0..1, auto-scale to percent (×100).
      File will typically have TWO rows per ticker: latest and ~30d prior.
    """
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()

    out = []
    for chunk in pd.read_csv(p, chunksize=200000):
        cols = {c.strip().lower(): c for c in chunk.columns}
        tcol = cols.get("ticker")
        m = (chunk[tcol] == ticker) if tcol else pd.Series(True, index=chunk.index)
        if m.any():
            part = chunk.loc[m].copy()
            part.columns = [c.strip().lower() for c in part.columns]
            out.append(part)
        elif out:
            break
    if not out:
        return pd.DataFrame()
    df = pd.concat(out, ignore_index=True)

    df = df.rename(columns={"zscore": "z", "prem_disc": "pd"})
    need = {"date", "z", "pd"}
    if not need.issubset(df.columns):
        return pd.DataFrame()

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["z"]    = pd.to_numeric(df["z"], errors="coerce")
    df["pd"]   = pd.to_numeric(df["pd"], errors="coerce")

    # Convert prem/discount to % if it looks like fraction
    #mx = df["pd"].abs().max()
    #if pd.notna(mx) and mx <= 1.0:
    df["pd"] = df["pd"] * 100.0

    return df.sort_values("date").reset_index(drop=True)

# ---- Plotters ----
def plot_g11_signal(df: pd.DataFrame, ticker: str):
    fig, ax = plt.subplots(figsize=(9.5, 3.9), dpi=150)

    # Left axis: Signal Score
    ax.plot(df["date"], df["score"], color=EXCEL_BLUE, linewidth=1.6, label="Signal Score")

    # Right axis: Close
    ax2 = ax.twinx()
    ax2.plot(df["date"], df["close"], color="black", linewidth=1.4, label="Close")

    ax.set_title(f"{ticker} – Signal Score", fontsize=12, pad=6)
    ax.grid(True, linewidth=0.4, alpha=0.4)

    # X axis (biweekly Mondays) with 5-day gutter
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO, interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d/%y"))
    plt.setp(ax.get_xticklabels(), rotation=90, ha="center", fontsize=8)

    pad = pd.Timedelta(days=5)
    ax.set_xlim(df["date"].min() - pad, df["date"].max() + pad)

    # Combined legend below
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], color=EXCEL_BLUE, linewidth=1.6, label="Signal Score"),
        Line2D([0], [0], color="black",    linewidth=1.4, label="Close"),
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.22),
              ncol=2, frameon=False, handlelength=2.8, fontsize=9)
    fig.subplots_adjust(bottom=0.30)
    plt.close(fig)  # 🔑 Prevents too many open figures
    return fig


def plot_g12_scatter(df: pd.DataFrame, ticker: str):
    """
    Dynamic, zero-centered bounds so both points always fit.
    - Both dots Excel blue
    - Dates under each dot
    - No legend
    - Zero lines through center
    """
    fig, ax = plt.subplots(figsize=(9.5, 3.9), dpi=150)

    # Ensure order: older first, latest last
    df = df.sort_values("date").reset_index(drop=True)
    older, latest = df.iloc[0], df.iloc[-1]

    # Plot both points
    ax.scatter([older["z"]],  [older["pd"]],  s=70, color=EXCEL_BLUE, zorder=4)
    ax.scatter([latest["z"]], [latest["pd"]], s=90, color=EXCEL_BLUE, zorder=5)

    ax.set_title(f"{ticker} – Ivol/Rvol % Spreads", fontsize=12, pad=6)
    ax.set_xlabel("Z-Score (30 day)")
    ax.set_ylabel("Ivol Prem/Disc (30 day)")

    # ----- Dynamic, zero-centered limits -----
    import math
    # X: at least ±5, otherwise expand to cover data with 10% pad and round up to 0.5 steps
    x_abs = max(5.0, abs(float(df["z"].min())), abs(float(df["z"].max())))
    x_abs = x_abs * 1.10
    x_abs = math.ceil(x_abs * 2) / 2.0  # round up to nearest 0.5
    ax.set_xlim(-x_abs, x_abs)

    # Y: at least ±100%, expand if needed with 10% pad and round up to 10%
    y_abs = max(100.0, abs(float(df["pd"].min())), abs(float(df["pd"].max())))
    y_abs = y_abs * 1.10
    y_abs = math.ceil(y_abs / 10.0) * 10.0
    ax.set_ylim(-y_abs, y_abs)

    # Zero lines through center
    ax.axhline(0.0, color="black", linewidth=1.0, zorder=1)
    ax.axvline(0.0, color="black", linewidth=1.0, zorder=1)

    # Percent y-axis
    from matplotlib.ticker import PercentFormatter
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100))

    ax.grid(True, linewidth=0.4, alpha=0.4)

    # Dates under each dot
    def under_label(row):
        try:
            dt = pd.to_datetime(row["date"], errors="coerce")
            txt = dt.strftime("%m/%d/%Y") if pd.notna(dt) else str(row["date"])
            ax.annotate(
                txt, (row["z"], row["pd"]),
                xytext=(0, -12), textcoords="offset points",
                ha="center", va="top",
                fontsize=8,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.9),
                zorder=6,
            )
        except Exception:
            pass

    under_label(older)
    under_label(latest)

    # Corner labels (slightly inset)
    def corner_label(text, xy_axes, color):
        ax.text(
            xy_axes[0], xy_axes[1], text,
            transform=ax.transAxes, ha="center", va="center", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=color, lw=1.2),
            zorder=3,
        )
    corner_label("Mean Reversion", (0.10, 0.90), "green")
    corner_label("Crowded Short",  (0.90, 0.90), "green")
    corner_label("Crowded Long",   (0.10, 0.10), "red")
    corner_label("Mean Reversion", (0.90, 0.10), "red")

    fig.subplots_adjust(bottom=0.18)
    plt.close(fig)  # 🔑 Prevents too many open figures
    return fig

# ---- Render: Notes | Graph 11 | Graph 12 ----
ncol, g11col, g12col = st.columns([1, 1, 1], gap="large")

with ncol:
    st.markdown(
        """
        <div class="calibri-text">
        <b>Note:</b><br>
        • High Line is Average plus 1 standard deviation<br>
        • Low Line is Average less 1 standard deviation<br>
        • Z-Score Rank is trailing 1 year percentile rank<br>
        • Sharpe Ratio Rank is trailing 1 year percentile rank
        </div>
        """,
        unsafe_allow_html=True
    )

with g11col:
    _ticker = st.session_state.get("ticker_select", DEFAULT_TICKER)
    _rng    = st.session_state.get("range_sel", "All")
    df11_all = load_g11_ticker(FILE_G11, _ticker)
    if df11_all.empty:
        st.info("No Signal Score data.")
    else:
        df11v = apply_window_with_gutter(df11_all, _rng, date_col="date", gutter_days=5)
        st.pyplot(plot_g11_signal(df11v, _ticker), clear_figure=True)

with g12col:
    _ticker = st.session_state.get("ticker_select", DEFAULT_TICKER)
    _rng    = st.session_state.get("range_sel", "All")
    df12_all = load_g12_ticker(FILE_G12, _ticker)
    if df12_all.empty:
        st.info("No scatter data.")
    else:
        df12v = apply_window_with_gutter(df12_all, _rng, date_col="date", gutter_days=5)
        st.pyplot(plot_g12_scatter(df12v, _ticker), clear_figure=True)

# ==============================
# ===== Notes + Graphs 11 & 12 (do not modify) END =====
# ==============================

# ==============================
# MASTER TOGGLE: Show/Hide Informational Charts (13–24)
# ==============================
if "show_informational_13_24" not in st.session_state:
    st.session_state.show_informational_13_24 = False

tL, tM, tR = st.columns([1.2, 3, 0.8])
with tM:
    st.toggle(
        "Show informational charts (13–24)",
        key="show_informational_13_24",     # widget owns state
        help="Turn on to render rows 13–24.",
    )

render_info = st.session_state.show_informational_13_24

# ----- NOTHING BELOW SHOULD RENDER UNLESS render_info IS TRUE -----
if render_info:

    # ==============================
    # ===== Graphs 13, 14 & 15 START =====
    # ==============================

    # ---- CSV paths ----
    FILE_G13 = DATA_DIR / "qry_graph_data_13.csv"  # Daily Returns % + bands
    FILE_G14 = DATA_DIR / "qry_graph_data_14.csv"  # Daily Range + bands
    FILE_G15 = DATA_DIR / "qry_graph_data_15.csv"  # Daily Volume + bands

    # ---- Loaders (ticker-only) ----
    @st.cache_data(show_spinner=False)
    def load_g13_ticker(path: Path, ticker: str) -> pd.DataFrame:
        """
        Graph 13: Daily Returns (%) with Avg/High/Low bands
          expected (case-insensitive):
            date, [ticker],
            daily_return_pct, daily_return_avg_pct, daily_return_hi_pct, daily_return_lo_pct
        """
        p = Path(path)
        if not p.exists():
            return pd.DataFrame()
        out = []
        for chunk in pd.read_csv(p, chunksize=200000):
            cols = {c.strip().lower(): c for c in chunk.columns}
            tcol = cols.get("ticker")
            m = (chunk[tcol] == ticker) if tcol else pd.Series(True, index=chunk.index)
            if m.any():
                part = chunk.loc[m].copy()
                part.columns = [c.strip().lower() for c in part.columns]
                out.append(part)
            elif out:
                break
        if not out:
            return pd.DataFrame()
        df = pd.concat(out, ignore_index=True)

        need = {
            "date", "daily_return_pct",
            "daily_return_avg_pct", "daily_return_hi_pct", "daily_return_lo_pct"
        }
        if not need.issubset(df.columns):
            return pd.DataFrame()

        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        for c in ["daily_return_pct", "daily_return_avg_pct", "daily_return_hi_pct", "daily_return_lo_pct"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        # if values look like 0..1, convert to % scale
        mx = pd.concat([
            df["daily_return_pct"],
            df["daily_return_avg_pct"],
            df["daily_return_hi_pct"],
            df["daily_return_lo_pct"]
        ]).abs().max()
        if pd.notna(mx) and mx <= 1.0:
            df[["daily_return_pct", "daily_return_avg_pct",
                "daily_return_hi_pct", "daily_return_lo_pct"]] *= 100.0

        return df.sort_values("date").reset_index(drop=True)


    @st.cache_data(show_spinner=False)
    def load_g14_ticker(path: Path, ticker: str) -> pd.DataFrame:
        """
        Graph 14: Daily Range with Avg/High/Low bands
          expected (case-insensitive):
            date, [ticker], daily_range, daily_range_avg, daily_range_hi, daily_range_lo
        """
        p = Path(path)
        if not p.exists():
            return pd.DataFrame()
        out = []
        for chunk in pd.read_csv(p, chunksize=200000):
            cols = {c.strip().lower(): c for c in chunk.columns}
            tcol = cols.get("ticker")
            m = (chunk[tcol] == ticker) if tcol else pd.Series(True, index=chunk.index)
            if m.any():
                part = chunk.loc[m].copy()
                part.columns = [c.strip().lower() for c in part.columns]
                out.append(part)
            elif out:
                break
        if not out:
            return pd.DataFrame()
        df = pd.concat(out, ignore_index=True)

        need = {"date", "daily_range", "daily_range_avg", "daily_range_hi", "daily_range_lo"}
        if not need.issubset(df.columns):
            return pd.DataFrame()

        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        for c in ["daily_range", "daily_range_avg", "daily_range_hi", "daily_range_lo"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        return df.sort_values("date").reset_index(drop=True)


    @st.cache_data(show_spinner=False)
    def load_g15_ticker(path: Path, ticker: str) -> pd.DataFrame:
        """
        Graph 15: Daily Volume with Avg/High/Low bands
          expected (case-insensitive):
            date, [ticker], daily_volume, daily_volume_avg, daily_volume_hi, daily_volume_lo
        """
        p = Path(path)
        if not p.exists():
            return pd.DataFrame()
        out = []
        for chunk in pd.read_csv(p, chunksize=200000):
            cols = {c.strip().lower(): c for c in chunk.columns}
            tcol = cols.get("ticker")
            m = (chunk[tcol] == ticker) if tcol else pd.Series(True, index=chunk.index)
            if m.any():
                part = chunk.loc[m].copy()
                part.columns = [c.strip().lower() for c in part.columns]
                out.append(part)
            elif out:
                break
        if not out:
            return pd.DataFrame()
        df = pd.concat(out, ignore_index=True)

        need = {"date", "daily_volume", "daily_volume_avg", "daily_volume_hi", "daily_volume_lo"}
        if not need.issubset(df.columns):
            return pd.DataFrame()

        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        for c in ["daily_volume", "daily_volume_avg", "daily_volume_hi", "daily_volume_lo"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        return df.sort_values("date").reset_index(drop=True)

    # ---- Plotters ----
    def plot_g13_daily_returns(df: pd.DataFrame, ticker: str):
        fig, ax = plt.subplots(figsize=(9.5, 3.9), dpi=150)

        # bar colors by sign (green positive, red negative)
        colors = ["green" if v >= 0 else "red" for v in df["daily_return_pct"]]
        ax.bar(df["date"], df["daily_return_pct"], width=1.0, color=colors, linewidth=0)

        # bands
        ax.axhline(y=df["daily_return_avg_pct"].iloc[0], color="black", linewidth=1.6, label="Avg")
        ax.axhline(y=df["daily_return_hi_pct"].iloc[0],  color="red",   linewidth=1.2, label="High")
        ax.axhline(y=df["daily_return_lo_pct"].iloc[0],  color="green", linewidth=1.2, label="Low")

        ax.set_title(f"{ticker} – Daily Returns", fontsize=12, pad=6)
        ax.grid(True, linewidth=0.4, alpha=0.4)

        # percent formatter on y
        from matplotlib.ticker import PercentFormatter
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=100))

        # biweekly Monday ticks + 5-day gutter
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO, interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d/%y"))
        plt.setp(ax.get_xticklabels(), rotation=90, ha="center", fontsize=8)

        pad = pd.Timedelta(days=5)
        ax.set_xlim(df["date"].min() - pad, df["date"].max() + pad)

        # small legend centered below
        from matplotlib.lines import Line2D
        handles = [
            Line2D([0], [0], color="black", linewidth=1.6, label="Avg"),
            Line2D([0], [0], color="red",   linewidth=1.2, label="High"),
            Line2D([0], [0], color="green", linewidth=1.2, label="Low"),
        ]
        ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.22),
                  ncol=3, frameon=False, handlelength=2.8, fontsize=9)
        fig.subplots_adjust(bottom=0.30)
        plt.close(fig)  # 🔑 Prevents too many open figures
        return fig


    def plot_g14_daily_range(df: pd.DataFrame, ticker: str):
        fig, ax = plt.subplots(figsize=(9.5, 3.9), dpi=150)

        ax.plot(df["date"], df["daily_range"], color=EXCEL_BLUE, linewidth=1.6, label="Range")
        ax.axhline(y=df["daily_range_avg"].iloc[0], color="black", linewidth=1.6, label="Avg")
        ax.axhline(y=df["daily_range_hi"].iloc[0],  color="red",   linewidth=1.2, label="High")
        ax.axhline(y=df["daily_range_lo"].iloc[0],  color="green", linewidth=1.2, label="Low")

        ax.set_title(f"{ticker} – Daily Range", fontsize=12, pad=6)
        ax.grid(True, linewidth=0.4, alpha=0.4)

        ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO, interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d/%y"))
        plt.setp(ax.get_xticklabels(), rotation=90, ha="center", fontsize=8)

        pad = pd.Timedelta(days=5)
        ax.set_xlim(df["date"].min() - pad, df["date"].max() + pad)

        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22),
                  ncol=4, frameon=False, handlelength=2.8, fontsize=9)
        fig.subplots_adjust(bottom=0.30)
        plt.close(fig)  # 🔑 Prevents too many open figures
        return fig


    def plot_g15_daily_volume(df: pd.DataFrame, ticker: str):
        fig, ax = plt.subplots(figsize=(9.5, 3.9), dpi=150)

        ax.plot(df["date"], df["daily_volume"], color=EXCEL_BLUE, linewidth=1.6, label="Volume")
        ax.axhline(y=df["daily_volume_avg"].iloc[0], color="black", linewidth=1.6, label="Avg")
        ax.axhline(y=df["daily_volume_hi"].iloc[0],  color="red",   linewidth=1.2, label="High")
        ax.axhline(y=df["daily_volume_lo"].iloc[0],  color="green", linewidth=1.2, label="Low")

        ax.set_title(f"{ticker} – Daily Volume", fontsize=12, pad=6)
        ax.grid(True, linewidth=0.4, alpha=0.4)

        ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO, interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d/%y"))
        plt.setp(ax.get_xticklabels(), rotation=90, ha="center", fontsize=8)

        pad = pd.Timedelta(days=5)
        ax.set_xlim(df["date"].min() - pad, df["date"].max() + pad)

        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22),
                  ncol=4, frameon=False, handlelength=2.8, fontsize=9)
        fig.subplots_adjust(bottom=0.30)
        plt.close(fig)  # 🔑 Prevents too many open figures
        return fig

    # ---- Render row horizontally (13, 14, 15) ----
    col13, col14, col15 = st.columns(3, gap="large")

    with col13:
        _ticker = st.session_state.get("ticker_select", DEFAULT_TICKER)
        _rng    = st.session_state.get("range_sel", "All")
        df13_all = load_g13_ticker(FILE_G13, _ticker)
        if df13_all.empty:
            st.info("No Daily Returns data.")
        else:
            df13v = apply_window_with_gutter(df13_all, _rng, date_col="date", gutter_days=5)
            st.pyplot(plot_g13_daily_returns(df13v, _ticker), clear_figure=True)

    with col14:
        _ticker = st.session_state.get("ticker_select", DEFAULT_TICKER)
        _rng    = st.session_state.get("range_sel", "All")
        df14_all = load_g14_ticker(FILE_G14, _ticker)
        if df14_all.empty:
            st.info("No Daily Range data.")
        else:
            df14v = apply_window_with_gutter(df14_all, _rng, date_col="date", gutter_days=5)
            st.pyplot(plot_g14_daily_range(df14v, _ticker), clear_figure=True)

    with col15:
        _ticker = st.session_state.get("ticker_select", DEFAULT_TICKER)
        _rng    = st.session_state.get("range_sel", "All")
        df15_all = load_g15_ticker(FILE_G15, _ticker)
        if df15_all.empty:
            st.info("No Daily Volume data.")
        else:
            df15v = apply_window_with_gutter(df15_all, _rng, date_col="date", gutter_days=5)
            st.pyplot(plot_g15_daily_volume(df15v, _ticker), clear_figure=True)

    # ==============================
    # ===== Graphs 13, 14 & 15 (do not modify) END =====
    # ==============================

    # ==============================
    # ===== Graphs 16, 17 & 18 START =====
    # ==============================

    # ---- CSV paths ----
    FILE_G16 = DATA_DIR / "qry_graph_data_16.csv"  # Weekly Returns % + bands
    FILE_G17 = DATA_DIR / "qry_graph_data_17.csv"  # Weekly Range + bands
    FILE_G18 = DATA_DIR / "qry_graph_data_18.csv"  # Weekly Volume + bands

    # ---- Loaders (ticker-only) ----
    @st.cache_data(show_spinner=False)
    def load_g16_ticker(path: Path, ticker: str) -> pd.DataFrame:
        """
        Graph 16: Weekly Returns (%) with Avg/High/Low bands
        expected (case-insensitive):
            date, [ticker],
            weekly_return_pct, weekly_return_avg_pct, weekly_return_hi_pct, weekly_return_lo_pct
        """
        p = Path(path)
        if not p.exists():
            return pd.DataFrame()
        out = []
        for chunk in pd.read_csv(p, chunksize=200000):
            cols = {c.strip().lower(): c for c in chunk.columns}
            tcol = cols.get("ticker")
            m = (chunk[tcol] == ticker) if tcol else pd.Series(True, index=chunk.index)
            if m.any():
                part = chunk.loc[m].copy()
                part.columns = [c.strip().lower() for c in part.columns]
                out.append(part)
            elif out:
                break
        if not out:
            return pd.DataFrame()
        df = pd.concat(out, ignore_index=True)

        need = {
            "date", "weekly_return_pct",
            "weekly_return_avg_pct", "weekly_return_hi_pct", "weekly_return_lo_pct"
        }
        if not need.issubset(df.columns):
            return pd.DataFrame()

        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        for c in ["weekly_return_pct", "weekly_return_avg_pct", "weekly_return_hi_pct", "weekly_return_lo_pct"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        # if values look like 0..1, convert to % scale
        mx = pd.concat([
            df["weekly_return_pct"],
            df["weekly_return_avg_pct"],
            df["weekly_return_hi_pct"],
            df["weekly_return_lo_pct"]
        ]).abs().max()
        if pd.notna(mx) and mx <= 1.0:
            df[["weekly_return_pct", "weekly_return_avg_pct",
                "weekly_return_hi_pct", "weekly_return_lo_pct"]] *= 100.0

        return df.sort_values("date").reset_index(drop=True)


    @st.cache_data(show_spinner=False)
    def load_g17_ticker(path: Path, ticker: str) -> pd.DataFrame:
        """
        Graph 17: Weekly Range with Avg/High/Low bands
        expected (case-insensitive):
            date, [ticker], weekly_range, weekly_range_avg, weekly_range_hi, weekly_range_lo
        """
        p = Path(path)
        if not p.exists():
            return pd.DataFrame()
        out = []
        for chunk in pd.read_csv(p, chunksize=200000):
            cols = {c.strip().lower(): c for c in chunk.columns}
            tcol = cols.get("ticker")
            m = (chunk[tcol] == ticker) if tcol else pd.Series(True, index=chunk.index)
            if m.any():
                part = chunk.loc[m].copy()
                part.columns = [c.strip().lower() for c in part.columns]
                out.append(part)
            elif out:
                break
        if not out:
            return pd.DataFrame()
        df = pd.concat(out, ignore_index=True)

        need = {"date", "weekly_range", "weekly_range_avg", "weekly_range_hi", "weekly_range_lo"}
        if not need.issubset(df.columns):
            return pd.DataFrame()

        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        for c in ["weekly_range", "weekly_range_avg", "weekly_range_hi", "weekly_range_lo"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        return df.sort_values("date").reset_index(drop=True)


    @st.cache_data(show_spinner=False)
    def load_g18_ticker(path: Path, ticker: str) -> pd.DataFrame:
        """
        Graph 18: Weekly Volume with Avg/High/Low bands
        expected (case-insensitive):
            date, [ticker], weekly_volume, weekly_volume_avg, weekly_volume_hi, weekly_volume_lo
        """
        p = Path(path)
        if not p.exists():
            return pd.DataFrame()
        out = []
        for chunk in pd.read_csv(p, chunksize=200000):
            cols = {c.strip().lower(): c for c in chunk.columns}
            tcol = cols.get("ticker")
            m = (chunk[tcol] == ticker) if tcol else pd.Series(True, index=chunk.index)
            if m.any():
                part = chunk.loc[m].copy()
                part.columns = [c.strip().lower() for c in part.columns]
                out.append(part)
            elif out:
                break
        if not out:
            return pd.DataFrame()
        df = pd.concat(out, ignore_index=True)

        need = {"date", "weekly_volume", "weekly_volume_avg", "weekly_volume_hi", "weekly_volume_lo"}
        if not need.issubset(df.columns):
            return pd.DataFrame()

        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        for c in ["weekly_volume", "weekly_volume_avg", "weekly_volume_hi", "weekly_volume_lo"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        return df.sort_values("date").reset_index(drop=True)

    # ---- Plotters ----
    def plot_g16_weekly_returns(df: pd.DataFrame, ticker: str):
        fig, ax = plt.subplots(figsize=(9.5, 3.9), dpi=150)

        # bar colors by sign
        colors = ["green" if v >= 0 else "red" for v in df["weekly_return_pct"]]
        ax.bar(df["date"], df["weekly_return_pct"], width=5.0, color=colors, linewidth=0)

        # bands
        ax.axhline(y=df["weekly_return_avg_pct"].iloc[0], color="black", linewidth=1.6, label="Avg")
        ax.axhline(y=df["weekly_return_hi_pct"].iloc[0],  color="red",   linewidth=1.2, label="High")
        ax.axhline(y=df["weekly_return_lo_pct"].iloc[0],  color="green", linewidth=1.2, label="Low")

        ax.set_title(f"{ticker} – Weekly Returns", fontsize=12, pad=6)
        ax.grid(True, linewidth=0.4, alpha=0.4)

        from matplotlib.ticker import PercentFormatter
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=100))

        # show weekly ticks (Mondays) and keep the 5-day gutter helper for consistency
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO, interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d/%y"))
        plt.setp(ax.get_xticklabels(), rotation=90, ha="center", fontsize=8)

        pad = pd.Timedelta(days=5)
        ax.set_xlim(df["date"].min() - pad, df["date"].max() + pad)

        from matplotlib.lines import Line2D
        handles = [
            Line2D([0], [0], color="black", linewidth=1.6, label="Avg"),
            Line2D([0], [0], color="red",   linewidth=1.2, label="High"),
            Line2D([0], [0], color="green", linewidth=1.2, label="Low"),
        ]
        ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.22),
                ncol=3, frameon=False, handlelength=2.8, fontsize=9)
        fig.subplots_adjust(bottom=0.30)
        plt.close(fig)  # 🔑 Prevents too many open figures
        return fig


    def plot_g17_weekly_range(df: pd.DataFrame, ticker: str):
        fig, ax = plt.subplots(figsize=(9.5, 3.9), dpi=150)

        ax.plot(df["date"], df["weekly_range"], color=EXCEL_BLUE, linewidth=1.6, label="Range")
        ax.axhline(y=df["weekly_range_avg"].iloc[0], color="black", linewidth=1.6, label="Avg")
        ax.axhline(y=df["weekly_range_hi"].iloc[0],  color="red",   linewidth=1.2, label="High")
        ax.axhline(y=df["weekly_range_lo"].iloc[0],  color="green", linewidth=1.2, label="Low")

        ax.set_title(f"{ticker} – Weekly Range", fontsize=12, pad=6)
        ax.grid(True, linewidth=0.4, alpha=0.4)

        ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO, interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d/%y"))
        plt.setp(ax.get_xticklabels(), rotation=90, ha="center", fontsize=8)

        pad = pd.Timedelta(days=5)
        ax.set_xlim(df["date"].min() - pad, df["date"].max() + pad)

        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22),
                ncol=4, frameon=False, handlelength=2.8, fontsize=9)
        fig.subplots_adjust(bottom=0.30)
        plt.close(fig)  # 🔑 Prevents too many open figures
        return fig


    def plot_g18_weekly_volume(df: pd.DataFrame, ticker: str):
        fig, ax = plt.subplots(figsize=(9.5, 3.9), dpi=150)

        ax.plot(df["date"], df["weekly_volume"], color=EXCEL_BLUE, linewidth=1.6, label="Volume")
        ax.axhline(y=df["weekly_volume_avg"].iloc[0], color="black", linewidth=1.6, label="Avg")
        ax.axhline(y=df["weekly_volume_hi"].iloc[0],  color="red",   linewidth=1.2, label="High")
        ax.axhline(y=df["weekly_volume_lo"].iloc[0],  color="green", linewidth=1.2, label="Low")

        ax.set_title(f"{ticker} – Weekly Volume", fontsize=12, pad=6)
        ax.grid(True, linewidth=0.4, alpha=0.4)

        ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO, interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d/%y"))
        plt.setp(ax.get_xticklabels(), rotation=90, ha="center", fontsize=8)

        pad = pd.Timedelta(days=5)
        ax.set_xlim(df["date"].min() - pad, df["date"].max() + pad)

        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22),
                ncol=4, frameon=False, handlelength=2.8, fontsize=9)
        fig.subplots_adjust(bottom=0.30)
        plt.close(fig)  # 🔑 Prevents too many open figures
        return fig

    # ---- Render row horizontally (16, 17, 18) ----
    col16, col17, col18 = st.columns(3, gap="large")

    with col16:
        _ticker = st.session_state.get("ticker_select", DEFAULT_TICKER)
        _rng    = st.session_state.get("range_sel", "All")
        df16_all = load_g16_ticker(FILE_G16, _ticker)
        if df16_all.empty:
            st.info("No Weekly Returns data.")
        else:
            df16v = apply_window_with_gutter(df16_all, _rng, date_col="date", gutter_days=5)
            st.pyplot(plot_g16_weekly_returns(df16v, _ticker), clear_figure=True)

    with col17:
        _ticker = st.session_state.get("ticker_select", DEFAULT_TICKER)
        _rng    = st.session_state.get("range_sel", "All")
        df17_all = load_g17_ticker(FILE_G17, _ticker)
        if df17_all.empty:
            st.info("No Weekly Range data.")
        else:
            df17v = apply_window_with_gutter(df17_all, _rng, date_col="date", gutter_days=5)
            st.pyplot(plot_g17_weekly_range(df17v, _ticker), clear_figure=True)

    with col18:
        _ticker = st.session_state.get("ticker_select", DEFAULT_TICKER)
        _rng    = st.session_state.get("range_sel", "All")
        df18_all = load_g18_ticker(FILE_G18, _ticker)
        if df18_all.empty:
            st.info("No Weekly Volume data.")
        else:
            df18v = apply_window_with_gutter(df18_all, _rng, date_col="date", gutter_days=5)
            st.pyplot(plot_g18_weekly_volume(df18v, _ticker), clear_figure=True)

    # ==============================
    # ===== Graphs 16, 17 & 18 (do not modify) END =====
    # ==============================

    # ==============================
    # ===== Graphs 19, 20 & 21 START =====
    # ==============================

    # ---- CSV paths ----
    FILE_G19 = DATA_DIR / "qry_graph_data_19.csv"  # Monthly Returns % + bands
    FILE_G20 = DATA_DIR / "qry_graph_data_20.csv"  # Monthly Range + bands
    FILE_G21 = DATA_DIR / "qry_graph_data_21.csv"  # Monthly Volume + bands

    # ---- Loaders (ticker-only) ----
    @st.cache_data(show_spinner=False)
    def load_g19_ticker(path: Path, ticker: str) -> pd.DataFrame:
        """
        Graph 19: Monthly Returns (%) with Avg/High/Low bands
          expected (case-insensitive):
            date, [ticker], monthly_return, monthly_return_avg, monthly_return_hi, monthly_return_lo
        """
        p = Path(path)
        if not p.exists():
            return pd.DataFrame()
        out = []
        for chunk in pd.read_csv(p, chunksize=200000):
            cols = {c.strip().lower(): c for c in chunk.columns}
            tcol = cols.get("ticker")
            m = (chunk[tcol] == ticker) if tcol else pd.Series(True, index=chunk.index)
            if m.any():
                part = chunk.loc[m].copy()
                part.columns = [c.strip().lower() for c in part.columns]
                out.append(part)
            elif out:
                break
        if not out:
            return pd.DataFrame()
        df = pd.concat(out, ignore_index=True)

        need = {"date", "monthly_return", "monthly_return_avg", "monthly_return_hi", "monthly_return_lo"}
        if not need.issubset(df.columns):
            return pd.DataFrame()

        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        for c in ["monthly_return", "monthly_return_avg", "monthly_return_hi", "monthly_return_lo"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        # If returns look like 0..1, convert to %
        mx = pd.concat([
            df["monthly_return"], df["monthly_return_avg"],
            df["monthly_return_hi"], df["monthly_return_lo"]
        ]).abs().max()
        if pd.notna(mx) and mx <= 1.0:
            df[["monthly_return", "monthly_return_avg",
                "monthly_return_hi", "monthly_return_lo"]] *= 100.0

        return df.sort_values("date").reset_index(drop=True)


    @st.cache_data(show_spinner=False)
    def load_g20_ticker(path: Path, ticker: str) -> pd.DataFrame:
        """
        Graph 20: Monthly Range with Avg/High/Low bands
          expected (case-insensitive):
            date, [ticker], monthly_range, monthly_range_avg, monthly_range_hi, monthly_range_lo
        """
        p = Path(path)
        if not p.exists():
            return pd.DataFrame()
        out = []
        for chunk in pd.read_csv(p, chunksize=200000):
            cols = {c.strip().lower(): c for c in chunk.columns}
            tcol = cols.get("ticker")
            m = (chunk[tcol] == ticker) if tcol else pd.Series(True, index=chunk.index)
            if m.any():
                part = chunk.loc[m].copy()
                part.columns = [c.strip().lower() for c in part.columns]
                out.append(part)
            elif out:
                break
        if not out:
            return pd.DataFrame()
        df = pd.concat(out, ignore_index=True)

        need = {"date", "monthly_range", "monthly_range_avg", "monthly_range_hi", "monthly_range_lo"}
        if not need.issubset(df.columns):
            return pd.DataFrame()

        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        for c in ["monthly_range", "monthly_range_avg", "monthly_range_hi", "monthly_range_lo"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        return df.sort_values("date").reset_index(drop=True)


    @st.cache_data(show_spinner=False)
    def load_g21_ticker(path: Path, ticker: str) -> pd.DataFrame:
        """
        Graph 21: Monthly Volume with Avg/High/Low bands
          expected (case-insensitive):
            date, [ticker], monthly_volume, monthly_volume_avg, monthly_volume_hi, monthly_volume_lo
        """
        p = Path(path)
        if not p.exists():
            return pd.DataFrame()
        out = []
        for chunk in pd.read_csv(p, chunksize=200000):
            cols = {c.strip().lower(): c for c in chunk.columns}
            tcol = cols.get("ticker")
            m = (chunk[tcol] == ticker) if tcol else pd.Series(True, index=chunk.index)
            if m.any():
                part = chunk.loc[m].copy()
                part.columns = [c.strip().lower() for c in part.columns]
                out.append(part)
            elif out:
                break
        if not out:
            return pd.DataFrame()
        df = pd.concat(out, ignore_index=True)

        need = {"date", "monthly_volume", "monthly_volume_avg", "monthly_volume_hi", "monthly_volume_lo"}
        if not need.issubset(df.columns):
            return pd.DataFrame()

        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        for c in ["monthly_volume", "monthly_volume_avg", "monthly_volume_hi", "monthly_volume_lo"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        return df.sort_values("date").reset_index(drop=True)

    # ---- Plotters ----
    def plot_g19_monthly_returns(df: pd.DataFrame, ticker: str):
        fig, ax = plt.subplots(figsize=(9.5, 3.9), dpi=150)
        colors = ["green" if v >= 0 else "red" for v in df["monthly_return"]]
        ax.bar(df["date"], df["monthly_return"], width=20.0, color=colors, linewidth=0)

        ax.axhline(y=df["monthly_return_avg"].iloc[0], color="black", linewidth=1.6, label="Avg")
        ax.axhline(y=df["monthly_return_hi"].iloc[0],  color="red",   linewidth=1.2, label="High")
        ax.axhline(y=df["monthly_return_lo"].iloc[0],  color="green", linewidth=1.2, label="Low")

        ax.set_title(f"{ticker} – Monthly Returns", fontsize=12, pad=6)
        ax.grid(True, linewidth=0.4, alpha=0.4)

        from matplotlib.ticker import PercentFormatter
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=100))

        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d/%y"))
        plt.setp(ax.get_xticklabels(), rotation=90, ha="center", fontsize=8)

        pad = pd.Timedelta(days=5)
        ax.set_xlim(df["date"].min() - pad, df["date"].max() + pad)

        from matplotlib.lines import Line2D
        handles = [
            Line2D([0], [0], color="black", linewidth=1.6, label="Avg"),
            Line2D([0], [0], color="red",   linewidth=1.2, label="High"),
            Line2D([0], [0], color="green", linewidth=1.2, label="Low"),
        ]
        ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.22),
                  ncol=3, frameon=False, handlelength=2.8, fontsize=9)
        fig.subplots_adjust(bottom=0.30)
        plt.close(fig)  # 🔑 Prevents too many open figures
        return fig


    def plot_g20_monthly_range(df: pd.DataFrame, ticker: str):
        fig, ax = plt.subplots(figsize=(9.5, 3.9), dpi=150)
        ax.plot(df["date"], df["monthly_range"], color=EXCEL_BLUE, linewidth=1.6, label="Range")
        ax.axhline(y=df["monthly_range_avg"].iloc[0], color="black", linewidth=1.6, label="Avg")
        ax.axhline(y=df["monthly_range_hi"].iloc[0],  color="red",   linewidth=1.2, label="High")
        ax.axhline(y=df["monthly_range_lo"].iloc[0],  color="green", linewidth=1.2, label="Low")

        ax.set_title(f"{ticker} – Monthly Range", fontsize=12, pad=6)
        ax.grid(True, linewidth=0.4, alpha=0.4)

        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d/%y"))
        plt.setp(ax.get_xticklabels(), rotation=90, ha="center", fontsize=8)

        pad = pd.Timedelta(days=5)
        ax.set_xlim(df["date"].min() - pad, df["date"].max() + pad)

        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22),
                  ncol=4, frameon=False, handlelength=2.8, fontsize=9)
        fig.subplots_adjust(bottom=0.30)
        plt.close(fig)  # 🔑 Prevents too many open figures
        return fig


    def plot_g21_monthly_volume(df: pd.DataFrame, ticker: str):
        fig, ax = plt.subplots(figsize=(9.5, 3.9), dpi=150)
        ax.plot(df["date"], df["monthly_volume"], color=EXCEL_BLUE, linewidth=1.6, label="Volume")
        ax.axhline(y=df["monthly_volume_avg"].iloc[0], color="black", linewidth=1.6, label="Avg")
        ax.axhline(y=df["monthly_volume_hi"].iloc[0],  color="red",   linewidth=1.2, label="High")
        ax.axhline(y=df["monthly_volume_lo"].iloc[0],  color="green", linewidth=1.2, label="Low")

        ax.set_title(f"{ticker} – Monthly Volume", fontsize=12, pad=6)
        ax.grid(True, linewidth=0.4, alpha=0.4)

        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d/%y"))
        plt.setp(ax.get_xticklabels(), rotation=90, ha="center", fontsize=8)

        pad = pd.Timedelta(days=5)
        ax.set_xlim(df["date"].min() - pad, df["date"].max() + pad)

        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22),
                  ncol=4, frameon=False, handlelength=2.8, fontsize=9)
        fig.subplots_adjust(bottom=0.30)
        plt.close(fig)  # 🔑 Prevents too many open figures
        return fig

    # ---- Render row horizontally (19, 20, 21) ----
    col19, col20, col21 = st.columns(3, gap="large")

    with col19:
        _ticker = st.session_state.get("ticker_select", DEFAULT_TICKER)
        _rng    = st.session_state.get("range_sel", "All")
        df19_all = load_g19_ticker(FILE_G19, _ticker)
        if df19_all.empty:
            st.info("No Monthly Returns data.")
        else:
            df19v = apply_window_with_gutter(df19_all, _rng, date_col="date", gutter_days=5)
            st.pyplot(plot_g19_monthly_returns(df19v, _ticker), clear_figure=True)

    with col20:
        _ticker = st.session_state.get("ticker_select", DEFAULT_TICKER)
        _rng    = st.session_state.get("range_sel", "All")
        df20_all = load_g20_ticker(FILE_G20, _ticker)
        if df20_all.empty:
            st.info("No Monthly Range data.")
        else:
            df20v = apply_window_with_gutter(df20_all, _rng, date_col="date", gutter_days=5)
            st.pyplot(plot_g20_monthly_range(df20v, _ticker), clear_figure=True)

    with col21:
        _ticker = st.session_state.get("ticker_select", DEFAULT_TICKER)
        _rng    = st.session_state.get("range_sel", "All")
        df21_all = load_g21_ticker(FILE_G21, _ticker)
        if df21_all.empty:
            st.info("No Monthly Volume data.")
        else:
            df21v = apply_window_with_gutter(df21_all, _rng, date_col="date", gutter_days=5)
            st.pyplot(plot_g21_monthly_volume(df21v, _ticker), clear_figure=True)

    # ==============================
    # ===== Graphs 19, 20 & 21 (do not modify) END =====
    # ==============================

        # ==============================
    # ===== Graphs 22, 23 & 24 START =====
    # ==============================

    # ---- CSV paths ----
    FILE_G22 = DATA_DIR / "qry_graph_data_22.csv"  # Short-Term Trend + bands
    FILE_G23 = DATA_DIR / "qry_graph_data_23.csv"  # Mid-Term Trend + bands
    FILE_G24 = DATA_DIR / "qry_graph_data_24.csv"  # Long-Term Trend + bands

    # ---- Loaders (ticker-only) ----
    @st.cache_data(show_spinner=False)
    def load_g22_ticker(path: Path, ticker: str) -> pd.DataFrame:
        """
        Graph 22: Short-Term Trend with Avg/High/Low bands
          expected (case-insensitive):
            date, [ticker], st_trend, st_avg, st_hi, st_lo
        """
        p = Path(path)
        if not p.exists():
            return pd.DataFrame()
        out = []
        for chunk in pd.read_csv(p, chunksize=200000):
            # case-insensitive access
            cols = {c.strip().lower(): c for c in chunk.columns}
            tcol = cols.get("ticker")
            m = (chunk[tcol] == ticker) if tcol else pd.Series(True, index=chunk.index)
            if m.any():
                part = chunk.loc[m].copy()
                part.columns = [c.strip().lower() for c in part.columns]
                out.append(part)
            elif out:
                break
        if not out:
            return pd.DataFrame()
        df = pd.concat(out, ignore_index=True)

        need = {"date", "st_trend", "st_avg", "st_hi", "st_lo"}
        if not need.issubset(df.columns):
            return pd.DataFrame()

        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        for c in ["st_trend", "st_avg", "st_hi", "st_lo"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        # If values look like 0..1, convert to % scale
        mx = pd.concat([df["st_trend"], df["st_avg"], df["st_hi"], df["st_lo"]]).abs().max()
        if pd.notna(mx) and mx <= 1.0:
            df[["st_trend", "st_avg", "st_hi", "st_lo"]] *= 100.0

        return df.sort_values("date").reset_index(drop=True)


    @st.cache_data(show_spinner=False)
    def load_g23_ticker(path: Path, ticker: str) -> pd.DataFrame:
        """
        Graph 23: Mid-Term Trend with Avg/High/Low bands
          expected (case-insensitive):
            date, [ticker], mt_trend, mt_avg, mt_hi, mt_lo
        """
        p = Path(path)
        if not p.exists():
            return pd.DataFrame()
        out = []
        for chunk in pd.read_csv(p, chunksize=200000):
            cols = {c.strip().lower(): c for c in chunk.columns}
            tcol = cols.get("ticker")
            m = (chunk[tcol] == ticker) if tcol else pd.Series(True, index=chunk.index)
            if m.any():
                part = chunk.loc[m].copy()
                part.columns = [c.strip().lower() for c in part.columns]
                out.append(part)
            elif out:
                break
        if not out:
            return pd.DataFrame()
        df = pd.concat(out, ignore_index=True)

        need = {"date", "mt_trend", "mt_avg", "mt_hi", "mt_lo"}
        if not need.issubset(df.columns):
            return pd.DataFrame()

        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        for c in ["mt_trend", "mt_avg", "mt_hi", "mt_lo"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        mx = pd.concat([df["mt_trend"], df["mt_avg"], df["mt_hi"], df["mt_lo"]]).abs().max()
        if pd.notna(mx) and mx <= 1.0:
            df[["mt_trend", "mt_avg", "mt_hi", "mt_lo"]] *= 100.0

        return df.sort_values("date").reset_index(drop=True)


    @st.cache_data(show_spinner=False)
    def load_g24_ticker(path: Path, ticker: str) -> pd.DataFrame:
        """
        Graph 24: Long-Term Trend with Avg/High/Low bands
          expected (case-insensitive):
            date, [ticker], lt_trend, lt_avg, lt_hi, lt_lo
        """
        p = Path(path)
        if not p.exists():
            return pd.DataFrame()
        out = []
        for chunk in pd.read_csv(p, chunksize=200000):
            cols = {c.strip().lower(): c for c in chunk.columns}
            tcol = cols.get("ticker")
            m = (chunk[tcol] == ticker) if tcol else pd.Series(True, index=chunk.index)
            if m.any():
                part = chunk.loc[m].copy()
                part.columns = [c.strip().lower() for c in part.columns]
                out.append(part)
            elif out:
                break
        if not out:
            return pd.DataFrame()
        df = pd.concat(out, ignore_index=True)

        need = {"date", "lt_trend", "lt_avg", "lt_hi", "lt_lo"}
        if not need.issubset(df.columns):
            return pd.DataFrame()

        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        for c in ["lt_trend", "lt_avg", "lt_hi", "lt_lo"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        mx = pd.concat([df["lt_trend"], df["lt_avg"], df["lt_hi"], df["lt_lo"]]).abs().max()
        if pd.notna(mx) and mx <= 1.0:
            df[["lt_trend", "lt_avg", "lt_hi", "lt_lo"]] *= 100.0

        return df.sort_values("date").reset_index(drop=True)

    # ---- Plotters ----
    def _plot_trend_generic(df: pd.DataFrame, ticker: str, series_col: str,
                            avg_col: str, hi_col: str, lo_col: str, title: str):
        fig, ax = plt.subplots(figsize=(9.5, 3.9), dpi=150)

        # main series
        ax.plot(df["date"], df[series_col], color=EXCEL_BLUE, linewidth=1.6, label=title.split(" – ")[-1])

        # horizontal bands
        ax.axhline(y=df[avg_col].iloc[0], color="gray",  linewidth=1.2, label="Avg")
        ax.axhline(y=df[hi_col].iloc[0],  color="red",   linewidth=1.2, label="High")
        ax.axhline(y=df[lo_col].iloc[0],  color="green", linewidth=1.2, label="Low")

        ax.set_title(title, fontsize=12, pad=6)
        ax.grid(True, linewidth=0.4, alpha=0.4)

        # Percent axis (values are in %)
        from matplotlib.ticker import PercentFormatter
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=100))

        # Month ticks + a small gutter
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d/%y"))
        plt.setp(ax.get_xticklabels(), rotation=90, ha="center", fontsize=8)

        pad = pd.Timedelta(days=5)
        ax.set_xlim(df["date"].min() - pad, df["date"].max() + pad)

        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22),
                  ncol=4, frameon=False, handlelength=2.8, fontsize=9)
        fig.subplots_adjust(bottom=0.30)
        plt.close(fig)  # 🔑 Prevents too many open figures
        return fig

    def plot_g22_st(df: pd.DataFrame, ticker: str):
        return _plot_trend_generic(
            df=df, ticker=ticker,
            series_col="st_trend", avg_col="st_avg", hi_col="st_hi", lo_col="st_lo",
            title=f"{ticker} – Short Term Trend Line"
        )

    def plot_g23_mt(df: pd.DataFrame, ticker: str):
        return _plot_trend_generic(
            df=df, ticker=ticker,
            series_col="mt_trend", avg_col="mt_avg", hi_col="mt_hi", lo_col="mt_lo",
            title=f"{ticker} – Mid Term Trend Line"
        )

    def plot_g24_lt(df: pd.DataFrame, ticker: str):
        return _plot_trend_generic(
            df=df, ticker=ticker,
            series_col="lt_trend", avg_col="lt_avg", hi_col="lt_hi", lo_col="lt_lo",
            title=f"{ticker} – Long Term Trend Line"
        )

    # ---- Render row horizontally (22, 23, 24) ----
    col22, col23, col24 = st.columns(3, gap="large")

    with col22:
        _ticker = st.session_state.get("ticker_select", DEFAULT_TICKER)
        _rng    = st.session_state.get("range_sel", "All")
        df22_all = load_g22_ticker(FILE_G22, _ticker)
        if df22_all.empty:
            st.info("No Short-Term Trend data.")
        else:
            df22v = apply_window_with_gutter(df22_all, _rng, date_col="date", gutter_days=5)
            st.pyplot(plot_g22_st(df22v, _ticker), clear_figure=True)

    with col23:
        _ticker = st.session_state.get("ticker_select", DEFAULT_TICKER)
        _rng    = st.session_state.get("range_sel", "All")
        df23_all = load_g23_ticker(FILE_G23, _ticker)
        if df23_all.empty:
            st.info("No Mid-Term Trend data.")
        else:
            df23v = apply_window_with_gutter(df23_all, _rng, date_col="date", gutter_days=5)
            st.pyplot(plot_g23_mt(df23v, _ticker), clear_figure=True)

    with col24:
        _ticker = st.session_state.get("ticker_select", DEFAULT_TICKER)
        _rng    = st.session_state.get("range_sel", "All")
        df24_all = load_g24_ticker(FILE_G24, _ticker)
        if df24_all.empty:
            st.info("No Long-Term Trend data.")
        else:
            df24v = apply_window_with_gutter(df24_all, _rng, date_col="date", gutter_days=5)
            st.pyplot(plot_g24_lt(df24v, _ticker), clear_figure=True)

    # ==============================
    # ===== Graphs 22, 23 & 24 (do not modify) END =====
    # ==============================