"""
ApexAdaptive multi-asset dashboard (Streamlit).

Secrets (Streamlit Cloud: App settings > Secrets, or .streamlit/secrets.toml locally):
    SUPABASE_URL = "https://xxxx.supabase.co"
    SUPABASE_SERVICE_KEY = "..."
Keep the app PRIVATE: it uses the service-role key. Never commit secrets.toml.
Run schema.sql, schema_ai.sql and schema_assets.sql in Supabase first.
"""
import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

st.set_page_config(page_title="ApexAdaptive", layout="wide")


def secret(name):
    try:
        return st.secrets[name]
    except Exception:
        return os.environ.get(name, "")


URL, KEY = secret("SUPABASE_URL"), secret("SUPABASE_SERVICE_KEY")
if not URL or not KEY:
    st.error("Set SUPABASE_URL and SUPABASE_SERVICE_KEY in the app's Secrets (or .streamlit/secrets.toml locally).")
    st.stop()
HEADERS = {"apikey": KEY, "Authorization": f"Bearer {KEY}"}
MISSING = []          # tables that do not exist yet (schema not run)


# ------------------------------------------------------------------ data access
@st.cache_data(ttl=60)
def fetch(table, order=None, select="*"):
    rows, start = [], 0
    while True:
        params = {"select": select}
        if order:
            params["order"] = order
        r = requests.get(f"{URL}/rest/v1/{table}", params=params, timeout=30,
                         headers={**HEADERS, "Range-Unit": "items", "Range": f"{start}-{start + 999}"})
        r.raise_for_status()
        batch = r.json()
        rows += batch
        if len(batch) < 1000:
            break
        start += 1000
    return pd.DataFrame(rows)


def load(table, order=None, select="*"):
    try:
        return fetch(table, order, select)
    except requests.HTTPError:
        MISSING.append(table)
        return pd.DataFrame()


def sb_write(method, table, *, params=None, json=None, prefer=None):
    h = {**HEADERS, "Content-Type": "application/json"}
    if prefer:
        h["Prefer"] = prefer
    r = requests.request(method, f"{URL}/rest/v1/{table}", headers=h, params=params, json=json, timeout=30)
    r.raise_for_status()


def ensure(df, cols):
    for c in cols:
        if c not in df:
            df[c] = pd.Series(dtype="object")
    return df


def num(df, cols):
    for c in cols:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def to_dt(df, cols):
    for c in cols:
        if c in df:
            df[c] = pd.to_datetime(df[c], utc=True, errors="coerce")
    return df


# ------------------------------------------------------------------ asset classes
CLASSES = ["Metals", "Forex", "Crypto", "Indices", "Energy", "Stocks", "Other"]
FX = {"USD", "EUR", "GBP", "JPY", "CHF", "AUD", "NZD", "CAD"}


def guess_class(sym):
    s = str(sym).upper()
    if s.startswith(("XAU", "XAG", "XPT", "XPD")):
        return "Metals"
    if s.startswith(("BTC", "ETH", "SOL", "XRP", "LTC", "ADA", "DOGE", "BNB", "DOT", "LINK")):
        return "Crypto"
    if any(k in s for k in ("US30", "US100", "US500", "NAS", "SPX", "DJ30", "GER", "DAX", "UK100", "JP225", "NDX")):
        return "Indices"
    if any(k in s for k in ("OIL", "WTI", "BRENT", "NGAS")):
        return "Energy"
    if len(s) >= 6 and s[:3] in FX and s[3:6] in FX:
        return "Forex"
    return "Other"


# ------------------------------------------------------------------ load everything
T_COLS = ["symbol", "magic", "direction", "opened_at", "closed_at", "last_deal", "closed", "lots", "entry_price",
          "sl_price", "risk_money", "pnl", "r_multiple", "sl_mult", "tp3_r", "adx_min", "adx", "bias_strength",
          "hour_open"]
A_COLS = ["symbol", "ts", "enabled", "sl_mult", "tp3_r", "adx_min", "train_n", "train_avg_r", "valid_n",
          "valid_avg_r", "note"]
S_COLS = ["symbol", "ts", "balance", "equity", "enabled", "open_positions"]
R_COLS = ["symbol", "asset_class", "enabled", "risk_pct", "notes"]
D_COLS = ["id", "created_at", "symbol", "mode", "model", "action", "sl_mult", "tp3_r", "confidence", "reason",
          "latency_ms"]
U_COLS = ["decision_id", "symbol", "setup_time", "taken", "r_rule", "r_ai"]

trades = to_dt(num(ensure(load("aa_trade_results", "opened_at.asc"), T_COLS),
                   ["pnl", "r_multiple", "risk_money", "lots", "bias_strength", "hour_open"]),
               ["opened_at", "closed_at", "last_deal"])
adapts = to_dt(num(ensure(load("aa_adaptations", "ts.desc"), A_COLS), ["sl_mult", "tp3_r", "adx_min", "valid_avg_r"]),
               ["ts"])
snaps = to_dt(num(ensure(load("aa_snapshots", "ts.asc"), S_COLS), ["balance", "equity", "open_positions"]), ["ts"])
assets = ensure(load("aa_assets", "symbol.asc"), R_COLS)
dec = to_dt(num(ensure(load("aa_ai_decisions", "created_at.desc", ",".join(D_COLS)), D_COLS),
                ["confidence", "latency_ms", "sl_mult", "tp3_r"]), ["created_at"])
sets = to_dt(num(ensure(load("aa_setups", "setup_time.desc"), U_COLS), ["r_rule", "r_ai"]), ["setup_time"])

reg_class = dict(zip(assets["symbol"], assets["asset_class"]))
reg_risk = dict(zip(assets["symbol"], assets["risk_pct"]))
reg_on = dict(zip(assets["symbol"], assets["enabled"]))
all_syms = sorted(set(trades["symbol"].dropna()) | set(snaps["symbol"].dropna()) |
                  set(adapts["symbol"].dropna()) | set(reg_class))


def klass(sym):
    return reg_class.get(sym) or guess_class(sym)


# ------------------------------------------------------------------ sidebar
st.sidebar.title("ApexAdaptive")
days = st.sidebar.slider("Window (days)", 7, 365, 90)
avail_classes = sorted({klass(s) for s in all_syms})
sel_classes = st.sidebar.multiselect("Asset classes", avail_classes, default=avail_classes)
class_syms = [s for s in all_syms if klass(s) in sel_classes]
sel_syms = st.sidebar.multiselect("Assets", class_syms, default=class_syms)

st.sidebar.subheader("Prop firm rules")
st.sidebar.caption("Set these to YOUR firm's rules. The risk tab measures against them.")
acct_size_in = st.sidebar.number_input("Account size (0 = earliest logged balance)", min_value=0.0, value=0.0,
                                       step=1000.0)
daily_lim = st.sidebar.number_input("Daily loss limit (%)", min_value=0.1, value=3.0, step=0.5)
dd_lim = st.sidebar.number_input("Max drawdown limit (%)", min_value=0.1, value=10.0, step=0.5)
dd_type = st.sidebar.selectbox("Drawdown type", ["Static (from initial balance)", "Trailing (from peak equity)"])
reset_h = st.sidebar.number_input("Trading day starts (UTC hour)", min_value=0, max_value=23, value=0)
if st.sidebar.button("Refresh now"):
    st.cache_data.clear()
    st.rerun()

now = pd.Timestamp.now(tz="UTC")
cutoff = now - pd.Timedelta(days=days)
today = (now - pd.Timedelta(hours=int(reset_h))).date()

# ------------------------------------------------------------------ derived data
tr = trades[trades["symbol"].isin(sel_syms)].copy()
tr["closed_at"] = tr["closed_at"].fillna(tr["last_deal"])
closed = tr[(tr["closed"] == True) & (tr["closed_at"] >= cutoff)].sort_values("closed_at").copy()  # noqa: E712
closed["pnl"] = closed["pnl"].fillna(0.0)
closed["class"] = closed["symbol"].map(klass)
closed["day"] = (closed["closed_at"] - pd.Timedelta(hours=int(reset_h))).dt.date
closed["dow"] = closed["opened_at"].dt.dayofweek
open_tr = tr[(tr["closed"] != True) & (tr["opened_at"] >= now - pd.Timedelta(days=7))].copy()  # noqa: E712
open_tr["class"] = open_tr["symbol"].map(klass)

# account-wide hourly equity (one EA instance per asset posts snapshots; equity is account level)
acct = pd.DataFrame(columns=["equity", "balance"], index=pd.DatetimeIndex([], tz="UTC"))
if snaps["ts"].notna().any():
    acct = (snaps.dropna(subset=["ts"]).set_index("ts")[["equity", "balance"]].sort_index()
            .resample("1h").last().dropna())
size = acct_size_in if acct_size_in > 0 else (float(acct["balance"].iloc[0]) if not acct.empty else 0.0)
eq_now = float(acct["equity"].iloc[-1]) if not acct.empty else None
bal_now = float(acct["balance"].iloc[-1]) if not acct.empty else None
last_hb = snaps["ts"].max() if snaps["ts"].notna().any() else None
hb_age_h = (now - last_hb).total_seconds() / 3600 if last_hb is not None else None

day_loss_pct = None          # positive = loss since the trading day started
if not acct.empty:
    day_idx = (acct.index - pd.Timedelta(hours=int(reset_h))).date
    todays = acct[day_idx == today]
    day_start_bal = float(todays["balance"].iloc[0]) if not todays.empty else bal_now
    if day_start_bal:
        day_loss_pct = (day_start_bal - eq_now) / day_start_bal * 100

dd_used_pct = None
if eq_now is not None and size > 0:
    if dd_type.startswith("Static"):
        dd_used_pct = (size - eq_now) / size * 100
    else:
        peak = max(float(acct["equity"].max()), size)
        dd_used_pct = (peak - eq_now) / peak * 100

last_ad = adapts.sort_values("ts").groupby("symbol").tail(1).set_index("symbol")
last_sn = snaps.sort_values("ts").groupby("symbol").tail(1).set_index("symbol")


# ------------------------------------------------------------------ KPI engine
def kpis(d, base):
    out = dict(n=0, net=None, win_rate=None, pf=None, exp_r=None, avg_win=None, avg_loss=None, payoff=None,
               sqn=None, max_dd=None, max_dd_pct=None, recovery=None, max_losses=None, avg_hold_h=None)
    if d.empty:
        return out
    d = d.sort_values("closed_at")
    pnl = d["pnl"].astype(float).fillna(0.0)
    r = d["r_multiple"].astype(float).dropna()
    wins, losses = pnl[pnl > 0], pnl[pnl < 0]
    gl = abs(losses.sum())
    b = base if base > 0 else 0.0
    curve = np.concatenate([[b], b + pnl.cumsum().to_numpy()])
    peak = np.maximum.accumulate(curve)
    dd = curve - peak
    streak = longest = 0
    for p in pnl:
        streak = streak + 1 if p < 0 else 0
        longest = max(longest, streak)
    hold = (d["closed_at"] - d["opened_at"]).dropna()
    avg_w = wins.mean() if len(wins) else np.nan
    avg_l = losses.mean() if len(losses) else np.nan
    out.update(
        n=len(d), net=float(pnl.sum()), win_rate=float((pnl > 0).mean()),
        pf=(wins.sum() / gl) if gl > 0 else (np.inf if wins.sum() > 0 else np.nan),
        exp_r=float(r.mean()) if len(r) else np.nan, avg_win=avg_w, avg_loss=avg_l,
        payoff=(avg_w / abs(avg_l)) if len(wins) and len(losses) else np.nan,
        sqn=float(r.mean() / r.std(ddof=1) * np.sqrt(len(r))) if len(r) >= 3 and r.std(ddof=1) > 0 else np.nan,
        max_dd=float(dd.min()), max_dd_pct=float((dd / peak).min() * 100) if b > 0 else np.nan,
        recovery=float(pnl.sum() / abs(dd.min())) if dd.min() < 0 else np.nan,
        max_losses=longest, avg_hold_h=float(hold.mean().total_seconds() / 3600) if len(hold) else np.nan)
    return out


def money(x):
    return "–" if x is None or pd.isna(x) else f"{x:,.2f}"


def ratio(x):
    if x is None or pd.isna(x):
        return "–"
    return "∞" if np.isinf(x) else f"{x:.2f}"


def pct(x, d=0):
    return "–" if x is None or pd.isna(x) else f"{x * 100:.{d}f}%"


def kpi_table(groups, label):
    rows = []
    for name, d in groups.items():
        k = kpis(d, size)
        rows.append({label: name, "Trades": k["n"], "Net P/L": k["net"],
                     "Win %": None if k["win_rate"] is None else round(k["win_rate"] * 100, 1),
                     "Profit factor": k["pf"], "Expectancy (R)": k["exp_r"], "Payoff": k["payoff"],
                     "SQN": k["sqn"], "Max DD": k["max_dd"], "Recovery": k["recovery"],
                     "Max loss streak": k["max_losses"]})
    df = pd.DataFrame(rows)
    return df.round(2) if not df.empty else df


def show(fig):
    st.plotly_chart(fig, width="stretch")


def table(df):
    st.dataframe(df, width="stretch", hide_index=True)


def gauge(label, used, limit):
    if used is None:
        st.caption(f"{label}: no data yet")
        return
    frac = min(max(used / limit, 0.0), 1.0)
    st.progress(frac, text=f"{label}: {used:.2f}% of {limit:.1f}% limit ({frac:.0%} used)")
    if frac >= 0.7:
        st.warning(f"{label} is above 70% of the limit.")


# ------------------------------------------------------------------ page
st.title("ApexAdaptive")
if MISSING:
    st.warning("Missing tables: " + ", ".join(sorted(set(MISSING))) +
               ". Run schema.sql, schema_ai.sql and schema_assets.sql in the Supabase SQL editor.")
if hb_age_h is not None and hb_age_h > 3:
    st.warning(f"Last heartbeat {hb_age_h:.1f} h ago. Expected when markets are closed (weekends); "
               "otherwise check the EA, the VPS and the WebRequest URL.")

t_over, t_assets, t_perf, t_risk, t_ai, t_log, t_manage = st.tabs(
    ["Overview", "Assets", "Performance", "Risk", "AI layer", "Trades & tuning", "Manage assets"])

kp = kpis(closed, size)

# ============================== Overview
with t_over:
    c = st.columns(6)
    c[0].metric("Equity", money(eq_now))
    c[1].metric("Balance", money(bal_now))
    c[2].metric("Day P/L", "–" if day_loss_pct is None else f"{-day_loss_pct:+.2f}%",
                help="Equity vs balance at the first snapshot of the trading day.")
    open_pos = int(last_sn.loc[last_sn.index.isin(sel_syms), "open_positions"].fillna(0).sum()) if not last_sn.empty else 0
    c[3].metric("Open positions", open_pos)
    open_risk = float(open_tr["risk_money"].fillna(0).sum())
    c[4].metric("Initial risk open", money(open_risk),
                help="Sum of the 1R money risk at entry of trades not yet closed. Moves to ~0 once breakeven is set.")
    c[5].metric("Net P/L (window)", money(kp["net"]))

    st.subheader(f"Closed trades, last {days} days")
    if kp["n"] < 30:
        st.caption("Fewer than 30 closed trades: treat every ratio below as noise.")
    k = st.columns(6)
    k[0].metric("Trades", kp["n"])
    k[1].metric("Win rate", pct(kp["win_rate"]))
    k[2].metric("Profit factor", ratio(kp["pf"]))
    k[3].metric("Expectancy", "–" if pd.isna(kp["exp_r"] or np.nan) else f"{kp['exp_r']:+.2f} R")
    k[4].metric("Payoff (avg win / avg loss)", ratio(kp["payoff"]))
    k[5].metric("Max drawdown (closed)", money(kp["max_dd"]),
                help="Peak to trough of the realised P/L curve in this window.")

    left, right = st.columns(2)
    with left:
        w = acct[acct.index >= cutoff]
        if w.empty:
            st.info("No equity snapshots in this window yet. The first one is logged within an hour of the EA starting.")
        else:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=w.index, y=w["equity"], name="Equity"))
            fig.add_trace(go.Scatter(x=w.index, y=w["balance"], name="Balance"))
            fig.update_layout(title="Account equity (hourly)", margin=dict(t=40, b=10))
            show(fig)
    with right:
        if closed.empty:
            st.info("No closed trades in this window yet.")
        else:
            closed["cum"] = closed.groupby("symbol")["pnl"].cumsum()
            fig = px.line(closed, x="closed_at", y="cum", color="symbol", labels={"cum": "Cumulative P/L", "closed_at": ""},
                          title="Cumulative P/L by asset")
            fig.add_trace(go.Scatter(x=closed["closed_at"], y=closed["pnl"].cumsum(), name="Portfolio",
                                     line=dict(color="black", dash="dot")))
            fig.update_layout(margin=dict(t=40, b=10))
            show(fig)

# ============================== Assets
with t_assets:
    st.subheader("Asset board")
    rows = []
    for s in sel_syms:
        k_ = kpis(closed[closed["symbol"] == s], size)
        ad = last_ad.loc[s] if s in last_ad.index else None
        sn = last_sn.loc[s] if s in last_sn.index else None
        if reg_on.get(s) is False:
            state = "Paused"
        elif ad is None and sn is None:
            state = "Waiting for data"
        elif ad is None:
            state = "No re-tune yet"
        else:
            state = "ACTIVE" if ad["enabled"] else "STAND DOWN"
        age = (now - sn["ts"]).total_seconds() / 3600 if sn is not None and pd.notna(sn["ts"]) else np.nan
        rows.append({
            "Asset": s, "Class": klass(s), "State": state,
            "Params in force": "–" if ad is None or not ad["enabled"] else
            f"SL {ad['sl_mult']}xATR / TP2 {ad['tp3_r']}R / ADX>={ad['adx_min']}",
            "Planned risk %": reg_risk.get(s), "Heartbeat (h ago)": age,
            "Open": None if sn is None else sn["open_positions"],
            "Trades": k_["n"], "Net P/L": k_["net"],
            "Win %": None if k_["win_rate"] is None else k_["win_rate"] * 100,
            "Profit factor": k_["pf"], "Expectancy (R)": k_["exp_r"], "SQN": k_["sqn"], "Max DD": k_["max_dd"]})
    if rows:
        table(pd.DataFrame(rows).round(2))
        st.caption("State comes from each asset's latest walk-forward re-tune: the EA stands down by itself when "
                   "no parameter set validates. Heartbeat gaps are normal while that market is closed.")
    else:
        st.info("No assets yet. Add one in the Manage assets tab, or attach the EA to a chart.")

    if not closed.empty:
        st.subheader("Comparison")
        by_asset = {s: closed[closed["symbol"] == s] for s in sorted(closed["symbol"].unique())}
        table(kpi_table(by_asset, "Asset"))
        by_class = {c_: closed[closed["class"] == c_] for c_ in sorted(closed["class"].unique())}
        table(kpi_table(by_class, "Class"))

        left, right = st.columns(2)
        with left:
            contrib = closed.groupby("symbol")["pnl"].sum().reset_index()
            show(px.bar(contrib, x="symbol", y="pnl", title="Net P/L contribution by asset"))
        with right:
            closed["roll_r"] = closed.groupby("symbol")["r_multiple"].transform(
                lambda s_: s_.rolling(20, min_periods=10).mean())
            rr = closed.dropna(subset=["roll_r"])
            if rr.empty:
                st.info("Rolling expectancy needs at least 10 closed trades per asset.")
            else:
                show(px.line(rr, x="closed_at", y="roll_r", color="symbol",
                             title="Rolling expectancy, last 20 trades (R)", labels={"roll_r": "avg R", "closed_at": ""}))

# ============================== Performance
with t_perf:
    if closed.empty:
        st.info("No closed trades in this window yet.")
    else:
        r = closed["r_multiple"].astype(float)
        daily = closed.groupby("day")["pnl"].sum()
        a = st.columns(6)
        a[0].metric("Expectancy (money)", money(closed["pnl"].mean()))
        a[1].metric("Avg win", money(kp["avg_win"]))
        a[2].metric("Avg loss", money(kp["avg_loss"]))
        a[3].metric("SQN", ratio(kp["sqn"]), help="Expectancy / std-dev of R, scaled by sqrt(trades). Above ~2 is decent; "
                                                  "only meaningful with 30+ trades.")
        a[4].metric("Recovery factor", ratio(kp["recovery"]), help="Net P/L divided by max closed drawdown.")
        a[5].metric("Longest loss streak", kp["max_losses"])
        b = st.columns(6)
        b[0].metric("Best trade", money(closed["pnl"].max()))
        b[1].metric("Worst trade", money(closed["pnl"].min()))
        b[2].metric("Best day", money(daily.max()))
        b[3].metric("Worst day", money(daily.min()))
        b[4].metric("Green days", f"{(daily > 0).mean():.0%}")
        b[5].metric("Avg hold", "–" if pd.isna(kp["avg_hold_h"]) else f"{kp['avg_hold_h']:.1f} h")

        left, right = st.columns(2)
        with left:
            d_df = daily.reset_index()
            d_df.columns = ["day", "pnl"]
            fig = px.bar(d_df, x="day", y="pnl", title="Daily realised P/L",
                         color=(d_df["pnl"] > 0).map({True: "Gain", False: "Loss"}),
                         color_discrete_map={"Gain": "#2e9e5b", "Loss": "#d64545"})
            fig.update_layout(showlegend=False)
            show(fig)
        with right:
            show(px.histogram(r.dropna(), nbins=30, title="R-multiple distribution", labels={"value": "R"}))

        left, right = st.columns(2)
        with left:
            bh = closed.groupby("hour_open")["r_multiple"].agg(["mean", "count"]).reset_index()
            show(px.bar(bh, x="hour_open", y="mean", hover_data=["count"], title="Avg R by entry hour (broker time)"))
        with right:
            names = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}
            bw = closed.groupby("dow")["r_multiple"].agg(["mean", "count"]).reset_index()
            bw["day"] = bw["dow"].map(names)
            show(px.bar(bw, x="day", y="mean", hover_data=["count"], title="Avg R by weekday"))

        left, right = st.columns(2)
        with left:
            closed["bias_bucket"] = pd.cut(closed["bias_strength"].astype(float), [0, 0.5, 0.65, 0.8, 1.0001]).astype(str)
            bb = closed.groupby("bias_bucket")["r_multiple"].agg(["mean", "count"]).reset_index()
            show(px.bar(bb, x="bias_bucket", y="mean", hover_data=["count"], title="Avg R by Asia bias strength"))
        with right:
            bd = closed.groupby("direction")["r_multiple"].agg(["mean", "count"]).reset_index()
            show(px.bar(bd, x="direction", y="mean", hover_data=["count"], title="Avg R by direction"))
        st.caption("Bars built from only a handful of trades are noise: check the count in the hover text.")

# ============================== Risk
with t_risk:
    st.subheader("Prop firm limits")
    g1, g2 = st.columns(2)
    with g1:
        gauge("Daily loss used", day_loss_pct, daily_lim)
    with g2:
        gauge("Max drawdown used", dd_used_pct, dd_lim)
    st.caption(f"Account size used: {money(size)} ({dd_type.split(' ')[0].lower()} drawdown). Day-start balance is the first "
               "hourly snapshot of the trading day, so it can differ slightly from your firm's own figure.")

    w = acct[acct.index >= cutoff]
    if not w.empty:
        dd_curve = (w["equity"] / w["equity"].cummax() - 1) * 100
        fig = go.Figure(go.Scatter(x=dd_curve.index, y=dd_curve, fill="tozeroy", name="Drawdown %"))
        fig.update_layout(title="Equity drawdown from running peak (%)", margin=dict(t=40, b=10))
        show(fig)

    st.subheader("Open exposure")
    if open_tr.empty:
        st.info("No open trades logged in the last 7 days.")
    else:
        by_c = open_tr.groupby("class")["risk_money"].sum().reset_index()
        show(px.bar(by_c, x="class", y="risk_money", title="Initial risk of open trades by class"))
        table(open_tr[["opened_at", "symbol", "direction", "lots", "entry_price", "sl_price", "risk_money"]]
              .sort_values("opened_at", ascending=False))
    st.caption("Each EA instance halts itself at its own daily loss limit (InpDailyLossPct). Several assets can each "
               "risk their own percentage at the same time, so watch combined open risk here.")

# ============================== AI layer
with t_ai:
    d_ = dec[dec["symbol"].isin(sel_syms) & (dec["created_at"] >= cutoff)]
    if d_.empty:
        st.info("No AI decisions logged. The AI layer is optional: it needs InpAIMode 1 or 2 in the EA and an "
                "ANTHROPIC_API_KEY in the Modal secret.")
    else:
        m = d_.merge(sets[["decision_id", "r_rule", "r_ai", "taken"]], left_on="id", right_on="decision_id", how="left")
        done = m.dropna(subset=["r_rule"])
        q = st.columns(4)
        q[0].metric("AI calls", len(m))
        q[1].metric("TRADE share", f"{(m['action'] == 'TRADE').mean():.0%}")
        q[2].metric("Avg latency", f"{m['latency_ms'].mean() / 1000:.1f} s")
        q[3].metric("Outcomes resolved", len(done), help="Simulated about 24h after each call.")
        if done.empty:
            st.caption("Outcomes appear about a day after each decision. Judge the AI only with 30+ per side.")
        else:
            left, right = st.columns(2)
            with left:
                by_act = done.groupby("action")["r_rule"].agg(["mean", "count"]).reset_index()
                show(px.bar(by_act, x="action", y="mean", hover_data=["count"],
                            title="Avg simulated R by AI call (SKIP should be lower)"))
            with right:
                t_ = done[done["action"] == "TRADE"]
                cmp_ = pd.DataFrame({"who": ["Rule SL/TP", "AI SL/TP"], "avg R": [t_["r_rule"].mean(), t_["r_ai"].mean()]})
                show(px.bar(cmp_, x="who", y="avg R", title="On TRADE calls: rule vs AI stop/target"))
        table(m[["created_at", "symbol", "action", "confidence", "sl_mult", "tp3_r", "r_rule", "r_ai", "taken", "reason"]].head(50))

# ============================== Trades & tuning
with t_log:
    st.subheader("Re-tune history")
    ah = adapts[adapts["symbol"].isin(sel_syms)]
    if ah.empty:
        st.info("No re-tunes logged yet.")
    else:
        table(ah.drop(columns=["id", "magic"], errors="ignore"))
    st.subheader("Recent trades")
    if tr.empty:
        st.info("No trades logged yet.")
    else:
        cols = ["opened_at", "symbol", "direction", "lots", "entry_price", "sl_price", "pnl", "r_multiple", "closed",
                "sl_mult", "tp3_r", "adx_min", "adx", "bias_strength"]
        table(tr.sort_values("opened_at", ascending=False)[cols].head(100))

# ============================== Manage assets
with t_manage:
    st.subheader("Asset registry")
    if "aa_assets" in MISSING:
        st.warning("Run schema_assets.sql in the Supabase SQL editor to enable the registry.")
    else:
        with st.form("add_asset", clear_on_submit=True):
            f = st.columns([2, 1.5, 1, 2])
            sym_in = f[0].text_input("Broker symbol (exactly as in MT5, including any suffix)")
            cls_in = f[1].selectbox("Asset class", ["Auto-detect"] + CLASSES)
            risk_in = f[2].number_input("Planned risk %", min_value=0.0, value=0.5, step=0.1)
            note_in = f[3].text_input("Notes")
            on_in = st.checkbox("Enabled (untick to mark as paused)", value=True)
            if st.form_submit_button("Add / update asset"):
                s_ = sym_in.strip()
                if not s_:
                    st.error("Enter a symbol.")
                else:
                    try:
                        sb_write("POST", "aa_assets", params={"on_conflict": "symbol"},
                                 json={"symbol": s_, "asset_class": guess_class(s_) if cls_in == "Auto-detect" else cls_in,
                                       "enabled": on_in, "risk_pct": risk_in, "notes": note_in or None},
                                 prefer="resolution=merge-duplicates,return=minimal")
                        st.cache_data.clear()
                        st.toast(f"Saved {s_}")
                        st.rerun()
                    except Exception as e:  # noqa: BLE001
                        st.error(f"Save failed: {e}")

        if not assets.empty:
            table(assets[["symbol", "asset_class", "enabled", "risk_pct", "notes"]])
            rm = st.selectbox("Remove from registry", [""] + list(assets["symbol"]))
            if rm and st.button(f"Remove {rm}"):
                try:
                    sb_write("DELETE", "aa_assets", params={"symbol": f"eq.{rm}"}, prefer="return=minimal")
                    st.cache_data.clear()
                    st.rerun()
                except Exception as e:  # noqa: BLE001
                    st.error(f"Remove failed: {e}")
            st.caption("Removing an asset only deletes its registry row. Its trade history stays in the database.")
        else:
            st.info("No assets registered yet. Anything the EA has already logged still appears automatically.")

    with st.expander("How to start trading a new asset"):
        st.markdown(
            "The dashboard tracks assets; **the EA trades whatever chart it is attached to**. For each new asset:\n"
            "1. Open a chart of that symbol in MT5 (M15) and attach ApexAdaptive with the same URL and key.\n"
            "2. Adjust the gold-tuned inputs for that market: `InpMaxSpreadPts`, `InpSimCostPts`, the session hours "
            "(the Asia-bias logic assumes a quiet Asia session, which does not exist for 24/7 crypto) and `InpRiskPct`.\n"
            "3. Run it on demo first. The walk-forward tuner will keep it on STAND DOWN until a parameter set "
            "validates, which is the built-in guard against a strategy that doesn't fit that market.\n"
            "4. Watch the Assets tab: state, heartbeat and the KPIs appear once it logs data.")
