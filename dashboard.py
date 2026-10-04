"""
ApexAdaptive performance dashboard (Streamlit).

Run locally:   streamlit run dashboard.py
Secrets:       .streamlit/secrets.toml  ->  SUPABASE_URL = "..."  SUPABASE_SERVICE_KEY = "..."
Keep the app private: it uses the service-role key. Never commit secrets.toml.
"""
import os

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
    st.error("Set SUPABASE_URL and SUPABASE_SERVICE_KEY in .streamlit/secrets.toml")
    st.stop()
HEADERS = {"apikey": KEY, "Authorization": f"Bearer {KEY}"}


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


def to_dt(df, cols):
    for c in cols:
        if c in df:
            df[c] = pd.to_datetime(df[c], utc=True, errors="coerce")
    return df


trades = to_dt(fetch("aa_trade_results", "opened_at.asc"), ["opened_at", "closed_at", "last_deal"])
adapts = to_dt(fetch("aa_adaptations", "ts.desc"), ["ts"])
snaps = to_dt(fetch("aa_snapshots", "ts.asc"), ["ts"])

st.title("ApexAdaptive performance")

# ---------- sidebar filters ----------
symbols = sorted(set(trades.get("symbol", pd.Series(dtype=str)).dropna()) |
                 set(snaps.get("symbol", pd.Series(dtype=str)).dropna()))
sym = st.sidebar.selectbox("Symbol", symbols) if symbols else None
days = st.sidebar.slider("Window (days)", 7, 365, 90)
cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days)
if st.sidebar.button("Refresh now"):
    st.cache_data.clear()
    st.rerun()

if sym:
    trades = trades[trades["symbol"] == sym]
    adapts = adapts[adapts["symbol"] == sym] if not adapts.empty else adapts
    snaps = snaps[snaps["symbol"] == sym] if not snaps.empty else snaps

closed = trades[(trades.get("closed") == True) & (trades["opened_at"] >= cutoff)].copy() if not trades.empty else trades
closed = closed.sort_values("closed_at") if not closed.empty else closed

# ---------- status row ----------
c1, c2, c3 = st.columns(3)
if not adapts.empty:
    a = adapts.iloc[0]
    state = "ACTIVE" if a["enabled"] else "STAND DOWN"
    c1.metric("Bot state", state, help=f"Last re-tune {a['ts']:%Y-%m-%d %H:%M} UTC")
    c2.metric("Params in force", f"SL {a['sl_mult']}xATR / TP3 {a['tp3_r']}R / ADX>={a['adx_min']}")
    c3.metric("Validation avg R", f"{a['valid_avg_r']}" if pd.notna(a["valid_avg_r"]) else "n/a",
              help=f"train n={a['train_n']}, valid n={a['valid_n']}")
else:
    c1.info("No re-tune logged yet")

if not snaps.empty:
    last = snaps.iloc[-1]
    age_h = (pd.Timestamp.now(tz="UTC") - last["ts"]).total_seconds() / 3600
    s1, s2, s3 = st.columns(3)
    s1.metric("Equity", f"{last['equity']:,.2f}")
    s2.metric("Balance", f"{last['balance']:,.2f}")
    s3.metric("Last heartbeat", f"{age_h:.1f} h ago")
    if age_h > 3:
        st.warning("No heartbeat for over 3 hours: check the EA, VPS and WebRequest URL.")

# ---------- KPIs ----------
st.subheader(f"Closed trades, last {days} days")
if closed.empty:
    st.info("No closed trades in this window yet.")
else:
    pnl = closed["pnl"].astype(float)
    r = closed["r_multiple"].astype(float)
    wins, losses = pnl[pnl > 0], pnl[pnl < 0]
    pf = wins.sum() / abs(losses.sum()) if len(losses) else float("inf")
    cum = pnl.cumsum()
    max_dd = (cum - cum.cummax()).min()
    k = st.columns(6)
    k[0].metric("Net P/L", f"{pnl.sum():,.2f}")
    k[1].metric("Trades", len(closed))
    k[2].metric("Win rate", f"{(pnl > 0).mean():.0%}")
    k[3].metric("Expectancy", f"{r.mean():+.2f} R")
    k[4].metric("Profit factor", f"{pf:.2f}")
    k[5].metric("Max drawdown (closed)", f"{max_dd:,.2f}")

    left, right = st.columns(2)
    with left:
        fig = px.line(x=closed["closed_at"], y=cum, labels={"x": "", "y": "Cumulative P/L"},
                      title="Cumulative P/L (closed trades)")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        fig = px.line(x=closed["closed_at"], y=r.cumsum(), labels={"x": "", "y": "Cumulative R"},
                      title="Cumulative R")
        st.plotly_chart(fig, use_container_width=True)

    left, right = st.columns(2)
    with left:
        st.plotly_chart(px.histogram(r, nbins=30, title="R-multiple distribution",
                                     labels={"value": "R"}), use_container_width=True)
    with right:
        by_hour = closed.groupby("hour_open")["r_multiple"].agg(["mean", "count"]).reset_index()
        st.plotly_chart(px.bar(by_hour, x="hour_open", y="mean", hover_data=["count"],
                               title="Avg R by entry hour (broker time)"), use_container_width=True)

    left, right = st.columns(2)
    with left:
        closed["bias_bucket"] = pd.cut(closed["bias_strength"].astype(float), [0, 0.5, 0.65, 0.8, 1.0])
        bb = closed.groupby("bias_bucket", observed=True)["r_multiple"].agg(["mean", "count"]).reset_index()
        bb["bias_bucket"] = bb["bias_bucket"].astype(str)
        st.plotly_chart(px.bar(bb, x="bias_bucket", y="mean", hover_data=["count"],
                               title="Avg R by Asia bias strength"), use_container_width=True)
    with right:
        bd = closed.groupby("direction")["r_multiple"].agg(["mean", "count"]).reset_index()
        st.plotly_chart(px.bar(bd, x="direction", y="mean", hover_data=["count"],
                               title="Avg R by direction"), use_container_width=True)

if not snaps.empty:
    w = snaps[snaps["ts"] >= cutoff]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=w["ts"], y=w["equity"], name="Equity"))
    fig.add_trace(go.Scatter(x=w["ts"], y=w["balance"], name="Balance"))
    fig.update_layout(title="Account equity (hourly snapshots)")
    st.plotly_chart(fig, use_container_width=True)

# ---------- AI decision layer ----------
st.subheader("AI decision layer")
dec = to_dt(fetch("aa_ai_decisions", "created_at.desc",
                  "id,created_at,symbol,mode,model,action,sl_mult,tp3_r,confidence,reason,latency_ms"),
            ["created_at"])
sets = to_dt(fetch("aa_setups", "setup_time.desc"), ["setup_time"])
if sym and not dec.empty:
    dec = dec[dec["symbol"] == sym]
if sym and not sets.empty:
    sets = sets[sets["symbol"] == sym]
if not dec.empty:
    dec = dec[dec["created_at"] >= cutoff]

if dec.empty:
    st.info("No AI decisions logged yet. Run schema_ai.sql, then set InpAIMode=1 (shadow) in the EA.")
else:
    if sets.empty:
        m = dec.assign(r_rule=float("nan"), r_ai=float("nan"), taken=None)
    else:
        m = dec.merge(sets[["decision_id", "r_rule", "r_ai", "taken"]],
                      left_on="id", right_on="decision_id", how="left")
    for c in ("r_rule", "r_ai", "confidence", "latency_ms"):
        m[c] = pd.to_numeric(m[c], errors="coerce")
    done = m.dropna(subset=["r_rule"])

    q = st.columns(4)
    q[0].metric("AI calls", len(m))
    q[1].metric("TRADE share", f"{(m['action'] == 'TRADE').mean():.0%}")
    q[2].metric("Avg latency", f"{m['latency_ms'].mean() / 1000:.1f} s")
    q[3].metric("Outcomes resolved", len(done), help="Simulated ~24h after each call (InpMaxHoldBars)")

    if done.empty:
        st.caption("Outcomes appear about a day after each decision. Judge the AI only once you have 30+ per side.")
    else:
        left, right = st.columns(2)
        with left:
            # The core question: do SKIPs really avoid worse setups than TRADEs? (SKIP avg R should be lower)
            by_act = done.groupby("action")["r_rule"].agg(["mean", "count"]).reset_index()
            st.plotly_chart(px.bar(by_act, x="action", y="mean", hover_data=["count"],
                                   title="Avg simulated R (rule SL/TP) by AI call",
                                   labels={"mean": "avg R"}), use_container_width=True)
        with right:
            # Does the AI's own SL/TP beat the rule's SL/TP on the setups it said TRADE to?
            t = done[done["action"] == "TRADE"]
            cmp = pd.DataFrame({"who": ["Rule SL/TP", "AI SL/TP"],
                                "avg R": [t["r_rule"].mean(), t["r_ai"].mean()], "n": [len(t), len(t)]})
            st.plotly_chart(px.bar(cmp, x="who", y="avg R", hover_data=["n"],
                                   title="On TRADE calls: rule vs AI stop/target"), use_container_width=True)
        done = done.assign(conf_bucket=pd.cut(done["confidence"], [0, 0.4, 0.55, 0.7, 1.0]).astype(str))
        cb = done.groupby("conf_bucket")["r_rule"].agg(["mean", "count"]).reset_index()
        st.plotly_chart(px.bar(cb, x="conf_bucket", y="mean", hover_data=["count"],
                               title="Avg R by AI confidence (should rise left to right)"),
                        use_container_width=True)

    st.dataframe(m[["created_at", "action", "confidence", "sl_mult", "tp3_r", "r_rule", "r_ai", "taken", "reason"]]
                 .head(50), use_container_width=True, hide_index=True)

# ---------- tables ----------
st.subheader("Re-tune history")
if not adapts.empty:
    st.dataframe(adapts.drop(columns=["id", "magic"], errors="ignore"), use_container_width=True, hide_index=True)

st.subheader("Recent trades")
if not trades.empty:
    cols = ["opened_at", "direction", "lots", "entry_price", "sl_price", "pnl", "r_multiple", "closed",
            "sl_mult", "tp3_r", "adx_min", "adx", "bias_strength"]
    st.dataframe(trades.sort_values("opened_at", ascending=False)[[c for c in cols if c in trades]].head(100),
                 use_container_width=True, hide_index=True)
