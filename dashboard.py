"""
ApexAdaptive control center (Streamlit).

One EA instance trades every asset you switch on here. The EA reports the MT5 account it runs on (login, server,
balance, equity, initial capital) and this dashboard picks it up automatically.

Secrets (Streamlit Cloud: App settings > Secrets, or .streamlit/secrets.toml locally):
    SUPABASE_URL = "https://xxxx.supabase.co"
    SUPABASE_SERVICE_KEY = "..."
Keep the app PRIVATE: it uses the service-role key and can flatten your account. Never commit secrets.toml.
Run schema.sql, schema_ai.sql, schema_assets.sql, schema_risk.sql, then schema_v2.sql in Supabase first.
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
def _fetch(table, order=None, select="*"):
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


@st.cache_data(ttl=60)
def fetch(table, order=None, select="*"):          # history tables
    return _fetch(table, order, select)


@st.cache_data(ttl=8)
def fetch_live(table, order=None, select="*"):     # control + status tables: keep these fresh
    return _fetch(table, order, select)


def load(table, order=None, select="*", live=False):
    try:
        return (fetch_live if live else fetch)(table, order, select)
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


def nz(v):
    """pandas NaN/NA -> None so the value is valid JSON."""
    return None if v is None or (not isinstance(v, (list, dict)) and pd.isna(v)) else v


# ------------------------------------------------------------------ asset classes
CLASSES = ["Metals", "Forex", "Crypto", "Indices", "Energy", "Stocks", "Other"]
FX = {"USD", "EUR", "GBP", "JPY", "CHF", "AUD", "NZD", "CAD", "CNH", "MXN", "NOK", "SEK", "SGD", "ZAR", "HKD", "CZK", "PLN", "HUF", "TRY"}


def guess_class(sym, path=""):
    p = str(path or "").lower()
    for key, cls in (("metal", "Metals"), ("forex", "Forex"), ("crypto", "Crypto"), ("indic", "Indices"),
                     ("energ", "Energy"), ("stock", "Stocks"), ("share", "Stocks")):
        if key in p:
            return cls
    s = str(sym).upper()
    if s.startswith(("XAU", "XAG", "XPT", "XPD")):
        return "Metals"
    if s.startswith(("BTC", "ETH", "SOL", "XRP", "LTC", "ADA", "DOGE", "BNB", "DOT", "LINK")):
        return "Crypto"
    if any(k in s for k in ("US30", "US100", "US500", "NAS", "SPX", "DJ30", "GER", "DAX", "UK100", "JP225", "NDX", "AUS200", "EU50", "FRA40")):
        return "Indices"
    if any(k in s for k in ("OIL", "WTI", "BRENT", "NGAS", "UKOIL", "USOIL")):
        return "Energy"
    if len(s) >= 6 and s[:3] in FX and s[3:6] in FX:
        return "Forex"
    return "Other"


PLANS = {   # FTMO presets (verify against your own plan: firms revise their rules)
    "FTMO 2-Step: 5% daily, 10% static max loss": dict(daily_loss_pct=5, max_loss_pct=10, dd_type="static"),
    "FTMO 1-Step: 3% daily, 10% trailing max loss": dict(daily_loss_pct=3, max_loss_pct=10, dd_type="trailing"),
}


# ------------------------------------------------------------------ load everything
T_COLS = ["login", "symbol", "magic", "direction", "opened_at", "closed_at", "last_deal", "closed", "lots", "entry_price",
          "sl_price", "risk_money", "pnl", "r_multiple", "sl_mult", "tp3_r", "adx_min", "adx", "bias_strength",
          "hour_open"]
A_COLS = ["login", "symbol", "ts", "enabled", "sl_mult", "tp3_r", "adx_min", "train_n", "train_avg_r", "valid_n",
          "valid_avg_r", "note"]
S_COLS = ["login", "ts", "balance", "equity", "open_positions"]
R_COLS = ["login", "symbol", "asset_class", "enabled", "risk_pct", "max_spread_pts", "sim_cost_pts", "asia_start_hour",
          "asia_end_hour", "trade_start_hour", "trade_end_hour", "force_close_hour", "notes"]
D_COLS = ["id", "login", "created_at", "symbol", "mode", "model", "action", "sl_mult", "tp3_r", "confidence", "reason",
          "latency_ms"]
U_COLS = ["decision_id", "login", "symbol", "setup_time", "taken", "r_rule", "r_ai"]
ACC_COLS = ["login", "label", "server", "company", "currency", "leverage", "trade_mode", "is_ftmo", "kind",
            "initial_balance", "balance", "equity", "day_start_balance", "peak_eod_balance", "last_seen", "halted",
            "kill_switch", "flatten_token", "ai_mode", "account_type", "daily_loss_pct", "max_loss_pct", "dd_type", "daily_buffer_pct",
            "loss_buffer_pct", "max_open_risk_pct", "max_positions", "profit_target_pct", "account_size_override"]
ST_COLS = ["login", "symbol", "ts", "state", "open_positions", "spread_pts", "sl_mult", "tp3_r", "adx_min", "last_adapt"]
HOUR_COLS = ["asia_start_hour", "asia_end_hour", "trade_start_hour", "trade_end_hour", "force_close_hour"]

accounts = to_dt(num(ensure(load("aa_accounts", "login.asc", live=True), ACC_COLS),
                     ["login", "initial_balance", "balance", "equity", "day_start_balance", "peak_eod_balance",
                      "flatten_token", "ai_mode", "daily_loss_pct", "max_loss_pct", "daily_buffer_pct",
                      "loss_buffer_pct", "max_open_risk_pct", "max_positions", "profit_target_pct",
                      "account_size_override"]), ["last_seen"])
status_all = to_dt(num(ensure(load("aa_asset_status", "symbol.asc", live=True), ST_COLS),
                       ["login", "open_positions", "spread_pts", "sl_mult", "tp3_r", "adx_min"]), ["ts", "last_adapt"])
assets_all = num(ensure(load("aa_assets", "symbol.asc", live=True), R_COLS),
                 ["login", "risk_pct", "max_spread_pts", "sim_cost_pts"] + HOUR_COLS)
broker = ensure(load("aa_broker_symbols", "symbol.asc", live=True), ["server", "symbol", "path", "description"])
trades_all = to_dt(num(ensure(load("aa_trade_results", "opened_at.asc"), T_COLS),
                       ["login", "pnl", "r_multiple", "risk_money", "lots", "bias_strength", "hour_open"]),
                   ["opened_at", "closed_at", "last_deal"])
adapts_all = to_dt(num(ensure(load("aa_adaptations", "ts.desc"), A_COLS),
                       ["login", "sl_mult", "tp3_r", "adx_min", "valid_avg_r"]), ["ts"])
snaps_all = to_dt(num(ensure(load("aa_snapshots", "ts.asc"), S_COLS), ["login", "balance", "equity", "open_positions"]), ["ts"])
dec_all = to_dt(num(ensure(load("aa_ai_decisions", "created_at.desc", ",".join(D_COLS)), D_COLS),
                    ["login", "confidence", "latency_ms", "sl_mult", "tp3_r"]), ["created_at"])
sets_all = to_dt(num(ensure(load("aa_setups", "setup_time.desc"), U_COLS), ["login", "r_rule", "r_ai"]), ["setup_time"])

# ------------------------------------------------------------------ account picker (auto-detected from the EA)
st.sidebar.title("ApexAdaptive")
opts = [int(x) for x in accounts["login"].dropna()]
if (trades_all["login"] == 0).any():
    opts.append(0)
if not opts:
    st.title("ApexAdaptive control center")
    if MISSING:
        st.warning("Missing tables: " + ", ".join(sorted(set(MISSING))) + ". Run the schema files, ending with schema_v2.sql.")
    st.info("No MT5 account has connected yet. Attach the EA to ONE chart (any symbol) with your Modal URL and key, "
            "allow the URL under Tools > Options > Expert Advisors > Allow WebRequest, and the account appears here "
            "within a minute, with no account number to type in.")
    st.stop()


def acct_label(lg):
    if lg == 0:
        return "Legacy data (before v2)"
    r = accounts[accounts["login"] == lg].iloc[0]
    tag = str(r["label"]) if pd.notna(r["label"]) and r["label"] else str(r["kind"] or "").upper()
    return f"{tag} · {lg} · {r['server']}"


SEL = st.sidebar.selectbox("MT5 account", opts, format_func=acct_label)
A = accounts[accounts["login"] == SEL].iloc[0] if SEL != 0 else None


def av(col, default=0.0):
    if A is None or pd.isna(A[col]):
        return default
    return A[col]


def mine(df):
    return df[df["login"] == SEL].copy() if "login" in df else df.copy()


trades, adapts, snaps = mine(trades_all), mine(adapts_all), mine(snaps_all)
dec, sets, assets, status = mine(dec_all), mine(sets_all), mine(assets_all), mine(status_all)

now = pd.Timestamp.now(tz="UTC")
size = float(av("account_size_override")) or float(av("initial_balance"))
daily_pct, maxloss_pct = float(av("daily_loss_pct", 5)), float(av("max_loss_pct", 10))
trailing = (av("dd_type", "static") == "trailing")
eq_now = float(A["equity"]) if A is not None and pd.notna(A["equity"]) else None
bal_now = float(A["balance"]) if A is not None and pd.notna(A["balance"]) else None
last_hb = A["last_seen"] if A is not None else None
hb_age_s = (now - last_hb).total_seconds() if last_hb is not None and pd.notna(last_hb) else None
online = hb_age_s is not None and hb_age_s < 120

reg_class = dict(zip(assets["symbol"], assets["asset_class"]))
reg_risk = dict(zip(assets["symbol"], assets["risk_pct"]))
all_syms = sorted(set(trades["symbol"].dropna()) | set(status["symbol"].dropna()) |
                  set(adapts["symbol"].dropna()) | set(reg_class))


def klass(sym):
    return reg_class.get(sym) or guess_class(sym)


# ------------------------------------------------------------------ sidebar
days = st.sidebar.slider("Window (days)", 7, 365, 90)
avail_classes = sorted({klass(s) for s in all_syms})
sel_classes = st.sidebar.multiselect("Asset classes", avail_classes, default=avail_classes)
class_syms = [s for s in all_syms if klass(s) in sel_classes]
sel_syms = st.sidebar.multiselect("Assets", class_syms, default=class_syms)
if A is not None:
    st.sidebar.caption(f"{A['kind'] and str(A['kind']).upper()} · {A['trade_mode']} · {A['currency']} · 1:{int(A['leverage']) if pd.notna(A['leverage']) else '?'}")
if st.sidebar.button("Refresh now"):
    st.cache_data.clear()
    st.rerun()
cutoff = now - pd.Timedelta(days=days)

# ------------------------------------------------------------------ derived data
tr = trades[trades["symbol"].isin(sel_syms)].copy()
tr["closed_at"] = tr["closed_at"].fillna(tr["last_deal"])
closed = tr[(tr["closed"] == True) & (tr["closed_at"] >= cutoff)].sort_values("closed_at").copy()  # noqa: E712
closed["pnl"] = closed["pnl"].fillna(0.0)
closed["class"] = closed["symbol"].map(klass)
closed["day"] = closed["closed_at"].dt.date
closed["dow"] = closed["opened_at"].dt.dayofweek
open_tr = tr[(tr["closed"] != True) & (tr["opened_at"] >= now - pd.Timedelta(days=7))].copy()  # noqa: E712
open_tr["class"] = open_tr["symbol"].map(klass)

acct = pd.DataFrame(columns=["equity", "balance"], index=pd.DatetimeIndex([], tz="UTC"))
if snaps["ts"].notna().any():
    acct = (snaps.dropna(subset=["ts"]).set_index("ts")[["equity", "balance"]].sort_index()
            .resample("1h").last().dropna())

# FTMO-style limits: loss amounts are a % of INITIAL capital, measured from the midnight balance (daily)
dsb = float(av("day_start_balance", bal_now or 0.0)) or (bal_now or 0.0)
daily_allowed = daily_pct / 100 * size
daily_used = max(0.0, dsb - eq_now) if eq_now is not None else None
peak_eod = float(av("peak_eod_balance", size)) or size
dd_base = max(size, peak_eod) if trailing else size
maxloss_allowed = maxloss_pct / 100 * size
maxloss_used = max(0.0, dd_base - eq_now) if eq_now is not None else None
daily_floor = dsb - daily_allowed
maxloss_floor = dd_base - maxloss_allowed
target_pct = float(av("profit_target_pct"))

last_st = status.sort_values("ts").groupby("symbol").tail(1).set_index("symbol") if not status.empty else status
last_ad = adapts.sort_values("ts").groupby("symbol").tail(1).set_index("symbol") if not adapts.empty else adapts



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


def gauge(label, used, allowed, floor, ccy=""):
    if used is None or not allowed:
        st.caption(f"{label}: no data yet")
        return
    frac = min(max(used / allowed, 0.0), 1.0)
    st.progress(frac, text=f"{label}: {used:,.2f} of {allowed:,.2f} {ccy} used ({frac:.0%}) · floor {floor:,.2f}")
    if frac >= 0.7:
        st.warning(f"{label} is above 70% of the limit.")



# ------------------------------------------------------------------ page
st.title("ApexAdaptive control center")
if MISSING:
    st.warning("Missing tables: " + ", ".join(sorted(set(MISSING))) +
               ". Run schema.sql, schema_ai.sql, schema_assets.sql, schema_risk.sql and schema_v2.sql in the Supabase SQL editor.")
if A is not None:
    h = st.columns([1.2, 1, 1, 1, 1.4])
    h[0].metric("EA", "ONLINE" if online else "OFFLINE",
                help="The EA syncs every ~30 s. Offline on weekends is normal only if the terminal is closed.")
    h[1].metric("Account", f"{SEL}")
    h[2].metric("Equity", money(eq_now))
    h[3].metric("Balance", money(bal_now))
    halted = str(A["halted"] or "")
    h[4].metric("EA state", {"": "Trading allowed", "daily": "HALTED: daily loss", "maxloss": "HALTED: max loss",
                             "target": "Profit target reached"}.get(halted, halted) +
                (" · KILL SWITCH" if bool(A["kill_switch"]) else ""))
    if not online:
        st.error("The EA has not synced for "
                 + ("a long time" if hb_age_s is None else f"{hb_age_s / 60:.0f} min")
                 + ". Check the terminal/VPS, AutoTrading and the WebRequest URL. While it is offline the EA keeps "
                   "its last known limits and blocks new entries after the configured config-age timeout.")
    if halted in ("daily", "maxloss"):
        st.error("The EA closed everything and halted itself to protect the account. Daily halts clear at the next "
                 "server-day rollover; a max-loss halt stays until you change the limits or the account.")
else:
    st.info("Legacy data from before v2 (read-only). Pick a connected account in the sidebar to use the control center.")

t_ctl, t_over, t_assets, t_perf, t_risk, t_ai, t_log = st.tabs(
    ["Control center", "Overview", "Assets", "Performance", "Risk", "AI layer", "Trades & tuning"])

kp = kpis(closed, size)

# ============================== Control center
with t_ctl:
    if A is None:
        st.info("Select a live account in the sidebar.")
    else:
        def patch_account(**kw):
            try:
                sb_write("PATCH", "aa_accounts", params={"login": f"eq.{SEL}"}, json=kw, prefer="return=minimal")
                st.cache_data.clear()
                return True
            except Exception as e:  # noqa: BLE001
                st.error(f"Save failed: {e}")
                return False

        st.caption("Changes reach the EA at its next sync (within about 30 seconds). Open positions are always managed, "
                   "even for paused assets.")

        # ---- panic controls
        st.subheader("Safety controls")
        c1, c2, c3 = st.columns([1.3, 1.3, 1.4])
        with c1:
            kill_cur = bool(A["kill_switch"])
            kill_new = st.toggle("Kill switch: block ALL new entries", value=kill_cur, key=f"kill_{SEL}")
            if kill_new != kill_cur and patch_account(kill_switch=kill_new):
                st.rerun()
            st.caption("Open trades keep being managed (breakeven, targets, runner).")
        with c2:
            ok_flat = st.checkbox("I want to close every open position on this account", key=f"cf_{SEL}")
            if st.button("Flatten all positions now", type="primary", disabled=not ok_flat, key=f"flat_{SEL}"):
                if patch_account(flatten_token=int(av("flatten_token")) + 1):
                    st.toast("Flatten command sent. The EA executes it at its next sync.")
                    st.rerun()
        with c3:
            modes = ["Off (rules only)", "Shadow (AI advises, rules trade)", "Filter (AI may veto / tune SL & TP)"]
            cur_m = int(av("ai_mode"))
            new_m = st.selectbox("AI decision layer", modes, index=min(max(cur_m, 0), 2), key=f"ai_{SEL}")
            if modes.index(new_m) != cur_m and patch_account(ai_mode=modes.index(new_m)):
                st.rerun()

        # ---- account type: the only thing that can ever allow the EA to hedge
        types = ["Prop firm account (hedging never allowed)", "Live account, own capital (a clean reversal may be hedged once only the runner is left)"]
        cur_t = 1 if str(av("account_type")) == "live" else 0
        new_t = st.selectbox("Account type", types, index=cur_t, key=f"atype_{SEL}")
        if types.index(new_t) != cur_t and patch_account(account_type="live" if types.index(new_t) == 1 else "prop"):
            st.rerun()
        st.caption("Only FTMO is detected automatically. Every other prop firm account must stay on Prop. "
                   "A hedge also needs a hedging-type MT5 account and the EA's InpHedge switch on.")

        # ---- assets
        st.subheader("Assets this EA trades")
        if "aa_assets" in MISSING or "login" not in assets_all.columns:
            st.warning("Run schema_v2.sql first.")
        else:
            stt = status.set_index("symbol")["state"].to_dict() if not status.empty else {}
            cols = ["symbol", "enabled", "risk_pct", "asset_class"] + ["max_spread_pts", "sim_cost_pts"] + HOUR_COLS
            view = assets[cols].copy().reset_index(drop=True)
            view["enabled"] = view["enabled"].fillna(True).astype(bool)
            view.insert(2, "live_state", view["symbol"].map(stt).fillna("(not reported yet)"))
            if view.empty:
                st.info("No assets yet. Add one below. New assets start paused until you enable them.")
            else:
                b1, b2, _ = st.columns([1, 1, 6])
                if b1.button("Enable all", key=f"ena_{SEL}"):
                    sb_write("PATCH", "aa_assets", params={"login": f"eq.{SEL}"}, json={"enabled": True}, prefer="return=minimal")
                    st.cache_data.clear(); st.rerun()
                if b2.button("Pause all", key=f"dis_{SEL}"):
                    sb_write("PATCH", "aa_assets", params={"login": f"eq.{SEL}"}, json={"enabled": False}, prefer="return=minimal")
                    st.cache_data.clear(); st.rerun()
                edited = st.data_editor(
                    view, key=f"editor_{SEL}", hide_index=True, width="stretch",
                    disabled=["symbol", "live_state"],
                    column_config={
                        "symbol": st.column_config.TextColumn("Asset"),
                        "enabled": st.column_config.CheckboxColumn("Trade", help="Untick to pause new entries."),
                        "live_state": st.column_config.TextColumn("Live state"),
                        "risk_pct": st.column_config.NumberColumn("Risk % / trade", min_value=0.0, max_value=5.0, step=0.05,
                                                                  help="0 = EA default. The EA also enforces its own hard ceiling."),
                        "asset_class": st.column_config.SelectboxColumn("Class", options=CLASSES),
                        "max_spread_pts": st.column_config.NumberColumn("Max spread (pts)", min_value=0.0,
                                                                        help="Empty = default gate (spread <= 15% of ATR)."),
                        "sim_cost_pts": st.column_config.NumberColumn("Sim cost (pts)", min_value=0.0,
                                                                     help="Spread+commission used by the tuner. Empty = 1.5x live spread."),
                        "asia_start_hour": st.column_config.NumberColumn("Asia start", min_value=0, max_value=23, step=1),
                        "asia_end_hour": st.column_config.NumberColumn("Asia end", min_value=0, max_value=23, step=1),
                        "trade_start_hour": st.column_config.NumberColumn("Trade start", min_value=0, max_value=23, step=1),
                        "trade_end_hour": st.column_config.NumberColumn("Trade end", min_value=0, max_value=23, step=1),
                        "force_close_hour": st.column_config.NumberColumn("Force close", min_value=0, max_value=23, step=1),
                    })
                st.caption("Session hours are BROKER SERVER hours; empty cells use the EA's defaults. The Asia-bias logic assumes "
                           "a quiet Asia session, so check the hours for 24/7 or non-gold markets before enabling them.")
                if st.button("Save asset changes", type="primary", key=f"saveassets_{SEL}"):
                    rows = []
                    for _, r in edited.iterrows():
                        row = {"login": int(SEL), "symbol": r["symbol"], "enabled": bool(r["enabled"]),
                               "risk_pct": nz(r["risk_pct"]), "asset_class": r["asset_class"] or "Other",
                               "max_spread_pts": nz(r["max_spread_pts"]), "sim_cost_pts": nz(r["sim_cost_pts"])}
                        for hc in HOUR_COLS:
                            v = nz(r[hc])
                            row[hc] = None if v is None else int(v)
                        rows.append(row)
                    try:
                        sb_write("POST", "aa_assets", params={"on_conflict": "login,symbol"}, json=rows,
                                 prefer="resolution=merge-duplicates,return=minimal")
                        st.cache_data.clear()
                        st.toast("Saved. The EA picks it up within ~30 s.")
                        st.rerun()
                    except Exception as e:  # noqa: BLE001
                        st.error(f"Save failed: {e}")

            with st.expander("Add an asset", expanded=view.empty):
                bs = broker[broker["server"] == A["server"]].copy()
                have = set(assets["symbol"])
                bs = bs[~bs["symbol"].isin(have)]
                a1, a2, a3 = st.columns([2, 1, 1])
                if bs.empty:
                    pick = a1.text_input("Broker symbol, exactly as in MT5 (the EA has not uploaded the symbol list yet)",
                                         key=f"addsym_{SEL}").strip()
                    path = ""
                else:
                    bs["label"] = bs["symbol"] + "  ·  " + bs["description"].fillna("") + "  ·  " + bs["path"].fillna("")
                    lab = a1.selectbox("Broker symbol (type to search)", [""] + list(bs["label"]), key=f"addpick_{SEL}")
                    pick = lab.split("  ·  ")[0] if lab else ""
                    path = bs.loc[bs["symbol"] == pick, "path"].iloc[0] if pick else ""
                risk_in = a2.number_input("Risk % / trade", 0.0, 5.0, 0.5, 0.05, key=f"addrisk_{SEL}")
                on_in = a3.checkbox("Enable immediately", value=False, key=f"addon_{SEL}")
                if st.button("Add asset", key=f"addbtn_{SEL}"):
                    if not pick:
                        st.error("Pick or type a symbol.")
                    else:
                        try:
                            sb_write("POST", "aa_assets", params={"on_conflict": "login,symbol"},
                                     json={"login": int(SEL), "symbol": pick, "asset_class": guess_class(pick, path),
                                           "enabled": on_in, "risk_pct": risk_in},
                                     prefer="resolution=merge-duplicates,return=minimal")
                            st.cache_data.clear(); st.rerun()
                        except Exception as e:  # noqa: BLE001
                            st.error(f"Add failed: {e}")
            with st.expander("Copy / remove"):
                others = [int(x) for x in accounts["login"] if int(x) != SEL]
                if others:
                    src = st.selectbox("Copy the asset list from another account", others, format_func=acct_label,
                                       key=f"cpsrc_{SEL}")
                    as_off = st.checkbox("Copy as paused (recommended)", value=True, key=f"cpoff_{SEL}")
                    if st.button("Copy assets", key=f"cpbtn_{SEL}"):
                        src_rows = assets_all[assets_all["login"] == src]
                        out = []
                        for _, r in src_rows.iterrows():
                            d = {k: nz(r[k]) for k in R_COLS if k != "login"}
                            for hc in HOUR_COLS:
                                d[hc] = None if d[hc] is None else int(d[hc])
                            d.update(login=int(SEL), enabled=False if as_off else bool(d["enabled"]))
                            out.append(d)
                        if out:
                            sb_write("POST", "aa_assets", params={"on_conflict": "login,symbol"}, json=out,
                                     prefer="resolution=merge-duplicates,return=minimal")
                            st.cache_data.clear(); st.rerun()
                if not assets.empty:
                    rm = st.selectbox("Remove an asset from this account", [""] + list(assets["symbol"]), key=f"rm_{SEL}")
                    if rm and st.button(f"Remove {rm}", key=f"rmbtn_{SEL}"):
                        sb_write("DELETE", "aa_assets", params={"login": f"eq.{SEL}", "symbol": f"eq.{rm}"}, prefer="return=minimal")
                        st.cache_data.clear(); st.rerun()
                    st.caption("Removing stops new entries; the EA still manages any open position. History stays in the database.")

        # ---- account rules
        st.subheader("Account rules")
        st.caption(f"Initial capital (auto-detected from the terminal): {money(av('initial_balance', None))} {A['currency']}"
                   + ("  ·  FTMO server detected" if bool(A["is_ftmo"]) else ""))
        p1, p2 = st.columns(2)
        for col, (name, vals) in zip((p1, p2), PLANS.items()):
            if col.button(f"Apply preset: {name}", key=f"plan_{name}_{SEL}"):
                if patch_account(**vals):
                    st.rerun()
        with st.form(f"rules_{SEL}"):
            f1, f2, f3 = st.columns(3)
            lab_in = f1.text_input("Nickname", value=str(A["label"] or ""))
            d_in = f2.number_input("Daily loss limit (% of initial)", 0.1, 20.0, float(daily_pct), 0.5)
            m_in = f3.number_input("Max loss limit (% of initial)", 0.1, 50.0, float(maxloss_pct), 0.5)
            g1, g2, g3 = st.columns(3)
            dt_in = g1.selectbox("Max loss type", ["static", "trailing"], index=1 if trailing else 0,
                                 help="FTMO 2-Step: static from initial capital. 1-Step: trails the highest end-of-day balance.")
            db_in = g2.number_input("Halt this many % BEFORE the daily limit", 0.0, 5.0, float(av("daily_buffer_pct", 1.0)), 0.1)
            lb_in = g3.number_input("Halt this many % BEFORE the max-loss limit", 0.0, 10.0, float(av("loss_buffer_pct", 1.5)), 0.1)
            k1, k2, k3, k4 = st.columns(4)
            mo_in = k1.number_input("Max combined open risk (% equity, 0 = off)", 0.0, 20.0, float(av("max_open_risk_pct")), 0.25)
            mp_in = k2.number_input("Max open positions (0 = off)", 0, 50, int(av("max_positions")), 1)
            pt_in = k3.number_input("Profit target % (stop new entries, 0 = off)", 0.0, 100.0, float(target_pct), 0.5)
            ov_in = k4.number_input("Account size override (0 = auto)", 0.0, 100000000.0, float(av("account_size_override")), 1000.0)
            if st.form_submit_button("Save rules", type="primary"):
                if patch_account(label=lab_in or None, daily_loss_pct=d_in, max_loss_pct=m_in, dd_type=dt_in,
                                 daily_buffer_pct=db_in, loss_buffer_pct=lb_in, max_open_risk_pct=mo_in,
                                 max_positions=int(mp_in), profit_target_pct=pt_in, account_size_override=ov_in):
                    st.toast("Rules saved")
                    st.rerun()
        st.caption("FTMO measures the daily limit on EQUITY (floating P/L included) from the balance at midnight CE(S)T, and both "
                   "loss amounts as a % of the INITIAL capital. The EA halts earlier, by the buffers above. Firms revise their "
                   "rules, so verify the numbers against your own FTMO plan.")

# ============================== Overview
with t_over:
    c = st.columns(6)
    c[0].metric("Equity", money(eq_now))
    c[1].metric("Balance", money(bal_now))
    day_pl = None if eq_now is None else eq_now - dsb
    c[2].metric("Day P/L", "–" if day_pl is None else f"{day_pl:+,.2f}",
                delta=None if day_pl is None or not size else f"{day_pl / size * 100:+.2f}% of initial",
                help="Equity vs the balance at server midnight.")
    open_pos = int(last_st["open_positions"].fillna(0).sum()) if not last_st.empty else 0
    c[3].metric("Open positions", open_pos)
    c[4].metric("Initial risk open", money(float(open_tr["risk_money"].fillna(0).sum())),
                help="Sum of the 1R money risk at entry of trades not yet closed. Moves to ~0 once breakeven is set.")
    c[5].metric("Net P/L (window)", money(kp["net"]))
    if target_pct > 0 and size > 0 and bal_now is not None:
        prog = (bal_now - size) / (size * target_pct / 100)
        st.progress(min(max(prog, 0.0), 1.0), text=f"Profit target {target_pct:g}%: {(bal_now - size) / size * 100:+.2f}% "
                                                  f"({prog:.0%} of the way)")
    all_closed = trades[(trades["closed"] == True)]  # noqa: E712
    days_traded = all_closed["closed_at"].fillna(all_closed["last_deal"]).dt.date.nunique()
    st.caption(f"Distinct trading days with a closed trade (all time): {days_traded}. Check your plan's minimum trading days.")

    st.subheader(f"Closed trades, last {days} days")
    if kp["n"] < 30:
        st.caption("Fewer than 30 closed trades: treat every ratio below as noise.")
    k = st.columns(6)
    k[0].metric("Trades", kp["n"])
    k[1].metric("Win rate", pct(kp["win_rate"]))
    k[2].metric("Profit factor", ratio(kp["pf"]))
    k[3].metric("Expectancy", "–" if pd.isna(kp["exp_r"] or np.nan) else f"{kp['exp_r']:+.2f} R")
    k[4].metric("Payoff (avg win / avg loss)", ratio(kp["payoff"]))
    k[5].metric("Max drawdown (closed)", money(kp["max_dd"]), help="Peak to trough of the realised P/L curve in this window.")

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
        sn = last_st.loc[s] if s in last_st.index else None
        ad = last_ad.loc[s] if s in last_ad.index else None
        state = "Not reported yet" if sn is None or pd.isna(sn["state"]) else sn["state"]
        if sn is not None and not online and state not in ("PAUSED",):
            state += " (EA offline)"
        rows.append({
            "Asset": s, "Class": klass(s), "State": state,
            "Params in force": "–" if sn is None or pd.isna(sn["sl_mult"]) or state.startswith(("PAUSED", "STAND", "TUNING")) else
            f"SL {sn['sl_mult']}xATR / TP2 {sn['tp3_r']}R / ADX>={sn['adx_min']}",
            "Risk %": reg_risk.get(s), "Spread (pts)": None if sn is None else sn["spread_pts"],
            "Open": None if sn is None else sn["open_positions"],
            "Last re-tune": None if ad is None else ad["ts"].strftime("%Y-%m-%d"),
            "Trades": k_["n"], "Net P/L": k_["net"],
            "Win %": None if k_["win_rate"] is None else k_["win_rate"] * 100,
            "Profit factor": k_["pf"], "Expectancy (R)": k_["exp_r"], "SQN": k_["sqn"], "Max DD": k_["max_dd"]})
    if rows:
        table(pd.DataFrame(rows).round(2))
        st.caption("State is reported live by the EA. STAND DOWN means the latest walk-forward re-tune found no parameter "
                   "set that validated for that asset: the built-in guard against a strategy that doesn't fit a market.")
    else:
        st.info("No assets yet. Add one in the Control center tab.")

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
    st.subheader("Prop-firm limits (FTMO method)")
    if A is None or size <= 0:
        st.info("Waiting for the EA to report the account's initial capital.")
    else:
        g1, g2 = st.columns(2)
        with g1:
            gauge("Daily loss used", daily_used, daily_allowed, daily_floor, A["currency"])
        with g2:
            gauge(f"Max loss used ({'trailing' if trailing else 'static'})", maxloss_used, maxloss_allowed, maxloss_floor, A["currency"])
        d_buf, l_buf = float(av("daily_buffer_pct", 1.0)), float(av("loss_buffer_pct", 1.5))
        lines = pd.DataFrame([
            {"": "Equity now", "Level": eq_now, "Headroom to firm floor": None},
            {"": "Daily loss floor (firm)", "Level": daily_floor, "Headroom to firm floor": None if eq_now is None else eq_now - daily_floor},
            {"": "Daily halt line (EA)", "Level": daily_floor + d_buf / 100 * size, "Headroom to firm floor": None if eq_now is None else eq_now - (daily_floor + d_buf / 100 * size)},
            {"": "Max loss floor (firm)", "Level": maxloss_floor, "Headroom to firm floor": None if eq_now is None else eq_now - maxloss_floor},
            {"": "Max loss halt line (EA)", "Level": maxloss_floor + l_buf / 100 * size, "Headroom to firm floor": None if eq_now is None else eq_now - (maxloss_floor + l_buf / 100 * size)},
        ])
        table(lines.round(2))
        st.caption(f"Initial capital {money(size)} · day-start (midnight) balance {money(dsb)} · "
                   + (f"highest end-of-day balance {money(peak_eod)} · " if trailing else "") +
                   "The 'headroom' column for the halt rows shows how far equity is above the EA's own earlier halt line. "
                   "Figures come from the EA's own terminal reading, so they match the firm's up to broker-time rounding.")

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
    st.caption("All assets share one account. The EA enforces the combined open-risk cap and position cap from the Control "
               "center, and refuses any entry whose full stop-out would cross its halt line.")


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

