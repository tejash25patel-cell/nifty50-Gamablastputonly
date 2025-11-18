# gamma_blast_pro_dashboard.py
import streamlit as st
from streamlit_autorefresh import st_autorefresh
import requests, math, time
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta, time as dtime
import statistics
import altair as alt

# -------------------------
# Page config
# -------------------------
st.set_page_config(page_title="Gamma Blast Pro Dashboard", layout="wide")
st.title("⚡ Gamma Blast Pro — Unified Market Action Score + Detector Feed")

# -------------------------
# Sidebar controls
# -------------------------
index_choice = st.sidebar.selectbox("Index", ["NIFTY", "BANKNIFTY", "SENSEX"])
provider = st.sidebar.selectbox("Provider", ["NSE (option-chain)", "yfinance (fallback)"])
poll_seconds = st.sidebar.slider("Refresh interval (seconds)", 15, 120, 60)
num_strikes = st.sidebar.number_input("ATM ± how many strikes (n)", min_value=1, max_value=6, value=2, step=1)
strike_gap_map = {"NIFTY":50, "BANKNIFTY":100, "SENSEX":100}
strike_gap = st.sidebar.number_input("Strike gap (override)", value=strike_gap_map.get(index_choice,50), step=50)
score_window = st.sidebar.number_input("Score window (samples)", min_value=10, max_value=500, value=120, step=10)
percentile_for_threshold = st.sidebar.slider("Threshold percentile", 80, 99, 95)
threshold_mode = st.sidebar.selectbox("Threshold mode", ["Percentile", "Mean+K*Std"])
k_std = st.sidebar.slider("K for Mean+K*Std",  value=2.0, min_value=0.5, max_value=6.0, step=0.5)
display_oi_breakdown = st.sidebar.checkbox("Show ATM±n OI/IV breakdown", value=True)
history_max = st.sidebar.number_input("Max saved alerts (session)", 50, 5000, 100, step=50)

# weights for detectors in composite score (sum to 1 ideally)
weights = {
    "gamma_blast": 0.25,
    "iv_shock": 0.15,
    "put_panic": 0.10,
    "call_short": 0.10,
    "oi_pressure": 0.10,
    "iv_oi_explosion": 0.10,
    "max_pain_shift": 0.05,
    "oi_wall_break": 0.05,
}

# -------------------------
# Session init
# -------------------------
if "history" not in st.session_state:
    st.session_state.history = pd.DataFrame(columns=[
        "Timestamp","Index","Spot","ATM","Strike","UnifiedScore","Regime","TopSignals","Details"
    ])
if "running" not in st.session_state:
    st.session_state.running = False
if "prev_oi" not in st.session_state:
    st.session_state.prev_oi = {}
if "prev_iv" not in st.session_state:
    st.session_state.prev_iv = {}
if "recent_scores" not in st.session_state:
    st.session_state.recent_scores = []
if "chart_history" not in st.session_state:
    st.session_state.chart_history = []
if "last_open" not in st.session_state:
    st.session_state.last_open = {}
if "past_max_pain" not in st.session_state:
    st.session_state.past_max_pain = None

# -------------------------
# Helpers: fetchers
# -------------------------
def fetch_nse_option_chain(symbol):
    base = "https://www.nseindia.com"
    api = f"{base}/api/option-chain-indices?symbol={symbol}"
    headers = {
        "User-Agent":"Mozilla/5.0",
        "Accept":"application/json, text/javascript, */*; q=0.01",
        "Referer": base
    }
    s = requests.Session()
    s.get(base, headers=headers, timeout=5)
    r = s.get(api, headers=headers, timeout=10)
    r.raise_for_status()
    return r.json()

def fetch_spot_yf(name):
    tick = {"NIFTY":"^NSEI","BANKNIFTY":"^NSEBANK","SENSEX":"^BSESN"}[name]
    t = yf.Ticker(tick)
    try:
        hist = t.history(period="1d", interval="1m")
        if hist is not None and not hist.empty:
            return float(hist["Close"].iloc[-1])
    except Exception:
        pass
    hist = t.history(period="2d", interval="1d")
    if hist is not None and not hist.empty:
        return float(hist["Close"].iloc[-1])
    return None

def get_open_price(name, oc=None):
    # prefer oc if it contains an open-like field
    if oc is not None:
        try:
            rec = oc.get("records", {})
            for k in ("open", "underlyingOpen", "underlyingOpenValue"):
                if rec.get(k):
                    return float(rec.get(k))
        except Exception:
            pass
    # fallback yfinance
    if name in st.session_state.last_open:
        return st.session_state.last_open[name]
    tick = {"NIFTY":"^NSEI","BANKNIFTY":"^NSEBANK","SENSEX":"^BSESN"}[name]
    t = yf.Ticker(tick)
    try:
        hist = t.history(period="2d", interval="1d")
        if hist is not None and not hist.empty:
            openp = float(hist["Open"].iloc[-1])
            st.session_state.last_open[name] = openp
            return openp
    except Exception:
        pass
    return None

# -------------------------
# Detector routines (heuristic implementations)
# Each returns a 0..1 score and a short message + raw values
# -------------------------
def detect_gamma_blast(oc, spot, strike_gap, n, gamma_threshold=0.0008, iv_threshold=0.20):
    """Look for gamma blast like behavior (big call OI % rise near ATM, low IV, near expiry)"""
    rec = oc.get("records",{})
    data = rec.get("data",[])
    total_oi = 0
    for e in data:
        for side in ("CE","PE"):
            v = e.get(side)
            if v and v.get("openInterest") is not None:
                try: total_oi += float(v.get("openInterest"))
                except: pass
    atm = int(round(spot/strike_gap)*strike_gap)
    strikes = [atm + i*strike_gap for i in range(-n,n+1)]
    # sum call OI now & prev
    sum_call_now = 0
    sum_call_prev = 0
    call_iv_atm = None
    for e in data:
        try:
            sp = int(e.get("strikePrice", -1))
        except Exception:
            continue
        if sp in strikes:
            ce = e.get("CE")
            if ce:
                oi_now = float(ce.get("openInterest") or 0)
                sum_call_now += oi_now
                # prev read
                prev = None
                for key in ("previousOpenInterest","prevOpenInterest","prevOI","previousOI"):
                    if key in ce and ce.get(key) is not None:
                        try:
                            prev = float(ce.get(key)); break
                        except: pass
                if prev is None:
                    # reconstruct
                    ch = ce.get("changeinOpenInterest") or ce.get("changeInOpenInterest") or 0
                    try:
                        prev = oi_now - float(ch)
                    except:
                        prev = None
                if prev:
                    sum_call_prev += prev
                if sp==atm:
                    iv = ce.get("impliedVolatility")
                    if iv is not None:
                        try: call_iv_atm = float(iv)/100.0
                        except: call_iv_atm=None
    delta_call_pct = 0.0 if sum_call_prev==0 else (sum_call_now - sum_call_prev)/sum_call_prev
    # heuristics: strong gamma blast if delta_call_pct large and totalOI big and days_to_expiry small
    # days to expiry - approximate from expiryDates
    days_left = 7
    try:
        edates = rec.get("expiryDates",[])
        if edates:
            # pick nearest >= today
            parsed = []
            for s in edates:
                for fmt in ("%d-%b-%Y","%d-%B-%Y","%d-%m-%Y"):
                    try:
                        parsed.append(datetime.strptime(s,fmt).date()); break
                    except: pass
            if parsed:
                today = datetime.now().date()
                parsed_sorted = sorted(parsed)
                for d in parsed_sorted:
                    if d>=today:
                        days_left = (d - today).days
                        break
    except Exception:
        pass
    # score build: base = total_oi * delta_call_pct, then adjust
    base = total_oi * max(delta_call_pct,0)
    # scale factors to map to 0..1 (heuristic)
    # choose denominators relative to order-of-magnitude of OI for NIFTY/BANKNIFTY
    denom = 5e6 if index_choice=="NIFTY" else 2e6 if index_choice=="BANKNIFTY" else 1e6
    raw = base/denom
    # boost if days_left<=1 and iv low
    iv_boost = 1.0
    if call_iv_atm and call_iv_atm < iv_threshold:
        iv_boost += 0.5
    if days_left<=1:
        iv_boost += 0.5
    raw = raw * iv_boost
    score = max(0.0, min(raw, 1.0))
    msg = f"ΔCallOI%={delta_call_pct*100:.2f}%, TotalOI={int(total_oi)}, ATM IV={call_iv_atm if call_iv_atm else 'n/a'}, days_left={days_left}"
    top_strike = atm
    return score, msg, {"delta_call_pct":delta_call_pct,"total_oi":total_oi,"atm_iv":call_iv_atm,"days_left":days_left,"strike":top_strike}

def detect_iv_shock(oc, spot, strike_gap, n):
    rec = oc.get("records",{}); data = rec.get("data",[])
    atm = int(round(spot/strike_gap)*strike_gap)
    # compute avg IV ATM±n now and prev (from session_state.prev_iv)
    ivs_now = []
    for e in data:
        try:
            sp = int(e.get("strikePrice", -1))
        except: continue
        if sp in [atm + i*strike_gap for i in range(-n,n+1)]:
            ce = e.get("CE")
            pe = e.get("PE")
            # use avg of CE and PE IV if present
            local = []
            if ce and ce.get("impliedVolatility") is not None:
                try: local.append(float(ce.get("impliedVolatility"))/100.0)
                except: pass
            if pe and pe.get("impliedVolatility") is not None:
                try: local.append(float(pe.get("impliedVolatility"))/100.0)
                except: pass
            if local:
                ivs_now.append(sum(local)/len(local))
    if not ivs_now:
        return 0.0, "no IV data", {"iv_now_avg":None}
    iv_now = sum(ivs_now)/len(ivs_now)
    prev_map = st.session_state.prev_iv.get(index_choice,{})
    prev_vals = [prev_map.get(s) for s in [atm + i*strike_gap for i in range(-n,n+1)] if prev_map.get(s) is not None]
    if not prev_vals:
        delta_iv = 0.0
    else:
        prev_avg = sum(prev_vals)/len(prev_vals)
        delta_iv = 0.0 if prev_avg==0 else (iv_now - prev_avg)/prev_avg
    raw = max(0.0, min(delta_iv*5.0,1.0))  # scale: 20% iv jump => raw=1
    msg = f"ΔIV%={delta_iv*100:.2f}%, IV_now={iv_now:.3f}, prev_avg={prev_avg if prev_vals else 'n/a'}"
    return raw, msg, {"iv_now_avg":iv_now,"delta_iv":delta_iv}

def detect_put_panic(oc, spot, strike_gap, n):
    rec = oc.get("records",{}); data = rec.get("data",[])
    atm = int(round(spot/strike_gap)*strike_gap)
    strikes = [atm + i*strike_gap for i in range(-n,n+1)]
    put_sum_now = 0.0; put_sum_prev = 0.0
    for e in data:
        try:
            sp = int(e.get("strikePrice", -1))
        except: continue
        if sp in strikes:
            pe = e.get("PE")
            if pe:
                oi = float(pe.get("openInterest") or 0)
                put_sum_now += oi
                prev = None
                for k in ("previousOpenInterest","prevOpenInterest","prevOI","previousOI"):
                    if k in pe and pe.get(k) is not None:
                        try: prev = float(pe.get(k)); break
                        except: pass
                if prev is None:
                    try:
                        ch = pe.get("changeinOpenInterest") or pe.get("changeInOpenInterest") or 0
                        prev = oi - float(ch)
                    except: prev=None
                if prev:
                    put_sum_prev += prev
    if put_sum_prev==0:
        raw = 0.0
        delta = 0.0
    else:
        delta = (put_sum_now - put_sum_prev)/put_sum_prev
        # panic defined as big negative drop in put OI (sellers covering) -> delta < 0 large negative
        raw = max(0.0, min(-delta*1.5, 1.0)) if delta<0 else 0.0
    msg = f"ΔPutOI%={delta*100:.2f}%, putOI_now={int(put_sum_now)}"
    return raw, msg, {"delta_put_pct":delta,"put_now":put_sum_now}

def detect_call_short_pressure(oc, spot, strike_gap, n):
    rec = oc.get("records",{}); data = rec.get("data",[])
    atm = int(round(spot/strike_gap)*strike_gap)
    strikes = [atm + i*strike_gap for i in range(-n,n+1)]
    call_sum_now = 0.0; call_sum_prev = 0.0
    for e in data:
        try:
            sp = int(e.get("strikePrice", -1))
        except: continue
        if sp in strikes:
            ce = e.get("CE")
            if ce:
                oi = float(ce.get("openInterest") or 0)
                call_sum_now += oi
                prev = None
                for k in ("previousOpenInterest","prevOpenInterest","prevOI","previousOI"):
                    if k in ce and ce.get(k) is not None:
                        try: prev = float(ce.get(k)); break
                        except: pass
                if prev is None:
                    try:
                        ch = ce.get("changeinOpenInterest") or ce.get("changeInOpenInterest") or 0
                        prev = oi - float(ch)
                    except: prev=None
                if prev:
                    call_sum_prev += prev
    if call_sum_prev==0:
        raw=0.0; delta=0.0
    else:
        delta = (call_sum_now - call_sum_prev)/call_sum_prev
        raw = max(0.0, min(delta*1.5,1.0))  # positive delta -> more call OI building (writers), indicates shorting pressure
    msg = f"ΔCallOI%={delta*100:.2f}%, callOI_now={int(call_sum_now)}"
    return raw, msg, {"delta_call_pct":delta,"call_now":call_sum_now}

def detect_oi_pressure_ratio(oc):
    rec = oc.get("records",{}); data = rec.get("data",[])
    sum_call=0; sum_put=0
    for e in data:
        ce = e.get("CE"); pe = e.get("PE")
        if ce and ce.get("openInterest") is not None:
            try: sum_call += float(ce.get("openInterest"))
            except: pass
        if pe and pe.get("openInterest") is not None:
            try: sum_put += float(pe.get("openInterest"))
            except: pass
    if sum_put==0:
        ratio = float('inf') if sum_call>0 else 1.0
    else:
        ratio = sum_call/sum_put
    # normalize ratio to 0..1 where >1.2 means call dominance (bullish pressure), <0.8 means put dominance (bearish)
    if ratio==float('inf'):
        raw=1.0
    else:
        if ratio>=1.2:
            raw = min((ratio-1.2)/2.0,1.0)  # map 1.2..3.2 to 0..1
        elif ratio<=0.8:
            raw = min((0.8-ratio)/0.8,1.0)
        else:
            raw=0.0
    msg = f"Call/Put OI ratio={ratio:.2f}"
    return raw, msg, {"ratio":ratio,"call":sum_call,"put":sum_put}

def detect_iv_oi_explosion(oc):
    # both IV up and total OI up recently -> strong move likely
    rec = oc.get("records",{}); data = rec.get("data",[])
    total_oi_now=0; ivs=[]
    for e in data:
        ce = e.get("CE"); pe = e.get("PE")
        if ce and ce.get("openInterest") is not None:
            try: total_oi_now += float(ce.get("openInterest"))
            except: pass
        if ce and ce.get("impliedVolatility") is not None:
            try: ivs.append(float(ce.get("impliedVolatility"))/100.0)
            except: pass
        if pe and pe.get("impliedVolatility") is not None:
            try: ivs.append(float(pe.get("impliedVolatility"))/100.0)
            except: pass
    iv_now_avg = sum(ivs)/len(ivs) if ivs else None
    # compare with prev saved values
    prev_total = st.session_state.prev_oi.get("_total_oi",0)
    prev_iv = st.session_state.prev_iv.get("_avg_iv", None)
    # store current for next round
    st.session_state.prev_oi["_total_oi"] = total_oi_now
    if iv_now_avg is not None:
        st.session_state.prev_iv["_avg_iv"] = iv_now_avg
    # compute changes
    delta_oi = 0.0 if prev_total==0 else (total_oi_now - prev_total)/prev_total
    delta_iv = 0.0 if prev_iv is None else (iv_now_avg - prev_iv)/prev_iv if prev_iv!=0 else 0.0
    raw = max(0.0, min((delta_oi*2.0 + delta_iv*3.0)/4.0,1.0))
    msg = f"ΔTotalOI%={delta_oi*100:.2f}%, ΔAvgIV%={delta_iv*100:.2f}%"
    return raw, msg, {"delta_oi":delta_oi,"delta_iv":delta_iv}

def detect_max_pain_shift(oc):
    # compute max pain from chain by summing P/L across strikes for options buyers (approx)
    data = oc.get("records",{}).get("data",[])
    # crude max pain: find strike minimizing sum of OI*abs(strike-current)
    # This is an approximation — fine for detection of shift
    try:
        strikes=[]
        pains=[]
        for e in data:
            sp = int(e.get("strikePrice",-1))
            ce = e.get("CE"); pe = e.get("PE")
            oi_sum = 0
            if ce and ce.get("openInterest") is not None:
                oi_sum += float(ce.get("openInterest"))
            if pe and pe.get("openInterest") is not None:
                oi_sum += float(pe.get("openInterest"))
            strikes.append(sp); pains.append(oi_sum)
        if not strikes:
            return 0.0, "no chain", {}
        # compute weighted center as proxy
        weighted = sum([s*p for s,p in zip(strikes,pains)])/sum(pains) if sum(pains)!=0 else strikes[len(strikes)//2]
        current_mp = int(round(weighted/strike_gap)*strike_gap)
        prev_mp = st.session_state.past_max_pain
        st.session_state.past_max_pain = current_mp
        if prev_mp is None:
            raw=0.0; delta=0
        else:
            delta = (current_mp - prev_mp)
            raw = max(0.0, min(abs(delta)/strike_gap, 1.0))
        msg = f"MaxPain shifted from {prev_mp} to {current_mp}" if prev_mp else f"MaxPain set {current_mp}"
        return raw, msg, {"current_mp":current_mp,"prev_mp":prev_mp}
    except Exception as e:
        return 0.0, f"err {e}", {}

def detect_oi_wall_break(oc, spot, strike_gap):
    # detect if spot crosses a strike which had heavy OI (wall)
    data = oc.get("records",{}).get("data",[])
    walls = []
    for e in data:
        try:
            sp = int(e.get("strikePrice",-1))
        except: continue
        ce = e.get("CE"); pe = e.get("PE")
        oi = 0
        if ce and ce.get("openInterest") is not None:
            oi += float(ce.get("openInterest"))
        if pe and pe.get("openInterest") is not None:
            oi += float(pe.get("openInterest"))
        walls.append((sp,oi))
    if not walls:
        return 0.0, "no walls", {}
    # find strike with max oi
    top = max(walls, key=lambda x:x[1])
    top_strike, top_oi = top
    # if spot just crossed that strike relative to previous spot, it's a break
    prev_spot = st.session_state.last_open.get("_last_spot", None)
    st.session_state.last_open["_last_spot"] = spot
    if prev_spot is None:
        return 0.0, "no prev spot", {"top_strike":top_strike,"top_oi":top_oi}
    crossed = (prev_spot < top_strike <= spot) or (prev_spot > top_strike >= spot)
    raw = 1.0 if crossed and top_oi>50000 else 0.0
    msg = f"Top OI wall at {top_strike} ({int(top_oi)}) crossed={crossed}"
    return raw, msg, {"top_strike":top_strike,"top_oi":top_oi,"crossed":crossed}

# -------------------------
# Composite / regime logic
# -------------------------
def normalize(v):
    return max(0.0, min(v,1.0))

def compute_composite(detector_scores, weights):
    total = 0.0
    for k,w in weights.items():
        total += detector_scores.get(k,0.0)*w
    # map to 0..100
    return max(0.0, min(total*100.0, 100.0))

def classify_regime(detector_scores):
    # quick rule-based regime classification
    if detector_scores.get("gamma_blast",0)>0.6 and detector_scores.get("iv_oi_explosion",0)>0.4:
        return "Explosive (News/Huge Flow)"
    if detector_scores.get("call_short",0)>0.5 and detector_scores.get("oi_pressure",0)>0.4:
        return "Pinned/Writer-Dominated (Bearish)"
    if detector_scores.get("put_panic",0)>0.5:
        return "Short-Covering (Bullish)"
    if detector_scores.get("iv_shock",0)>0.5:
        return "Volatility Surge"
    return "Normal / Balanced"

# -------------------------
# UI placeholders
# -------------------------
left, mid, right = st.columns([2,1,1])
spot_ph = left.empty()
pct_ph = mid.empty()
score_ph = right.empty()

status_area = st.empty()
detail_box = st.empty()
table_area = st.empty()
chart_area = st.empty()
history_area = st.empty()

c1,c2,c3 = st.columns([1,1,1])
if c1.button("Start"): st.session_state.running=True
if c2.button("Stop"): st.session_state.running=False
if c3.button("Clear Log"):
    st.session_state.history = st.session_state.history.iloc[0:0]
    st.success("Cleared history")

st.write("---")

# auto-refresh while running
if st.session_state.running:
    st_autorefresh(interval=poll_seconds*1000, key="autorefresh_pro")

# single-run compute on each refresh
if st.session_state.running:
    try:
        # fetch data
        oc=None
        spot=None
        if provider=="NSE (option-chain)" and index_choice in ("NIFTY","BANKNIFTY"):
            try:
                oc = fetch_nse_option_chain(index_choice)
                spot = float(oc.get("records",{}).get("underlyingValue", None))
            except Exception as e:
                st.warning(f"NSE fetch failed: {e}. Falling back to yfinance spot.")
                spot = fetch_spot_yf(index_choice)
        else:
            spot = fetch_spot_yf(index_choice)
        if spot is None:
            st.error("Cannot fetch spot. Stop and check provider/network.")
            st.stop()

        # run detectors (for SENSEX use fallback compute where oc is None)
        detector_scores = {}
        detector_msgs = {}
        extras = {}

        if oc is not None:
            s_g, m_g, ex_g = detect_gamma_blast(oc, spot, strike_gap, int(num_strikes))
            s_iv, m_iv, ex_iv = detect_iv_shock(oc, spot, strike_gap, int(num_strikes))
            s_put, m_put, ex_put = detect_put_panic(oc, spot, strike_gap, int(num_strikes))
            s_call, m_call, ex_call = detect_call_short_pressure(oc, spot, strike_gap, int(num_strikes))
            s_ratio, m_ratio, ex_ratio = detect_oi_pressure_ratio(oc)
            s_expl, m_expl, ex_expl = detect_iv_oi_explosion(oc)
            s_mp, m_mp, ex_mp = detect_max_pain_shift(oc)
            s_wall, m_wall, ex_wall = detect_oi_wall_break(oc, spot, strike_gap)
        else:
            # no chain: fallback minimal detectors from spot movement
            s_g, m_g, ex_g = 0.0, "no chain", {}
            s_iv, m_iv, ex_iv = 0.0, "no chain", {}
            s_put, m_put, ex_put = 0.0, "no chain", {}
            s_call, m_call, ex_call = 0.0, "no chain", {}
            s_ratio, m_ratio, ex_ratio = 0.0, "no chain", {}
            s_expl, m_expl, ex_expl = 0.0, "no chain", {}
            s_mp, m_mp, ex_mp = 0.0, "no chain", {}
            s_wall, m_wall, ex_wall = 0.0, "no chain", {}

        detector_scores["gamma_blast"] = normalize(s_g)
        detector_scores["iv_shock"] = normalize(s_iv)
        detector_scores["put_panic"] = normalize(s_put)
        detector_scores["call_short"] = normalize(s_call)
        detector_scores["oi_pressure"] = normalize(s_ratio)
        detector_scores["iv_oi_explosion"] = normalize(s_expl)
        detector_scores["max_pain_shift"] = normalize(s_mp)
        detector_scores["oi_wall_break"] = normalize(s_wall)

        detector_msgs["gamma_blast"] = m_g
        detector_msgs["iv_shock"] = m_iv
        detector_msgs["put_panic"] = m_put
        detector_msgs["call_short"] = m_call
        detector_msgs["oi_pressure"] = m_ratio
        detector_msgs["iv_oi_explosion"] = m_expl
        detector_msgs["max_pain_shift"] = m_mp
        detector_msgs["oi_wall_break"] = m_wall

        # composite
        unified = compute_composite(detector_scores, weights)
        st.session_state.recent_scores.append(unified)
        if len(st.session_state.recent_scores)>score_window:
            st.session_state.recent_scores.pop(0)

        # threshold selection
        if threshold_mode=="Percentile":
            threshold = float(pd.Series(st.session_state.recent_scores).quantile(percentile_for_threshold/100.0)) if st.session_state.recent_scores else 999999
        else:
            arr = st.session_state.recent_scores
            mean = float(pd.Series(arr).mean()) if arr else 0.0
            std = float(pd.Series(arr).std()) if arr else 0.0
            threshold = mean + k_std*std

        # pack top signals
        top_list = sorted(detector_scores.items(), key=lambda x: x[1], reverse=True)
        top_signals = [k for k,v in top_list if v>0.2][:3]
        top_msg = ", ".join(top_signals) if top_signals else "None"

        # regime
        regime = classify_regime(detector_scores)

        # display main metrics
        spot_ph.metric(f"📈 {index_choice} Spot", f"{spot:,.2f}")
        openp = get_open_price(index_choice, oc)
        pct = ((spot-openp)/openp*100) if openp else 0.0
        pct_ph.metric("Intraday %", f"{pct:.3f}%")
        score_ph.metric("Market Action Score", f"{unified:.2f} / 100", delta=None)

        # ALERT block
        is_blast = unified>threshold and unified>0
        if is_blast:
            status_area.error(
                f"🔥 MARKET ACTION ALERT — Score {unified:.2f} > threshold {threshold:.2f}\n\n"
                f"Regime: {regime}\nTopSignals: {top_msg}\nSpot: {spot:.2f} ({pct:.2f}% intraday)"
            )
            # log
            details_msg = "; ".join([f"{k}:{detector_msgs[k]}" for k in detector_msgs.keys()])
            newrow = {
                "Timestamp": datetime.now().isoformat(),
                "Index": index_choice,
                "Spot": round(spot,2),
                "ATM": int(round(spot/strike_gap)*strike_gap),
                "Strike": int(round(spot/strike_gap)*strike_gap),
                "UnifiedScore": round(unified,2),
                "Regime": regime,
                "TopSignals": top_msg,
                "Details": details_msg
            }
            st.session_state.history = pd.concat([pd.DataFrame([newrow]), st.session_state.history], ignore_index=True).head(history_max)
        else:
            status_area.success(f"🟢 Stable — Score {unified:.2f} (threshold {threshold:.2f}) — Regime: {regime} — Top: {top_msg}")

        # Option B informational box (expandable)
        with detail_box.expander("Detector details (why the score) — click to expand", expanded=False):
            for key in ["gamma_blast","iv_shock","iv_oi_explosion","call_short","put_panic","oi_pressure","max_pain_shift","oi_wall_break"]:
                st.write(f"**{key}** — score={detector_scores.get(key,0):.3f} | {detector_msgs.get(key,'')}")
            st.write("---")
            st.write("Raw recent scores (last points):")
            st.write(st.session_state.recent_scores[-10:])

        # breakdown table
        if display_oi_breakdown and oc is not None and index_choice in ("NIFTY","BANKNIFTY"):
            # prepare ATM±n call OI & IV
            atm = int(round(spot/strike_gap)*strike_gap)
            strikes = [atm + i*strike_gap for i in range(-int(num_strikes), int(num_strikes)+1)]
            per_oi = {}
            per_iv = {}
            for e in oc.get("records",{}).get("data",[]):
                try:
                    sp = int(e.get("strikePrice",-1))
                except: continue
                if sp in strikes:
                    ce = e.get("CE")
                    if ce:
                        per_oi[sp] = int(ce.get("openInterest") or 0)
                        try:
                            per_iv[sp] = float(ce.get("impliedVolatility"))/100.0 if ce.get("impliedVolatility") is not None else None
                        except:
                            per_iv[sp] = None
                    else:
                        per_oi[sp]=0; per_iv[sp]=None
            df_break = pd.DataFrame({"Strike":list(per_oi.keys()), "CallOI":list(per_oi.values()), "CallIV":[per_iv.get(s) for s in per_oi.keys()]}).sort_values("Strike")
            table_area.subheader("ATM ± n breakdown")
            table_area.dataframe(df_break, use_container_width=True)
        else:
            table_area.write("")

        # history table
        history_area.subheader("Alert History (session)")
        history_area.dataframe(st.session_state.history.reset_index(drop=True), use_container_width=True)

        # charts: composite + threshold + markers
        st.session_state.chart_history.append({"t":datetime.now(),"score":unified,"threshold":threshold,"blast":bool(is_blast)})
        st.session_state.chart_history = st.session_state.chart_history[-500:]
        cdf = pd.DataFrame(st.session_state.chart_history)
        if not cdf.empty:
            base = alt.Chart(cdf).encode(x=alt.X("t:T", title="Time"))
            line = base.mark_line(color="steelblue").encode(y=alt.Y("score:Q", title="Market Action Score"))
            rule = base.mark_rule(color="orange", strokeDash=[4,4]).encode(y="threshold:Q")
            pts = alt.Chart(cdf).mark_circle(size=70).encode(
                x="t:T", y="score:Q",
                color=alt.condition(alt.datum.blast==True, alt.value("red"), alt.value("steelblue")),
                tooltip=[alt.Tooltip("t:T","Time"), alt.Tooltip("score:Q",format=".2f"), alt.Tooltip("threshold:Q",format=".2f"), "blast"]
            )
            final = (line + rule + pts).properties(height=320).interactive()
            chart_area.subheader("📈 Market Action Score (live)")
            chart_area.altair_chart(final, use_container_width=True)

    except Exception as e:
        st.error(f"Monitoring error: {e}")
        st.session_state.running = False

else:
    st.info("Press Start to begin monitoring. Alerts are session-only and reset on refresh.")
    history_area.subheader("Alert History (session)")
    history_area.dataframe(st.session_state.history.reset_index(drop=True), use_container_width=True)

# -------------------------
# Extra: export CSV
# -------------------------
st.write("---")
col_a, col_b = st.columns([1,3])
with col_a:
    if st.button("Export History CSV"):
        csv = st.session_state.history.to_csv(index=False)
        st.download_button("Download CSV", data=csv, file_name=f"gamma_blast_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
with col_b:
    st.write("Tips: Use threshold percentile and window size to calibrate sensitivity. For production reliability use a broker API (Kite/Fyers) for robust option chains.")