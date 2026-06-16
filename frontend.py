"""
NOC Agentic Copilot - Frontend
Streamlit multi-page application
Run: streamlit run frontend.py
Requires backend running at http://localhost:8000
"""

import streamlit as st
import requests, json, random
from datetime import datetime
import pandas as pd

# ── Config ─────────────────────────────────────────────────────────────────────
API = "http://localhost:8000"
st.set_page_config(
    page_title="NOC Agentic Copilot",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Colour tokens ──────────────────────────────────────────────────────────────
SEV_COLOR = {"Critical":"#FF3B3B","Major":"#FF8C00","Minor":"#FFD700","Warning":"#4FC3F7"}
STATUS_COLOR = {"Open":"#FF3B3B","In Progress":"#FF8C00","Resolved":"#00C853","Acknowledged":"#4FC3F7"}

# ── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* ── palette ── */
  :root{
    --bg-dark:#0A0E1A;--bg-card:#111827;--bg-card2:#1a2236;
    --accent:#00E5FF;--accent2:#7C3AED;--accent3:#00C853;
    --sev-crit:#FF3B3B;--sev-maj:#FF8C00;--sev-min:#FFD700;
    --text:#E5E7EB;--text-muted:#6B7280;--border:#1F2D45;
  }
  /* app shell */
  .stApp{background:var(--bg-dark);color:var(--text);font-family:'JetBrains Mono',monospace}
  .stSidebar{background:#070C18 !important;border-right:1px solid var(--border)}
  /* header strip */
  .noc-header{background:linear-gradient(90deg,#0A0E1A 0%,#0d2040 60%,#0A0E1A 100%);
    border-bottom:1px solid var(--accent);padding:12px 24px;
    display:flex;align-items:center;gap:16px;margin-bottom:24px}
  .noc-logo{font-size:2rem;color:var(--accent);animation:pulse 2s infinite}
  .noc-title{font-size:1.4rem;font-weight:700;letter-spacing:.06em;color:var(--accent)}
  .noc-sub{font-size:.75rem;color:var(--text-muted);letter-spacing:.12em;text-transform:uppercase}
  @keyframes pulse{0%,100%{text-shadow:0 0 8px var(--accent)}50%{text-shadow:0 0 24px var(--accent)}}
  /* metric cards */
  .metric-row{display:flex;gap:16px;flex-wrap:wrap;margin-bottom:20px}
  .metric-card{background:var(--bg-card);border:1px solid var(--border);
    border-radius:10px;padding:18px 22px;min-width:160px;flex:1}
  .metric-val{font-size:2rem;font-weight:800;color:var(--accent)}
  .metric-label{font-size:.72rem;text-transform:uppercase;letter-spacing:.1em;color:var(--text-muted);margin-top:4px}
  /* table */
  .noc-table{width:100%;border-collapse:collapse;font-size:.82rem}
  .noc-table th{background:var(--bg-card2);color:var(--text-muted);
    text-transform:uppercase;letter-spacing:.08em;padding:10px 14px;
    border-bottom:1px solid var(--border);text-align:left}
  .noc-table td{padding:9px 14px;border-bottom:1px solid var(--border);color:var(--text)}
  .noc-table tr:hover td{background:var(--bg-card2)}
  /* severity badges */
  .badge{display:inline-block;border-radius:4px;padding:2px 10px;font-size:.72rem;font-weight:700}
  .badge-Critical{background:#3B0000;color:var(--sev-crit);border:1px solid var(--sev-crit)}
  .badge-Major{background:#2D1500;color:var(--sev-maj);border:1px solid var(--sev-maj)}
  .badge-Minor{background:#2D2800;color:var(--sev-min);border:1px solid var(--sev-min)}
  .badge-Resolved{background:#003D1A;color:var(--accent3);border:1px solid var(--accent3)}
  .badge-Open{background:#3B0000;color:var(--sev-crit);border:1px solid var(--sev-crit)}
  .badge-InProgress{background:#2D1500;color:var(--sev-maj);border:1px solid var(--sev-maj)}
  /* agent cards */
  .agent-card{background:var(--bg-card);border:1px solid var(--border);border-radius:10px;
    padding:16px 20px;margin-bottom:14px}
  .agent-header{display:flex;align-items:center;gap:10px;margin-bottom:10px}
  .agent-icon{font-size:1.4rem}
  .agent-name{font-weight:700;color:var(--accent);font-size:.95rem}
  .agent-ts{font-size:.7rem;color:var(--text-muted);margin-left:auto}
  .agent-body{font-size:.82rem;color:var(--text);padding-left:4px}
  /* confidence bar */
  .conf-bar-wrap{background:#1F2D45;border-radius:4px;height:8px;overflow:hidden;margin-top:6px}
  .conf-bar{height:100%;border-radius:4px;background:linear-gradient(90deg,var(--accent),var(--accent2))}
  /* step list */
  .step-item{display:flex;gap:12px;align-items:flex-start;padding:8px 0;border-bottom:1px solid var(--border)}
  .step-num{background:var(--accent2);color:#fff;border-radius:50%;
    width:24px;height:24px;display:flex;align-items:center;justify-content:center;
    font-size:.75rem;font-weight:700;flex-shrink:0}
  .step-auto{font-size:.7rem;color:var(--accent3);border:1px solid var(--accent3);
    border-radius:3px;padding:1px 6px;margin-left:auto;flex-shrink:0}
  .step-manual{font-size:.7rem;color:var(--sev-maj);border:1px solid var(--sev-maj);
    border-radius:3px;padding:1px 6px;margin-left:auto;flex-shrink:0}
  /* topology */
  .topo-node{display:inline-block;background:var(--bg-card2);border:1px solid var(--accent);
    border-radius:6px;padding:6px 12px;font-size:.75rem;font-weight:600;color:var(--accent);margin:4px}
  .topo-link{color:var(--text-muted);font-size:.75rem;margin:2px 8px}
  /* trace step */
  .trace-step{border-left:3px solid var(--accent);padding-left:14px;margin-bottom:18px}
  .trace-agent{font-weight:700;color:var(--accent);font-size:.9rem}
  /* section divider */
  .sec-divider{border:none;border-top:1px solid var(--border);margin:20px 0}
  /* sidebar nav */
  div[data-testid="stSidebarNav"] a{color:var(--text-muted) !important}
  div[data-testid="stSidebarNav"] a:hover{color:var(--accent) !important}
  /* input overrides */
  .stTextInput>div>div>input,.stTextArea>div>div>textarea,
  .stSelectbox>div>div{background:var(--bg-card2)!important;color:var(--text)!important;
    border:1px solid var(--border)!important;border-radius:6px!important}
  .stButton>button{background:linear-gradient(135deg,#0d2040,#0a3870);
    color:var(--accent);border:1px solid var(--accent);border-radius:6px;
    font-weight:600;padding:8px 22px;letter-spacing:.06em;transition:.2s}
  .stButton>button:hover{background:var(--accent);color:#000;box-shadow:0 0 16px var(--accent)}
  /* success/danger overrides */
  .stSuccess{background:#003D1A!important;border-color:var(--accent3)!important}
  .stError{background:#3B0000!important;border-color:var(--sev-crit)!important}
  .stWarning{background:#2D1500!important;border-color:var(--sev-maj)!important}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def api(method, path, **kwargs):
    try:
        r = getattr(requests, method)(f"{API}{path}", timeout=60, **kwargs)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        st.error("⚠️  Cannot reach backend. Start it with: `uvicorn backend:app --reload`")
        return None
    except Exception as e:
        st.error(f"API error: {e}")
        return None

def badge(text, cls=None):
    cls = cls or f"badge-{text.replace(' ','')}"
    return f'<span class="badge {cls}">{text}</span>'

def conf_bar(val: int):
    return (f'<div class="conf-bar-wrap"><div class="conf-bar" style="width:{val}%"></div></div>'
            f'<div style="font-size:.7rem;color:#6B7280;margin-top:2px">{val}% confidence</div>')

def render_header(icon, title, subtitle=""):
    st.markdown(f"""
    <div class="noc-header">
      <span class="noc-logo">{icon}</span>
      <div>
        <div class="noc-title">{title}</div>
        <div class="noc-sub">{subtitle}</div>
      </div>
    </div>""", unsafe_allow_html=True)

def ts_fmt(ts_str):
    try:
        return datetime.fromisoformat(ts_str.rstrip("Z")).strftime("%b %d %H:%M")
    except Exception:
        return ts_str or "—"

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:12px 0 20px">
      <div style="font-size:2.5rem">🛰️</div>
      <div style="font-size:1.1rem;font-weight:700;color:#00E5FF;letter-spacing:.08em">NOC COPILOT</div>
      <div style="font-size:.68rem;color:#6B7280;letter-spacing:.14em">AGENTIC AI PLATFORM</div>
    </div>""", unsafe_allow_html=True)

    pages = {
        "🏠  Dashboard":          "dashboard",
        "🔔  Real-Time Alarms":   "alarms",
        "🔗  Event Correlation":  "correlation",
        "🔍  Root Cause Analysis":"rca",
        "📚  Knowledge Retrieval":"knowledge",
        "🛡️  SLA Risk Monitor":   "sla",
        "🔧  Remediation Center": "remediation",
        "⚙️  Automation Execution":"automation",
        "🎫  ITSM Tickets":       "itsm",
        "🌐  Network Topology":   "topology",
        "📊  Agent Trace":        "trace",
        "➕  Manual Alarm Input": "manual",
    }
    if "page" not in st.session_state:
        st.session_state.page = "dashboard"

    for label, key in pages.items():
        active = "color:#00E5FF;font-weight:700" if st.session_state.page==key else "color:#6B7280"
        if st.button(label, key=f"nav_{key}", use_container_width=True):
            st.session_state.page = key

    st.markdown("<hr style='border-color:#1F2D45;margin:16px 0'>", unsafe_allow_html=True)

    # Quick health check
    health = api("get", "/health")
    if health:
        st.markdown(f"""<div style='font-size:.72rem;color:#00C853'>
          ✅ Backend Online<br>
          <span style='color:#6B7280'>Agents: {health.get('agents',8)} | FAISS: {health.get('vector_docs','—')}</span>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("<div style='font-size:.72rem;color:#FF3B3B'>❌ Backend Offline</div>",
                    unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.page == "dashboard":
    render_header("🛰️","NOC Agentic Copilot","Real-Time Network Operations Intelligence")

    data = api("get","/alarms",params={"limit":200})
    alarms = data["alarms"] if data else []

    # KPI row
    sev_counts = {"Critical":0,"Major":0,"Minor":0,"Warning":0}
    for a in alarms:
        s = a.get("severity","")
        if s in sev_counts: sev_counts[s]+=1

    cols = st.columns(5)
    metrics = [
        ("Total Alarms", len(alarms), "#00E5FF"),
        ("Critical",     sev_counts["Critical"], "#FF3B3B"),
        ("Major",        sev_counts["Major"],    "#FF8C00"),
        ("Minor",        sev_counts["Minor"],    "#FFD700"),
        ("Resolved",     sum(1 for a in alarms if a.get("status")=="Resolved"), "#00C853"),
    ]
    for col,(lbl,val,clr) in zip(cols,metrics):
        col.markdown(f"""<div class="metric-card">
          <div class="metric-val" style="color:{clr}">{val}</div>
          <div class="metric-label">{lbl}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<hr class='sec-divider'>", unsafe_allow_html=True)

    # Charts row
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Alarm Severity Distribution**")
        df_sev = pd.DataFrame(list(sev_counts.items()), columns=["Severity","Count"])
        st.bar_chart(df_sev.set_index("Severity"), color="#00E5FF")

    with c2:
        st.markdown("**Top Alarm Types**")
        from collections import Counter
        top = Counter(a.get("alarm","") for a in alarms).most_common(6)
        df_top = pd.DataFrame(top, columns=["Alarm","Count"])
        if not df_top.empty:
            st.bar_chart(df_top.set_index("Alarm"), color="#7C3AED")

    # Recent alarms table
    st.markdown("**Recent Critical Alarms**")
    crit = [a for a in alarms if a.get("severity")=="Critical"][:10]
    rows = ""
    for a in crit:
        rows += (f"<tr><td>{a['alarm_id']}</td><td>{a['device']}</td>"
                 f"<td>{badge(a['severity'])}</td><td>{a['alarm']}</td>"
                 f"<td>{badge(a.get('status','Open'))}</td>"
                 f"<td>{ts_fmt(a['timestamp'])}</td></tr>")
    st.markdown(f"""<table class='noc-table'><thead><tr>
      <th>ID</th><th>Device</th><th>Severity</th><th>Alarm</th><th>Status</th><th>Time</th>
    </tr></thead><tbody>{rows}</tbody></table>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: REAL-TIME ALARMS
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == "alarms":
    render_header("🔔","Real-Time Alarm Dashboard","Live alarm feed with severity classification")

    c1,c2,c3 = st.columns([2,2,1])
    with c1:
        sev_filter = st.selectbox("Filter by Severity",["All","Critical","Major","Minor","Warning"])
    with c2:
        limit = st.slider("Records to show", 10, 200, 50)
    with c3:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔄 Refresh"):
            st.rerun()

    params = {"limit": limit}
    if sev_filter != "All":
        params["severity"] = sev_filter
    data = api("get","/alarms",params=params)
    alarms = data["alarms"] if data else []

    # KPI averages
    kpis = [a.get("kpi",{}) for a in alarms if a.get("kpi")]
    if kpis:
        avg_cpu  = sum(k.get("cpu",0) for k in kpis) / len(kpis)
        avg_mem  = sum(k.get("memory",0) for k in kpis) / len(kpis)
        avg_loss = sum(k.get("packet_loss",0) for k in kpis) / len(kpis)
        kcols    = st.columns(3)
        for col,(lbl,val,unit,clr) in zip(kcols,[
            ("Avg CPU",f"{avg_cpu:.1f}","%","#FF8C00"),
            ("Avg Memory",f"{avg_mem:.1f}","%","#7C3AED"),
            ("Avg Packet Loss",f"{avg_loss:.1f}","%","#FF3B3B"),
        ]):
            col.markdown(f"""<div class="metric-card">
              <div class="metric-val" style="color:{clr}">{val}<span style="font-size:1rem">{unit}</span></div>
              <div class="metric-label">{lbl}</div>
            </div>""", unsafe_allow_html=True)

    # Chart
    if alarms:
        df = pd.DataFrame(alarms)
        df["ts"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.sort_values("ts")
        sev_map = {"Critical":4,"Major":3,"Minor":2,"Warning":1}
        df["sev_num"] = df["severity"].map(sev_map).fillna(0)
        st.markdown("**Alarm Timeline**")
        st.line_chart(df.set_index("ts")["sev_num"], color="#00E5FF")

    # Table
    rows = ""
    for a in alarms:
        kpi  = a.get("kpi",{})
        rows += (f"<tr><td>{a['alarm_id']}</td><td>{a['device']}</td>"
                 f"<td>{badge(a['severity'])}</td>"
                 f"<td style='max-width:180px'>{a['alarm']}</td>"
                 f"<td>{badge(a.get('status','Open'))}</td>"
                 f"<td>{kpi.get('cpu','—')}%</td>"
                 f"<td>{kpi.get('memory','—')}%</td>"
                 f"<td>{kpi.get('packet_loss','—')}%</td>"
                 f"<td>{ts_fmt(a['timestamp'])}</td></tr>")
    st.markdown(f"""<table class='noc-table'><thead><tr>
      <th>ID</th><th>Device</th><th>Severity</th><th>Alarm</th><th>Status</th>
      <th>CPU</th><th>Mem</th><th>Loss</th><th>Time</th>
    </tr></thead><tbody>{rows}</tbody></table>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: MANUAL ALARM INPUT
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == "manual":
    render_header("➕","Manual Alarm & KPI Input","Inject alarms and KPI data into the agentic pipeline")

    tab1, tab2 = st.tabs(["🔔 Submit Alarm", "📊 Submit KPI"])

    with tab1:
        st.markdown("**Submit a New Alarm**")
        c1,c2 = st.columns(2)
        with c1:
            device   = st.selectbox("Device", ["RTR-CORE-01","RTR-CORE-02","RTR-EDGE-01",
                                                "RTR-EDGE-02","SW-AGG-01","PE-RTR-01","CE-RTR-01"])
            alarm_t  = st.selectbox("Alarm Type", [
                "BGP Neighbor Down","OSPF Adjacency Failure","Interface Flapping",
                "High CPU Utilization","Memory Leak Detected","Packet Loss > 5%",
                "Fiber Cut Detected","Core Router Failure","Link Down","Custom…"])
            if alarm_t == "Custom…":
                alarm_t = st.text_input("Custom alarm description")
            severity = st.selectbox("Severity", ["Critical","Major","Minor","Warning"])
        with c2:
            cpu    = st.slider("CPU %",   0, 100, 75)
            mem    = st.slider("Memory %",0, 100, 70)
            loss   = st.slider("Packet Loss %", 0.0, 20.0, 2.0, step=0.5)
            lat    = st.slider("Latency ms", 0, 500, 50)
            bw     = st.slider("Bandwidth Util %", 0, 100, 60)

        auto_approve = st.checkbox("Auto-approve remediation (skip human approval step)")

        if st.button("🚀 Process Alarm via LangGraph Pipeline"):
            payload = {
                "device": device, "alarm": alarm_t, "severity": severity,
                "kpi":{"cpu":cpu,"memory":mem,"packet_loss":loss,"latency_ms":lat,"bandwidth_util":bw},
                "auto_approve": auto_approve,
            }
            with st.spinner("Running all 8 agents via LangGraph…"):
                result = api("post","/alarm",json=payload)
            if result:
                st.success(f"✅ Alarm processed! Ticket: **{result.get('ticket',{}).get('ticket_id','—')}**")
                st.session_state["last_result"] = result
                with st.expander("📋 Full Pipeline Result"):
                    st.json(result)

    with tab2:
        st.markdown("**Manually Inject KPI Metrics**")
        kpi_device = st.selectbox("Device (KPI)", ["RTR-CORE-01","RTR-CORE-02","RTR-EDGE-01",
                                                    "RTR-EDGE-02","SW-AGG-01","PE-RTR-01"])
        k1,k2,k3 = st.columns(3)
        with k1: kpi_cpu  = st.number_input("CPU %",0,100,80)
        with k2: kpi_mem  = st.number_input("Memory %",0,100,75)
        with k3: kpi_loss = st.number_input("Packet Loss %",0.0,20.0,3.0,step=0.5)
        k4,k5 = st.columns(2)
        with k4: kpi_lat  = st.number_input("Latency ms",0,1000,100)
        with k5: kpi_bw   = st.number_input("Bandwidth Util %",0,100,70)
        if st.button("📤 Submit KPI"):
            r = api("post","/kpi",json={"device":kpi_device,"cpu":kpi_cpu,"memory":kpi_mem,
                                         "packet_loss":kpi_loss,"latency_ms":kpi_lat,"bandwidth_util":kpi_bw})
            if r: st.success(f"KPI submitted for {kpi_device}")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: EVENT CORRELATION
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == "correlation":
    render_header("🔗","Event Correlation","Incident grouping and cascading failure detection")

    data = api("get","/incidents",params={"limit":50})
    incidents = data["incidents"] if data else []

    sev_c = {"Critical":0,"Major":0,"Minor":0}
    for i in incidents:
        s = i.get("severity","")
        if s in sev_c: sev_c[s]+=1

    cols = st.columns(3)
    for col,(k,v,clr) in zip(cols,[("Critical",sev_c["Critical"],"#FF3B3B"),
                                     ("Major",sev_c["Major"],"#FF8C00"),
                                     ("Minor",sev_c["Minor"],"#FFD700")]):
        col.markdown(f"""<div class="metric-card">
          <div class="metric-val" style="color:{clr}">{v}</div>
          <div class="metric-label">{k} Incidents</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("**Correlation Groups**")
    for inc in incidents[:15]:
        devs     = inc.get("affected_devices",[])
        aids     = inc.get("alarm_ids",[])
        sev      = inc.get("severity","Major")
        status   = inc.get("status","Open")
        cascade  = len(devs) > 2
        with st.expander(f"{'🔴' if sev=='Critical' else '🟠'} {inc['incident_id']} — {sev} "
                         f"| {len(devs)} devices | {status}", expanded=False):
            c1,c2 = st.columns(2)
            with c1:
                st.markdown(f"**Affected Devices:** {', '.join(devs)}")
                st.markdown(f"**Alarm Count:** {len(aids)}")
                st.markdown(f"**Cascading Failure:** {'⚠️ Yes' if cascade else '✅ No'}")
            with c2:
                st.markdown(f"**Root Cause:** {inc.get('root_cause','Analyzing…')}")
                st.markdown(f"**Resolution:** {inc.get('resolution','Pending…')}")
                st.markdown(f"**Time:** {ts_fmt(inc.get('timestamp',''))}")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: ROOT CAUSE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == "rca":
    render_header("🔍","Root Cause Analysis","AI-powered RCA from alarm, KPI and log correlation")

    result = st.session_state.get("last_result")
    if not result:
        st.info("ℹ️  No pipeline result in session. Submit an alarm via **Manual Alarm Input** to see live RCA.")
    else:
        rca = result.get("root_cause",{})
        st.markdown(f"""<div class="agent-card">
          <div class="agent-header">
            <span class="agent-icon">🔬</span>
            <span class="agent-name">Root Cause Analysis Agent</span>
          </div>
          <div class="agent-body">
            <b style="color:#00E5FF">Root Cause:</b><br>
            <span style="font-size:1rem">{rca.get('root_cause','—')}</span><br><br>
            <b>Impact Scope:</b> {rca.get('impact_scope','—')}<br>
            <b>Est. Recovery:</b> {rca.get('estimated_recovery_time_minutes','—')} minutes<br>
            <b>Contributing Factors:</b> {', '.join(rca.get('contributing_factors',[]) or [])}
          </div>
          {conf_bar(rca.get('confidence',0))}
        </div>""", unsafe_allow_html=True)

        norm = result.get("normalized_alarm",{})
        corr = result.get("correlation",{})
        c1,c2 = st.columns(2)
        with c1:
            st.markdown("**Normalized Alarm**")
            st.json(norm)
        with c2:
            st.markdown("**Correlation Data**")
            st.json(corr)

    # Historical incidents table
    st.markdown("<hr class='sec-divider'>", unsafe_allow_html=True)
    st.markdown("**Historical Incident RCA Database**")
    data = api("get","/incidents",params={"limit":30})
    incs = data["incidents"] if data else []
    rows = ""
    for i in incs[:12]:
        rows += (f"<tr><td>{i['incident_id']}</td>"
                 f"<td>{badge(i.get('severity','Minor'))}</td>"
                 f"<td style='max-width:200px;font-size:.78rem'>{i.get('root_cause','—')[:80]}</td>"
                 f"<td style='max-width:200px;font-size:.78rem'>{i.get('resolution','—')[:80]}</td>"
                 f"<td>{badge(i.get('status','Open'))}</td></tr>")
    st.markdown(f"""<table class='noc-table'><thead><tr>
      <th>Incident ID</th><th>Severity</th><th>Root Cause</th><th>Resolution</th><th>Status</th>
    </tr></thead><tbody>{rows}</tbody></table>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: KNOWLEDGE RETRIEVAL
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == "knowledge":
    render_header("📚","Knowledge Retrieval","FAISS semantic search over SOP, KB and historical incidents")

    query = st.text_input("🔍 Search Query", "BGP neighbor down memory")
    k     = st.slider("Number of results", 3, 10, 5)

    if st.button("Search Knowledge Base"):
        with st.spinner("Running semantic search…"):
            data = api("get","/knowledge",params={"q":query,"k":k})
        if data:
            results = data.get("results",[])
            st.success(f"Found {len(results)} relevant documents")
            for i,r in enumerate(results,1):
                score_pct = int(r.get("score",0)*100)
                src_icon  = {"historical":"📜","sop":"📋","vendor_kb":"🏭","runbook":"📖"}.get(r["source"],"📄")
                with st.expander(f"{src_icon} Result {i} — {r['source'].upper()} | Relevance: {score_pct}%",
                                 expanded=i==1):
                    st.markdown(f"""<div class="agent-body">{r['content']}</div>
                    {conf_bar(score_pct)}""", unsafe_allow_html=True)

    # Show from last pipeline result too
    result = st.session_state.get("last_result")
    if result and result.get("knowledge"):
        st.markdown("<hr class='sec-divider'>", unsafe_allow_html=True)
        st.markdown("**Knowledge retrieved during last pipeline run:**")
        for r in result["knowledge"]:
            score_pct = int(r.get("score",0)*100)
            src_icon  = {"historical":"📜","sop":"📋","vendor_kb":"🏭","runbook":"📖"}.get(r.get("source",""),"📄")
            with st.expander(f"{src_icon} {r.get('source','').upper()} | {score_pct}% match"):
                st.text(r["content"])


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: SLA RISK MONITOR
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == "sla":
    render_header("🛡️","SLA Risk Monitor","Breach probability prediction and escalation alerts")

    result = st.session_state.get("last_result")
    if result:
        sla = result.get("sla_risk",{})
        prob = sla.get("breach_probability",0)
        risk = sla.get("sla_risk","Unknown")
        clr  = {"Low":"#00C853","Medium":"#FFD700","High":"#FF8C00","Critical":"#FF3B3B"}.get(risk,"#6B7280")

        c1,c2,c3 = st.columns(3)
        c1.markdown(f"""<div class="metric-card">
          <div class="metric-val" style="color:{clr}">{risk}</div>
          <div class="metric-label">SLA Risk Level</div>
        </div>""", unsafe_allow_html=True)
        c2.markdown(f"""<div class="metric-card">
          <div class="metric-val" style="color:{clr}">{prob}%</div>
          <div class="metric-label">Breach Probability</div>
        </div>""", unsafe_allow_html=True)
        c3.markdown(f"""<div class="metric-card">
          <div class="metric-val" style="color:#00E5FF">{sla.get('time_to_breach_minutes','—')}</div>
          <div class="metric-label">Minutes to Breach</div>
        </div>""", unsafe_allow_html=True)

        st.markdown(f"""<div class="agent-card" style="margin-top:20px">
          <div class="agent-header">
            <span class="agent-icon">🛡️</span>
            <span class="agent-name">SLA Risk Agent</span>
          </div>
          <div class="agent-body">
            <b>Breach Type:</b> {sla.get('breach_type','—')}<br>
            <b>Escalation Required:</b> {'⚠️ YES' if sla.get('escalation_required') else '✅ No'}<br>
            <b>Escalation Level:</b> {sla.get('escalation_level','—')}<br>
          </div>
          {conf_bar(prob)}
        </div>""", unsafe_allow_html=True)
    else:
        st.info("Submit an alarm first via **Manual Alarm Input** to see SLA risk assessment.")

    # Simulated SLA trend chart
    st.markdown("<hr class='sec-divider'>", unsafe_allow_html=True)
    st.markdown("**Simulated SLA Breach Probability Trend**")
    trend = [random.randint(30,95) for _ in range(24)]
    df_trend = pd.DataFrame({"Hour":list(range(24)),"Breach Probability %":trend})
    st.area_chart(df_trend.set_index("Hour"), color="#FF3B3B")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: REMEDIATION CENTER
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == "remediation":
    render_header("🔧","Remediation Center","AI-generated playbooks with human approval workflow")

    result = st.session_state.get("last_result")
    if not result:
        st.info("Submit an alarm via **Manual Alarm Input** to see remediation recommendations.")
    else:
        rem    = result.get("remediation",{})
        steps  = rem.get("steps",[])
        ticket = result.get("ticket",{})
        alarm_id = result.get("alarm_id","")

        c1,c2,c3 = st.columns(3)
        risk_clr = {"Low":"#00C853","Medium":"#FFD700","High":"#FF8C00"}.get(rem.get("risk_score",""),"#6B7280")
        c1.markdown(f"""<div class="metric-card">
          <div class="metric-val" style="color:{risk_clr}">{rem.get('risk_score','—')}</div>
          <div class="metric-label">Risk Score</div>
        </div>""", unsafe_allow_html=True)
        c2.markdown(f"""<div class="metric-card">
          <div class="metric-val" style="color:#00C853">{rem.get('success_probability','—')}%</div>
          <div class="metric-label">Success Probability</div>
        </div>""", unsafe_allow_html=True)
        c3.markdown(f"""<div class="metric-card">
          <div class="metric-val" style="color:#00E5FF">{rem.get('estimated_duration_minutes','—')}</div>
          <div class="metric-label">Est. Minutes</div>
        </div>""", unsafe_allow_html=True)

        st.markdown("**Remediation Steps**")
        for s in steps:
            auto_tag = '<span class="step-auto">AUTO</span>' if s.get("automated") else '<span class="step-manual">MANUAL</span>'
            cmd_block = f'<code style="background:#0A0E1A;padding:2px 8px;border-radius:3px;font-size:.78rem">{s["command"]}</code>' if s.get("command") else ""
            st.markdown(f"""<div class="step-item">
              <div class="step-num">{s.get('step_number','?')}</div>
              <div style="flex:1">
                <div style="font-size:.85rem">{s.get('action','')}</div>
                {cmd_block}
              </div>
              {auto_tag}
            </div>""", unsafe_allow_html=True)

        st.markdown(f"""<div style="margin-top:12px;font-size:.82rem;color:#6B7280">
          🔄 <b>Rollback Plan:</b> {rem.get('rollback_plan','—')}
        </div>""", unsafe_allow_html=True)

        st.markdown("<hr class='sec-divider'>", unsafe_allow_html=True)
        st.markdown("**🔐 Human Approval**")
        approved = result.get("ticket",{}).get("automation_used", False)

        if approved:
            st.success("✅ Remediation approved and automation executed.")
        else:
            st.warning("⏳ Awaiting human approval before automation executes.")
            if st.button(f"✅ Approve & Execute Automation for {alarm_id}"):
                with st.spinner("Approving and running automation…"):
                    r = api("post", f"/alarm/approve/{alarm_id}")
                if r:
                    st.success(f"Approved! Ticket: {r.get('ticket',{}).get('ticket_id','—')}")
                    result["ticket"] = r.get("ticket",{})
                    result["automation_result"] = r.get("automation_result",{})
                    st.session_state["last_result"] = result
                    st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: AUTOMATION EXECUTION
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == "automation":
    render_header("⚙️","Automation Execution","Simulated API command execution status")

    result = st.session_state.get("last_result")
    auto   = result.get("automation_result",{}) if result else {}

    if auto and auto.get("executed_steps"):
        overall = auto.get("overall_status","—")
        clr = "#00C853" if overall=="Success" else "#FF8C00"
        st.markdown(f"""<div class="metric-card" style="margin-bottom:20px;max-width:200px">
          <div class="metric-val" style="color:{clr}">{overall}</div>
          <div class="metric-label">Overall Status</div>
        </div>""", unsafe_allow_html=True)

        rows = ""
        for s in auto["executed_steps"]:
            st_clr = "#00C853" if s["status"]=="Success" else "#FF8C00"
            rows += (f"<tr><td>{s.get('step','—')}</td>"
                     f"<td>{s.get('action','—')}</td>"
                     f"<td><code style='font-size:.78rem'>{s.get('command') or '—'}</code></td>"
                     f"<td><span style='color:{st_clr};font-weight:700'>{s['status']}</span></td>"
                     f"<td style='color:#6B7280;font-size:.78rem'>{s.get('response','—')}</td></tr>")
        st.markdown(f"""<table class='noc-table'><thead><tr>
          <th>Step</th><th>Action</th><th>Command</th><th>Status</th><th>Response</th>
        </tr></thead><tbody>{rows}</tbody></table>""", unsafe_allow_html=True)
    else:
        st.info("No automation steps executed yet. Approve remediation on the Remediation Center page.")

    # Manual MCP tool execution
    st.markdown("<hr class='sec-divider'>", unsafe_allow_html=True)
    st.markdown("**🔧 Manual MCP Tool Execution**")
    t1,t2,t3 = st.tabs(["Router Tool","Log Search Tool","KPI Tool"])
    with t1:
        dev = st.selectbox("Device", ["RTR-CORE-01","RTR-EDGE-01","PE-RTR-01"])
        act = st.selectbox("Action", ["bgp_reset","interface_up","router_restart","cache_clear"])
        if st.button("Execute Router Tool"):
            r = api("get", f"/mcp/router/{dev}/{act}")
            if r: st.json(r)
    with t2:
        dev2 = st.selectbox("Device (logs)", ["RTR-CORE-01","RTR-EDGE-01"])
        q    = st.text_input("Log query","error")
        if st.button("Search Logs"):
            r = api("get", f"/mcp/logs/{dev2}", params={"q":q})
            if r: st.json(r)
    with t3:
        dev3 = st.selectbox("Device (KPI)", ["RTR-CORE-01","RTR-EDGE-01","PE-RTR-01"])
        if st.button("Fetch KPIs"):
            r = api("get", f"/mcp/kpi/{dev3}")
            if r: st.json(r)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: ITSM TICKETS
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == "itsm":
    render_header("🎫","ITSM Ticket Management","Automated incident ticket creation and tracking")

    if st.button("🔄 Refresh"):
        st.rerun()

    data    = api("get","/ticket",params={"limit":50})
    tickets = data["tickets"] if data else []

    if not tickets:
        st.info("No tickets yet. Process an alarm to auto-generate a ticket.")
    else:
        open_c     = sum(1 for t in tickets if t.get("status")=="Open")
        inprog_c   = sum(1 for t in tickets if t.get("status")=="In Progress")
        resolved_c = sum(1 for t in tickets if t.get("status")=="Resolved")

        c1,c2,c3 = st.columns(3)
        for col,(lbl,val,clr) in zip([c1,c2,c3],[
            ("Open",open_c,"#FF3B3B"),
            ("In Progress",inprog_c,"#FF8C00"),
            ("Resolved",resolved_c,"#00C853"),
        ]):
            col.markdown(f"""<div class="metric-card">
              <div class="metric-val" style="color:{clr}">{val}</div>
              <div class="metric-label">{lbl} Tickets</div>
            </div>""", unsafe_allow_html=True)

        rows = ""
        for t in tickets:
            st_  = t.get("status","Open")
            st_clr = STATUS_COLOR.get(st_,"#6B7280")
            auto = "🤖 Yes" if t.get("automation_used") else "👤 No"
            rows += (f"<tr><td style='font-weight:700;color:#00E5FF'>{t['ticket_id']}</td>"
                     f"<td>{t.get('device','—')}</td>"
                     f"<td>{badge(t.get('severity','Minor'))}</td>"
                     f"<td style='font-size:.78rem;max-width:180px'>{t.get('alarm','—')[:60]}</td>"
                     f"<td>{badge(st_)}</td>"
                     f"<td>{t.get('priority','—')}</td>"
                     f"<td>{auto}</td>"
                     f"<td>{ts_fmt(t.get('created_at',''))}</td></tr>")
        st.markdown(f"""<table class='noc-table'><thead><tr>
          <th>Ticket ID</th><th>Device</th><th>Severity</th><th>Alarm</th>
          <th>Status</th><th>Priority</th><th>Auto</th><th>Created</th>
        </tr></thead><tbody>{rows}</tbody></table>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: NETWORK TOPOLOGY
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == "topology":
    render_header("🌐","Network Topology","Live mock network topology visualization")

    data = api("get","/topology")
    if not data or "error" in data:
        st.error("Topology data not available")
    else:
        nodes = data.get("nodes",[])
        links = data.get("links",[])

        # Group nodes by type
        node_types = {}
        for n in nodes:
            t = n.get("type","Unknown")
            node_types.setdefault(t,[]).append(n["id"])

        type_color = {
            "Core Router": "#FF3B3B",
            "Edge Router": "#FF8C00",
            "PE Router":   "#7C3AED",
            "CE Router":   "#FFD700",
            "Aggregation Switch": "#00E5FF",
            "Firewall":    "#00C853",
            "Load Balancer":"#4FC3F7",
        }
        type_icon = {
            "Core Router":"🔴","Edge Router":"🟠","PE Router":"🟣",
            "CE Router":"🟡","Aggregation Switch":"🔵","Firewall":"🟢","Load Balancer":"🩵",
        }

        st.markdown("**Legend**")
        leg_html = ""
        for t,clr in type_color.items():
            leg_html += f'<span style="background:{clr}22;border:1px solid {clr};border-radius:4px;padding:3px 10px;font-size:.75rem;color:{clr};margin:3px;display:inline-block">{type_icon.get(t,"●")} {t}</span>'
        st.markdown(leg_html, unsafe_allow_html=True)

        st.markdown("<hr class='sec-divider'>", unsafe_allow_html=True)
        st.markdown("**Network Segments**")
        for t,ids in node_types.items():
            clr = type_color.get(t,"#6B7280")
            ico = type_icon.get(t,"●")
            nodes_html = "".join(f'<span class="topo-node" style="border-color:{clr};color:{clr}">{ico} {n}</span>' for n in ids)
            st.markdown(f"""<div style="margin-bottom:14px">
              <div style="font-size:.75rem;color:{clr};text-transform:uppercase;letter-spacing:.1em;margin-bottom:6px">{t}</div>
              {nodes_html}
            </div>""", unsafe_allow_html=True)

        st.markdown("<hr class='sec-divider'>", unsafe_allow_html=True)
        st.markdown("**Network Links**")
        rows = ""
        for lnk in links:
            lt   = lnk.get("type","—")
            lt_clr = {"Core":"#FF3B3B","Transit":"#FF8C00","MPLS":"#7C3AED",
                       "Access":"#00E5FF","Customer":"#FFD700","Security":"#00C853","DMZ":"#4FC3F7"}.get(lt,"#6B7280")
            rows += (f"<tr><td style='color:#00E5FF'>{lnk['source']}</td>"
                     f"<td style='color:#6B7280;text-align:center'>↔</td>"
                     f"<td style='color:#00E5FF'>{lnk['target']}</td>"
                     f"<td><span style='color:{lt_clr}'>{lt}</span></td>"
                     f"<td style='color:{lt_clr}'>{lnk.get('bandwidth','—')}</td></tr>")
        st.markdown(f"""<table class='noc-table'><thead><tr>
          <th>Source</th><th></th><th>Target</th><th>Link Type</th><th>Bandwidth</th>
        </tr></thead><tbody>{rows}</tbody></table>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: AGENT TRACE
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == "trace":
    render_header("📊","Agent Execution Trace","LangGraph pipeline trace and agent reasoning log")

    data   = api("get","/traces",params={"limit":20})
    traces = data["traces"] if data else []

    # Session result trace takes priority
    result = st.session_state.get("last_result")
    if result and result.get("trace"):
        st.markdown("**Last Pipeline Execution Trace**")
        agent_icons = {
            "AlarmIngestion":"🔔","EventCorrelation":"🔗","RootCauseAnalysis":"🔬",
            "KnowledgeRetrieval":"📚","SLARisk":"🛡️","Remediation":"🔧",
            "Automation":"⚙️","ITSM":"🎫",
        }
        for step in result["trace"]:
            agent = step.get("agent","Agent")
            icon  = agent_icons.get(agent,"🤖")
            ts    = ts_fmt(step.get("ts",""))
            with st.expander(f"{icon} {agent} — {ts}", expanded=False):
                st.json(step.get("output",{}))

        # Mermaid-style flow
        st.markdown("<hr class='sec-divider'>", unsafe_allow_html=True)
        st.markdown("**LangGraph Flow**")
        agents_in_trace = [s["agent"] for s in result["trace"]]
        flow_html = ""
        for i,a in enumerate(agents_in_trace):
            icon = agent_icons.get(a,"🤖")
            flow_html += f'<span class="topo-node" style="border-color:#00E5FF;color:#00E5FF">{icon} {a}</span>'
            if i < len(agents_in_trace)-1:
                flow_html += '<span class="topo-link">→</span>'
        st.markdown(f'<div style="margin:12px 0;flex-wrap:wrap;display:flex;align-items:center">{flow_html}</div>',
                    unsafe_allow_html=True)

    # Historical traces
    if traces:
        st.markdown("<hr class='sec-divider'>", unsafe_allow_html=True)
        st.markdown("**Historical Traces**")
        for t in traces:
            alarm  = t.get("alarm",{})
            n_steps = len(t.get("result",{}).get("trace",[]))
            ticket  = t.get("result",{}).get("ticket",{})
            with st.expander(f"🆔 {t.get('alarm_id','—')} — {alarm.get('alarm','—')} | {alarm.get('device','—')} | {n_steps} steps"):
                c1,c2 = st.columns(2)
                with c1:
                    st.markdown(f"**Severity:** {alarm.get('severity','—')}")
                    st.markdown(f"**Ticket:** {ticket.get('ticket_id','—')}")
                    st.markdown(f"**Status:** {ticket.get('status','—')}")
                with c2:
                    st.markdown(f"**Root Cause:** {t.get('result',{}).get('root_cause',{}).get('root_cause','—')}")
                    st.markdown(f"**SLA Risk:** {t.get('result',{}).get('sla_risk',{}).get('sla_risk','—')}")
    else:
        st.info("No historical traces yet. Process an alarm to generate traces.")