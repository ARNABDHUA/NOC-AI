"""
NOC Agentic Copilot - Backend
FastAPI + LangGraph + FAISS + 8 Agents
Run: uvicorn backend:app --host 0.0.0.0 --port 8000 --reload

FIX 1: Automation agent no longer randomly returns "Partial" — all automated steps succeed.
FIX 2: Remediation approval updates the EXISTING ticket instead of creating a new one.
FIX 3: Remediation agent uses alarm-aware fallback steps (not hardcoded BGP).
FIX 4: Manual alarm input now persists to alarms.json so it appears in the alarm list.
"""

import os, json, random, uuid, logging, httpx
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
# from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import FakeEmbeddings
from langchain_core.documents import Document
from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("noc_copilot")

# ═══════════════════════════════════════════════════════════════════════════════
# 1. LLM FACTORY
# ═══════════════════════════════════════════════════════════════════════════════
_http_client = httpx.Client(verify=False, timeout=120)

def get_llm():
    provider = os.environ.get("LLM_PROVIDER", "openai_custom")
    model    = os.environ.get("CUSTOM_MODEL_NAME", "Qwen3-30B-A3B")
    base_url = os.environ.get("BASE_URL", "https://afraid-aliens-read.loca.lt/v1")
    api_key  = os.environ.get("OPENAI_API_KEY", "abc-123")
    if provider == "openai_custom":
        return ChatOpenAI(
            model=model,
            openai_api_key=api_key,
            openai_api_base=base_url,
            temperature=0,
            request_timeout=120,
            max_retries=1,
            http_client=_http_client,
            default_headers={"Authorization": f"Bearer {api_key}"},
        )
    raise ValueError(f"Unknown LLM_PROVIDER: '{provider}'.")


# ═══════════════════════════════════════════════════════════════════════════════
# 2. SAMPLE DATA GENERATION
# ═══════════════════════════════════════════════════════════════════════════════
DATA_DIR = Path("noc_data")
DATA_DIR.mkdir(exist_ok=True)

DEVICES = [
    "RTR-CORE-01","RTR-CORE-02","RTR-EDGE-01","RTR-EDGE-02",
    "SW-AGG-01","SW-AGG-02","FIREWALL-01","FIREWALL-02",
    "LB-01","PE-RTR-01","PE-RTR-02","CE-RTR-01",
]
ALARM_TYPES = [
    ("BGP Neighbor Down",       "Critical"),
    ("OSPF Adjacency Failure",  "Critical"),
    ("Interface Flapping",      "Major"),
    ("High CPU Utilization",    "Major"),
    ("Memory Leak Detected",    "Critical"),
    ("Packet Loss > 5%",        "Major"),
    ("Fiber Cut Detected",      "Critical"),
    ("Core Router Failure",     "Critical"),
    ("Link Down",               "Critical"),
    ("MPLS Label Stack Error",  "Major"),
    ("NTP Sync Failed",         "Minor"),
    ("Spanning Tree Topology",  "Major"),
    ("DHCP Pool Exhausted",     "Minor"),
    ("Route Table Overflow",    "Critical"),
    ("Interface CRC Errors",    "Minor"),
]

def _rand_ts(days_back=7):
    return (datetime.utcnow() - timedelta(
        days=random.randint(0, days_back),
        hours=random.randint(0, 23),
        minutes=random.randint(0, 59),
    )).strftime("%Y-%m-%dT%H:%M:%SZ")

def _save_alarms():
    """Persist in-memory alarm store to disk."""
    (DATA_DIR / "alarms.json").write_text(
        json.dumps(_alarms_store, indent=2), encoding="utf-8"
    )

def generate_alarms(n=100):
    path = DATA_DIR / "alarms.json"
    alarms = []
    for i in range(n):
        atype, sev = random.choice(ALARM_TYPES)
        alarms.append({
            "alarm_id":  f"ALM-{2026000+i}",
            "device":    random.choice(DEVICES),
            "alarm":     atype,
            "severity":  sev,
            "timestamp": _rand_ts(),
            "status":    random.choice(["Open","Acknowledged","Resolved"]),
            "kpi": {
                "cpu":            random.randint(10, 99),
                "memory":         random.randint(20, 98),
                "packet_loss":    round(random.uniform(0, 20), 2),
                "latency_ms":     random.randint(1, 500),
                "bandwidth_util": random.randint(5, 99),
            },
        })
    path.write_text(json.dumps(alarms, indent=2), encoding="utf-8")
    log.info(f"Generated {n} alarms -> {path}")
    return alarms

def generate_incidents(alarms, n=50):
    path = DATA_DIR / "incidents.json"
    incidents = []
    for i in range(n):
        group = random.sample(alarms, k=random.randint(2, 5))
        devices = list({a["device"] for a in group})
        incidents.append({
            "incident_id":      f"INC-2026-{i+1:03d}",
            "group_id":         f"GRP-{i+1:03d}",
            "alarm_ids":        [a["alarm_id"] for a in group],
            "affected_devices": devices,
            "severity":         group[0]["severity"],
            "timestamp":        _rand_ts(14),
            "status":           random.choice(["Open","In Progress","Resolved"]),
            "root_cause":       random.choice([
                "BGP session reset due to memory pressure",
                "Fiber cut causing cascading link failures",
                "CPU spike from routing loop",
                "Interface flapping due to hardware fault",
                "Misconfigured BGP policy",
                "DDoS traffic flooding core interfaces",
                "Memory leak in routing daemon",
                "OSPF metric change causing rerouting storm",
            ]),
            "resolution": random.choice([
                "Restarted BGP daemon; verified neighbor sessions",
                "Rerouted traffic via backup fiber path",
                "Cleared routing table cache; applied rate-limit",
                "Replaced faulty SFP module on interface",
                "Rolled back BGP policy to previous version",
                "Applied ACL to drop DDoS traffic at edge",
                "Patched routing daemon; restarted process",
                "Adjusted OSPF cost; stabilised topology",
            ]),
        })
    path.write_text(json.dumps(incidents, indent=2), encoding="utf-8")
    return incidents

def generate_sop():
    sop = """SOP-001: BGP Neighbor Down
Symptoms: BGP session drops, routing table gaps, traffic blackholing.
Steps:
  1. Verify physical connectivity: ping neighbor IP, check interface status.
  2. Review BGP logs: journalctl -u bgpd | tail -100
  3. Check CPU & memory: show processes | include bgpd
  4. Clear BGP session: clear ip bgp <neighbor> soft
  5. If CPU>90%: restart routing daemon: systemctl restart bgpd
  6. Verify peers re-establish: show ip bgp summary
  7. Escalate to Tier-3 if not resolved in 15 minutes.

SOP-002: High CPU on Core Router
Symptoms: CPU utilization >90%, packet drops, sluggish response.
Steps:
  1. Identify top processes: show processes cpu sorted
  2. Check for routing loops: show ip route summary
  3. Apply CPU rate-limiting: router cpu-threshold 80
  4. Restart offending process if safe.
  5. Engage vendor TAC if hardware fault suspected.

SOP-003: Interface Flapping
Symptoms: Interface up/down cycles, log storms, instability.
Steps:
  1. Check physical layer: inspect SFP, fiber connectors, cables.
  2. Review error counters: show interface <intf> counters error
  3. Enable dampening: interface dampening half-life 5 reuse 1000 suppress 2000 max-suppress 20
  4. Replace faulty hardware if errors persist.

SOP-004: Fiber Cut / Link Down
Symptoms: Multiple link alarms, traffic rerouting, SLA breach risk.
Steps:
  1. Confirm cut via optical loss readings (OTDR).
  2. Activate backup path: ip route <prefix> <backup-next-hop>
  3. Notify NOC Bridge; engage field crew for physical repair.
  4. Monitor traffic switch-over via NetFlow.

SOP-005: Memory Leak
Symptoms: Memory utilization climbing, process restart logs.
Steps:
  1. Identify leaking process: show processes memory sorted
  2. Restart process in maintenance window if allowed.
  3. Apply vendor patch if available.
  4. Schedule hardware memory upgrade if recurrent.

SOP-006: Packet Loss > 5%
Symptoms: Customer complaints, QoS alerts, latency spikes.
Steps:
  1. Run traceroute to isolate loss point.
  2. Check interface error counters and queue drops.
  3. Apply traffic shaping or QoS policy adjustment.
  4. Escalate to backbone team if transit provider involved.
"""
    (DATA_DIR / "sop_documents.txt").write_text(sop, encoding="utf-8")
    return sop

def generate_vendor_kb():
    kb = """KB-001: Cisco IOS-XR BGP Memory Issue (CSCab12345)
Platform: Cisco ASR 9000 | IOS-XR 7.x
Issue: BGP process leaks memory when processing large route tables (>700k routes).
Fix: Upgrade to IOS-XR 7.5.2 or later. Workaround: restart bgp process every 24h.

KB-002: Juniper OSPF Adjacency Flap (PR1234567)
Platform: Juniper MX Series | JunOS 21.x
Issue: OSPF adjacency drops under heavy traffic due to hello timer drift.
Fix: Apply JunOS patch PR1234567. Workaround: increase dead interval to 40s.

KB-003: Nokia SR-OS Interface Error
Platform: Nokia 7750 SR | SR-OS 22.x
Issue: 100G interface reports CRC errors under sustained high traffic.
Fix: Replace QSFP28 module. Ensure firmware version >= 3.12.

KB-004: Huawei VRP Routing Table Overflow
Platform: Huawei NE40E | VRP V800R011
Issue: Route table exceeds 2M entries causing process crash.
Fix: Tune BGP route policy to filter unwanted prefixes. Upgrade to V800R021.

KB-005: Arista EOS MLAG Split-Brain
Platform: Arista 7050X | EOS 4.27
Issue: MLAG peer-link flap causes split-brain; dual active state.
Fix: Apply EOS 4.27.3F bugfix. Configure rapid recovery timers.
"""
    (DATA_DIR / "vendor_kb.txt").write_text(kb, encoding="utf-8")
    return kb

def generate_runbooks():
    rb = """RUNBOOK-001: BGP Recovery Procedure
Trigger: BGP Neighbor Down alarm (Critical)
Owner: NOC Tier-2
SLA: Restore within 15 minutes
Steps:
  1. [Auto] Ping neighbor - expect response
  2. [Auto] POST /bgp/reset {neighbor: <ip>}
  3. [Manual] Verify BGP summary shows Established
  4. [Auto] POST /ticket/create {priority: P1}
  5. [Manual] Confirm with customer if traffic restored

RUNBOOK-002: Core Router Failover
Trigger: Core Router Failure alarm (Critical)
Owner: NOC Tier-3
SLA: Restore within 30 minutes
Steps:
  1. [Auto] Verify standby router readiness
  2. [Manual] Execute failover: router hsrp preempt
  3. [Auto] POST /interface/up {device: backup}
  4. [Auto] Notify customer via NMS
  5. [Manual] Root cause analysis within 2 hours

RUNBOOK-003: DDoS Mitigation
Trigger: Packet Loss > 5% + Traffic spike on edge
Owner: Security NOC
SLA: Mitigate within 10 minutes
Steps:
  1. [Auto] Identify attack vector via flow analysis
  2. [Auto] Apply edge ACL: POST /acl/block {src: <ip>}
  3. [Manual] Activate scrubbing center
  4. [Auto] Monitor drop in attack traffic
  5. [Manual] Post-incident report within 4 hours
"""
    (DATA_DIR / "runbooks.txt").write_text(rb, encoding="utf-8")
    return rb

def generate_historical_incidents():
    hist = [
        {"incident":"BGP Neighbor Down","root_cause":"Memory Leak in BGP process","resolution":"Restart BGP Daemon; apply vendor patch","device":"RTR-CORE-01","severity":"Critical"},
        {"incident":"Packet Loss > 5%","root_cause":"Interface Queue Congestion","resolution":"Clear interface queue; apply QoS policy","device":"RTR-EDGE-01","severity":"Major"},
        {"incident":"Core Router Failure","root_cause":"Hardware PSU failure","resolution":"Replace PSU; failover to secondary router","device":"RTR-CORE-02","severity":"Critical"},
        {"incident":"OSPF Adjacency Failure","root_cause":"MTU mismatch on peer link","resolution":"Set MTU to 9000 on both sides","device":"SW-AGG-01","severity":"Major"},
        {"incident":"High CPU Utilization","root_cause":"Routing loop due to misconfigured BGP policy","resolution":"Remove incorrect route-map; clear BGP","device":"PE-RTR-01","severity":"Major"},
        {"incident":"Fiber Cut","root_cause":"Physical fiber damage in underground duct","resolution":"Rerouted via backup path; dispatched repair crew","device":"RTR-EDGE-02","severity":"Critical"},
        {"incident":"Interface Flapping","root_cause":"Faulty SFP transceiver","resolution":"Replace SFP module; test with BERT","device":"CE-RTR-01","severity":"Major"},
        {"incident":"Memory Leak","root_cause":"Routing daemon software bug","resolution":"Upgrade to patched version; monitor for 24h","device":"FIREWALL-01","severity":"Critical"},
        {"incident":"BGP Route Table Overflow","root_cause":"Full internet table accepted from transit peer","resolution":"Apply max-prefix limit; filter bogons","device":"PE-RTR-02","severity":"Critical"},
        {"incident":"MPLS Label Stack Error","root_cause":"P-router misconfiguration after maintenance","resolution":"Restore MPLS config from backup; verify LSPs","device":"RTR-CORE-01","severity":"Major"},
    ]
    path = DATA_DIR / "historical_incidents.json"
    path.write_text(json.dumps(hist, indent=2), encoding="utf-8")
    return hist

def generate_topology():
    topo = {
        "nodes": [
            {"id": d, "type": t, "region": r}
            for d, t, r in [
                ("RTR-CORE-01","Core Router","DC-North"),
                ("RTR-CORE-02","Core Router","DC-South"),
                ("RTR-EDGE-01","Edge Router","POP-East"),
                ("RTR-EDGE-02","Edge Router","POP-West"),
                ("SW-AGG-01","Aggregation Switch","DC-North"),
                ("SW-AGG-02","Aggregation Switch","DC-South"),
                ("PE-RTR-01","PE Router","POP-East"),
                ("PE-RTR-02","PE Router","POP-West"),
                ("CE-RTR-01","CE Router","Customer-Site"),
                ("FIREWALL-01","Firewall","DC-North"),
                ("LB-01","Load Balancer","DC-North"),
            ]
        ],
        "links": [
            {"source":"RTR-CORE-01","target":"RTR-CORE-02","type":"Core","bandwidth":"100G"},
            {"source":"RTR-CORE-01","target":"RTR-EDGE-01","type":"Transit","bandwidth":"40G"},
            {"source":"RTR-CORE-01","target":"RTR-EDGE-02","type":"Transit","bandwidth":"40G"},
            {"source":"RTR-CORE-02","target":"RTR-EDGE-01","type":"Transit","bandwidth":"40G"},
            {"source":"RTR-CORE-02","target":"RTR-EDGE-02","type":"Transit","bandwidth":"40G"},
            {"source":"RTR-CORE-01","target":"SW-AGG-01","type":"Access","bandwidth":"10G"},
            {"source":"RTR-CORE-02","target":"SW-AGG-02","type":"Access","bandwidth":"10G"},
            {"source":"RTR-EDGE-01","target":"PE-RTR-01","type":"MPLS","bandwidth":"10G"},
            {"source":"RTR-EDGE-02","target":"PE-RTR-02","type":"MPLS","bandwidth":"10G"},
            {"source":"PE-RTR-01","target":"CE-RTR-01","type":"Customer","bandwidth":"1G"},
            {"source":"SW-AGG-01","target":"FIREWALL-01","type":"Security","bandwidth":"10G"},
            {"source":"FIREWALL-01","target":"LB-01","type":"DMZ","bandwidth":"10G"},
        ],
    }
    (DATA_DIR / "topology.json").write_text(json.dumps(topo, indent=2), encoding="utf-8")
    return topo


# ═══════════════════════════════════════════════════════════════════════════════
# 3. FAISS VECTOR STORE
# ═══════════════════════════════════════════════════════════════════════════════
_vector_store: Optional[FAISS] = None

def _safe_embeddings():
    key  = os.environ.get("OPENAI_API_KEY", "")
    base = os.environ.get("BASE_URL", "")
    if key:
        try:
            return OpenAIEmbeddings(
                openai_api_key=key,
                openai_api_base=base or None,
                http_client=_http_client,
            )
        except Exception:
            pass
    log.warning("No valid embedding key — using FakeEmbeddings")
    return FakeEmbeddings(size=768)

def build_vector_store():
    global _vector_store
    emb  = _safe_embeddings()
    docs = []
    hist_path = DATA_DIR / "historical_incidents.json"
    if hist_path.exists():
        for h in json.loads(hist_path.read_text(encoding="utf-8")):
            text = (f"Incident: {h['incident']}\n"
                    f"Root Cause: {h['root_cause']}\n"
                    f"Resolution: {h['resolution']}")
            docs.append(Document(page_content=text, metadata={"source":"historical"}))
    for fname, src in [("sop_documents.txt","sop"),("vendor_kb.txt","vendor_kb"),("runbooks.txt","runbook")]:
        p = DATA_DIR / fname
        if p.exists():
            for chunk in p.read_text(encoding="utf-8").split("\n\n"):
                if len(chunk.strip()) > 30:
                    docs.append(Document(page_content=chunk.strip(), metadata={"source":src}))
    if not docs:
        docs = [Document(page_content="NOC placeholder document")]
    _vector_store = FAISS.from_documents(docs, emb)
    log.info(f"FAISS built with {len(docs)} documents")
    return _vector_store

def retrieve_knowledge(query: str, k: int = 4) -> list[dict]:
    if _vector_store is None:
        return [{"content":"Vector store not initialised","source":"none","score":0}]
    results = _vector_store.similarity_search_with_score(query, k=k)
    return [{"content": doc.page_content,
             "source":  doc.metadata.get("source","unknown"),
             "score":   float(round(1 - score/2, 3))} for doc, score in results]


# ═══════════════════════════════════════════════════════════════════════════════
# 4. IN-MEMORY STATE STORES
# ═══════════════════════════════════════════════════════════════════════════════
_alarms_store:    list[dict] = []
_incidents_store: list[dict] = []
_tickets_store:   list[dict] = []
_traces_store:    list[dict] = []

def _load_stores():
    global _alarms_store, _incidents_store
    _alarms_store    = json.loads((DATA_DIR/"alarms.json").read_text(encoding="utf-8"))
    _incidents_store = json.loads((DATA_DIR/"incidents.json").read_text(encoding="utf-8"))


# ═══════════════════════════════════════════════════════════════════════════════
# 5. LANGGRAPH STATE
# ═══════════════════════════════════════════════════════════════════════════════
class NOCState(TypedDict):
    alarm:             dict
    normalized_alarm:  Optional[dict]
    correlation:       Optional[dict]
    root_cause:        Optional[dict]
    knowledge:         Optional[list]
    sla_risk:          Optional[dict]
    remediation:       Optional[dict]
    automation_result: Optional[dict]
    ticket:            Optional[dict]
    human_approved:    bool
    trace:             list[dict]


def _llm_call(system: str, user: str) -> str:
    try:
        llm  = get_llm()
        resp = llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
        return resp.content
    except Exception as e:
        log.warning(f"LLM call failed: {e}")
        return json.dumps({"error": str(e), "fallback": True})


# ═══════════════════════════════════════════════════════════════════════════════
# 5b. ALARM-AWARE REMEDIATION FALLBACK
# ═══════════════════════════════════════════════════════════════════════════════
def _get_fallback_remediation(alarm_type: str, kpi: dict) -> dict:
    a = alarm_type.upper()

    if "MEMORY" in a or "LEAK" in a:
        return {
            "steps": [
                {"step_number":1,"action":"Identify top memory-consuming processes","command":"show processes memory sorted","automated":True},
                {"step_number":2,"action":"Check memory utilization trend over last hour","command":"show memory statistics history","automated":True},
                {"step_number":3,"action":"Restart leaking routing daemon (schedule maintenance window)","command":"systemctl restart routing-daemon","automated":False},
                {"step_number":4,"action":"Verify memory stabilizes after restart","command":"show processes memory sorted","automated":True},
                {"step_number":5,"action":"Apply vendor patch for memory leak if available","command":None,"automated":False},
                {"step_number":6,"action":"Schedule hardware memory upgrade if issue recurs","command":None,"automated":False},
            ],
            "risk_score":"High","success_probability":82,"estimated_duration_minutes":25,
            "rollback_plan":"Restore previous daemon version from backup; revert patch if applied",
        }
    elif "BGP" in a:
        return {
            "steps": [
                {"step_number":1,"action":"Verify BGP neighbor reachability","command":"ping <neighbor_ip>","automated":True},
                {"step_number":2,"action":"Clear BGP session soft reset","command":"clear ip bgp <neighbor> soft","automated":True},
                {"step_number":3,"action":"Check BGP logs for error cause","command":"show logging | include BGP","automated":True},
                {"step_number":4,"action":"Restart BGP daemon if session not re-established","command":"systemctl restart bgpd","automated":False},
                {"step_number":5,"action":"Verify route table restoration","command":"show ip bgp summary","automated":True},
                {"step_number":6,"action":"Confirm with customer that traffic is restored","command":None,"automated":False},
            ],
            "risk_score":"Medium","success_probability":87,"estimated_duration_minutes":15,
            "rollback_plan":"Restore BGP config from last known good backup",
        }
    elif "OSPF" in a:
        return {
            "steps": [
                {"step_number":1,"action":"Check OSPF neighbor state","command":"show ip ospf neighbor","automated":True},
                {"step_number":2,"action":"Verify MTU consistency on peer links","command":"show interface <intf> | include MTU","automated":True},
                {"step_number":3,"action":"Increase OSPF dead interval to 40s if hello drift suspected","command":"ip ospf dead-interval 40","automated":False},
                {"step_number":4,"action":"Clear OSPF process to force re-adjacency","command":"clear ip ospf process","automated":False},
                {"step_number":5,"action":"Verify adjacency re-established","command":"show ip ospf neighbor","automated":True},
            ],
            "risk_score":"Medium","success_probability":85,"estimated_duration_minutes":20,
            "rollback_plan":"Revert dead-interval to default; restore OSPF config from backup",
        }
    elif "CPU" in a:
        return {
            "steps": [
                {"step_number":1,"action":"Identify top CPU-consuming processes","command":"show processes cpu sorted","automated":True},
                {"step_number":2,"action":"Check for routing loops in route table","command":"show ip route summary","automated":True},
                {"step_number":3,"action":"Apply CPU rate-limiting threshold","command":"router cpu-threshold 80","automated":False},
                {"step_number":4,"action":"Restart offending process if safe","command":"systemctl restart <process>","automated":False},
                {"step_number":5,"action":"Engage vendor TAC if hardware fault suspected","command":None,"automated":False},
            ],
            "risk_score":"High","success_probability":78,"estimated_duration_minutes":30,
            "rollback_plan":"Remove CPU rate-limit config if it causes traffic issues",
        }
    elif "INTERFACE" in a or "FLAPPING" in a or "CRC" in a:
        return {
            "steps": [
                {"step_number":1,"action":"Check interface error counters","command":"show interface <intf> counters error","automated":True},
                {"step_number":2,"action":"Inspect SFP and fiber connectors physically","command":None,"automated":False},
                {"step_number":3,"action":"Enable interface dampening to reduce log storms","command":"interface dampening half-life 5 reuse 1000 suppress 2000 max-suppress 20","automated":False},
                {"step_number":4,"action":"Replace faulty SFP transceiver if errors persist","command":None,"automated":False},
                {"step_number":5,"action":"Run BERT test after SFP replacement","command":"test interface bert <intf>","automated":True},
            ],
            "risk_score":"Medium","success_probability":83,"estimated_duration_minutes":45,
            "rollback_plan":"Remove dampening config; restore original interface settings",
        }
    elif "LINK DOWN" in a or "FIBER" in a:
        return {
            "steps": [
                {"step_number":1,"action":"Confirm link down via optical loss readings (OTDR)","command":"show interface <intf> transceiver","automated":True},
                {"step_number":2,"action":"Activate backup path immediately","command":"ip route <prefix> <backup-next-hop>","automated":False},
                {"step_number":3,"action":"Monitor traffic switchover via NetFlow","command":"show ip flow top-talkers","automated":True},
                {"step_number":4,"action":"Notify NOC Bridge and dispatch field crew for repair","command":None,"automated":False},
                {"step_number":5,"action":"Restore primary path after physical repair confirmed","command":"no ip route <prefix> <backup-next-hop>","automated":False},
            ],
            "risk_score":"High","success_probability":75,"estimated_duration_minutes":60,
            "rollback_plan":"Revert to backup path if primary repair fails; escalate to Tier-3",
        }
    elif "PACKET" in a or "LOSS" in a:
        return {
            "steps": [
                {"step_number":1,"action":"Run traceroute to isolate packet loss point","command":"traceroute <destination> source <local-ip>","automated":True},
                {"step_number":2,"action":"Check interface error counters and queue drops","command":"show interface counters error","automated":True},
                {"step_number":3,"action":"Review QoS queue statistics","command":"show policy-map interface","automated":True},
                {"step_number":4,"action":"Apply traffic shaping or QoS policy adjustment","command":"policy-map QOS_REMEDIATION","automated":False},
                {"step_number":5,"action":"Escalate to backbone team if transit provider involved","command":None,"automated":False},
            ],
            "risk_score":"Medium","success_probability":80,"estimated_duration_minutes":20,
            "rollback_plan":"Remove QoS policy if it causes further traffic degradation",
        }
    elif "CORE ROUTER" in a or "ROUTER FAILURE" in a:
        return {
            "steps": [
                {"step_number":1,"action":"Verify standby router readiness and sync state","command":"show redundancy state","automated":True},
                {"step_number":2,"action":"Execute HSRP/VRRP failover to standby","command":"router hsrp preempt","automated":False},
                {"step_number":3,"action":"Bring up backup interfaces","command":"interface <backup-intf> no shutdown","automated":True},
                {"step_number":4,"action":"Notify customers via NMS alert","command":None,"automated":False},
                {"step_number":5,"action":"Conduct root cause analysis within 2 hours","command":None,"automated":False},
            ],
            "risk_score":"High","success_probability":70,"estimated_duration_minutes":30,
            "rollback_plan":"Revert failover if standby router is unstable; engage vendor TAC",
        }
    elif "MPLS" in a:
        return {
            "steps": [
                {"step_number":1,"action":"Verify MPLS label switched paths (LSPs)","command":"show mpls ldp neighbor","automated":True},
                {"step_number":2,"action":"Check MPLS forwarding table","command":"show mpls forwarding-table","automated":True},
                {"step_number":3,"action":"Restore MPLS config from last known good backup","command":None,"automated":False},
                {"step_number":4,"action":"Re-verify LSPs after config restore","command":"show mpls ldp neighbor","automated":True},
                {"step_number":5,"action":"Escalate to MPLS team if LSPs not restored","command":None,"automated":False},
            ],
            "risk_score":"High","success_probability":79,"estimated_duration_minutes":35,
            "rollback_plan":"Roll back MPLS config changes to previous state",
        }
    elif "ROUTE" in a and "OVERFLOW" in a:
        return {
            "steps": [
                {"step_number":1,"action":"Check current route table size","command":"show ip route summary","automated":True},
                {"step_number":2,"action":"Apply max-prefix limit on BGP peers","command":"neighbor <peer> maximum-prefix 750000","automated":False},
                {"step_number":3,"action":"Filter bogon routes from transit peers","command":"ip prefix-list BOGONS deny <bogon-prefix>","automated":False},
                {"step_number":4,"action":"Verify route table size reduced","command":"show ip route summary","automated":True},
            ],
            "risk_score":"High","success_probability":85,"estimated_duration_minutes":20,
            "rollback_plan":"Remove max-prefix limit if legitimate routes are being filtered",
        }
    elif "NTP" in a:
        return {
            "steps": [
                {"step_number":1,"action":"Check NTP synchronization status","command":"show ntp status","automated":True},
                {"step_number":2,"action":"Verify NTP server reachability","command":"ping <ntp-server-ip>","automated":True},
                {"step_number":3,"action":"Re-configure NTP server if unreachable","command":"ntp server <backup-ntp-ip>","automated":False},
                {"step_number":4,"action":"Force NTP re-sync","command":"ntp update-calendar","automated":True},
            ],
            "risk_score":"Low","success_probability":92,"estimated_duration_minutes":10,
            "rollback_plan":"Revert to previous NTP server configuration",
        }
    elif "DHCP" in a:
        return {
            "steps": [
                {"step_number":1,"action":"Check DHCP pool utilization","command":"show ip dhcp pool","automated":True},
                {"step_number":2,"action":"Identify and release stale DHCP bindings","command":"clear ip dhcp binding *","automated":False},
                {"step_number":3,"action":"Expand DHCP pool range if stale release insufficient","command":"ip dhcp pool <pool-name> network <new-range>","automated":False},
                {"step_number":4,"action":"Verify DHCP pool has available addresses","command":"show ip dhcp pool","automated":True},
            ],
            "risk_score":"Medium","success_probability":88,"estimated_duration_minutes":15,
            "rollback_plan":"Revert DHCP pool changes; restore original range",
        }
    elif "SPANNING" in a or "STP" in a:
        return {
            "steps": [
                {"step_number":1,"action":"Check spanning tree topology changes","command":"show spanning-tree detail","automated":True},
                {"step_number":2,"action":"Identify root bridge and verify correctness","command":"show spanning-tree root","automated":True},
                {"step_number":3,"action":"Enable BPDU guard on access ports","command":"spanning-tree portfast bpduguard default","automated":False},
                {"step_number":4,"action":"Configure portfast on access ports","command":"spanning-tree portfast","automated":False},
                {"step_number":5,"action":"Verify topology stabilizes","command":"show spanning-tree detail","automated":True},
            ],
            "risk_score":"Medium","success_probability":84,"estimated_duration_minutes":25,
            "rollback_plan":"Remove BPDU guard and portfast if it causes connectivity issues",
        }
    else:
        cpu = kpi.get("cpu", 0)
        mem = kpi.get("memory", 0)
        return {
            "steps": [
                {"step_number":1,"action":f"Investigate '{alarm_type}' — collect device logs","command":"show log | tail -200","automated":True},
                {"step_number":2,"action":"Collect full diagnostics bundle","command":"show tech-support","automated":True},
                {"step_number":3,"action":f"Check CPU ({cpu}%) and Memory ({mem}%) utilization","command":"show processes cpu sorted","automated":True},
                {"step_number":4,"action":"Escalate to Tier-2 NOC with diagnostics","command":None,"automated":False},
                {"step_number":5,"action":"Engage vendor TAC if issue persists beyond 30 minutes","command":None,"automated":False},
            ],
            "risk_score":"Medium","success_probability":70,"estimated_duration_minutes":30,
            "rollback_plan":"Restore device config from last known good backup",
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 6. AGENTS
# ═══════════════════════════════════════════════════════════════════════════════

def alarm_ingestion_agent(state: NOCState) -> NOCState:
    alarm  = state["alarm"]
    prompt = (f"You are a Telecom NOC Alarm Ingestion Agent.\n"
              f"Normalize this alarm, remove noise, classify severity and return ONLY valid JSON.\n"
              f"Input alarm: {json.dumps(alarm)}\n"
              f"Return JSON with keys: normalized_alarm, severity, confidence (0-100), category, affected_service")
    raw = _llm_call("You normalize telecom alarms. Return ONLY JSON, no markdown.", prompt)
    try:
        norm = json.loads(raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip())
    except Exception:
        norm = {
            "normalized_alarm": alarm.get("alarm","Unknown"),
            "severity":         alarm.get("severity","Unknown"),
            "confidence":       85,
            "category":         "Routing" if "BGP" in alarm.get("alarm","") else "General",
            "affected_service": "Network Connectivity",
        }
    state["normalized_alarm"] = norm
    state["trace"].append({"agent":"AlarmIngestion","output":norm,"ts":datetime.utcnow().isoformat()})
    return state


def event_correlation_agent(state: NOCState) -> NOCState:
    alarm   = state["normalized_alarm"] or state["alarm"]
    device  = state["alarm"].get("device","")
    related = [a for a in _alarms_store if a.get("device") == device][:5]
    prompt  = (f"You are a Telecom Event Correlation Agent.\n"
               f"Primary alarm: {json.dumps(alarm)}\n"
               f"Related alarms on same device: {json.dumps(related)}\n"
               f"Return ONLY JSON with keys:\n"
               f"group_id, affected_devices (list), correlated_alarms (list of alarm types),\n"
               f"cascading_failure (bool), confidence (0-100), incident_type")
    raw = _llm_call("You correlate telecom events. Return ONLY JSON.", prompt)
    try:
        corr = json.loads(raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip())
    except Exception:
        corr = {
            "group_id":          f"GRP-{uuid.uuid4().hex[:6].upper()}",
            "affected_devices":  [device],
            "correlated_alarms": [a["alarm"] for a in related],
            "cascading_failure": len(related) > 2,
            "confidence":        88,
            "incident_type":     "Network Instability",
        }
    state["correlation"] = corr
    state["trace"].append({"agent":"EventCorrelation","output":corr,"ts":datetime.utcnow().isoformat()})
    return state


def root_cause_agent(state: NOCState) -> NOCState:
    alarm      = state["normalized_alarm"] or state["alarm"]
    alarm_type = state["alarm"].get("alarm","")
    kpi        = state["alarm"].get("kpi", {})
    prompt     = (f"You are a Telecom Root Cause Analysis (RCA) Agent.\n"
                  f"Alarm: {json.dumps(alarm)}\nKPI metrics: {json.dumps(kpi)}\n"
                  f"Correlation data: {json.dumps(state.get('correlation', {}))}\n"
                  f"Return ONLY JSON with keys:\n"
                  f"root_cause (string), contributing_factors (list), confidence (0-100),\n"
                  f"impact_scope (Local/Regional/National), estimated_recovery_time_minutes (int)")
    raw = _llm_call("You perform root cause analysis. Return ONLY JSON.", prompt)
    try:
        rca = json.loads(raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip())
    except Exception:
        cpu = kpi.get("cpu", 0)
        mem = kpi.get("memory", 0)
        a   = alarm_type.upper()
        if "MEMORY" in a or "LEAK" in a:
            root = "Memory leak in routing daemon causing gradual memory exhaustion"
        elif "BGP" in a:
            root = "BGP session dropped due to keepalive timeout or memory pressure"
        elif "CPU" in a:
            root = "CPU overload caused by routing loop or high-volume traffic processing"
        elif "PACKET" in a or "LOSS" in a:
            root = "Interface queue congestion causing packet drops"
        elif "INTERFACE" in a or "FLAPPING" in a or "CRC" in a:
            root = "Faulty SFP transceiver or fiber causing repeated link state changes"
        elif "FIBER" in a or "LINK DOWN" in a:
            root = "Physical fiber damage or connector fault causing link loss"
        elif "OSPF" in a:
            root = "OSPF adjacency lost due to MTU mismatch or hello timer drift"
        elif "MPLS" in a:
            root = "MPLS misconfiguration or label stack corruption after maintenance"
        elif "ROUTE" in a:
            root = "Routing table overflow from unconstrained prefix acceptance"
        elif "NTP" in a:
            root = "NTP server unreachable causing clock drift on device"
        elif "DHCP" in a:
            root = "DHCP pool exhausted due to stale leases or subnet undersizing"
        elif "SPANNING" in a or "STP" in a:
            root = "Spanning tree topology change causing brief forwarding instability"
        elif "CORE ROUTER" in a:
            root = "Core router hardware failure — PSU or board-level fault"
        else:
            root = ("Memory leak causing routing daemon instability" if mem > 85
                    else "CPU overload causing packet drops" if cpu > 90
                    else f"Network fault: {alarm_type}")
        rca = {
            "root_cause":        root,
            "contributing_factors": [f"CPU:{cpu}%", f"Memory:{mem}%",
                                     f"Packet Loss:{kpi.get('packet_loss',0)}%"],
            "confidence":        87,
            "impact_scope":      "Regional",
            "estimated_recovery_time_minutes": 20,
        }
    state["root_cause"] = rca
    state["trace"].append({"agent":"RootCauseAnalysis","output":rca,"ts":datetime.utcnow().isoformat()})
    return state


def knowledge_retrieval_agent(state: NOCState) -> NOCState:
    alarm   = state["normalized_alarm"] or state["alarm"]
    query   = f"{alarm.get('normalized_alarm', alarm.get('alarm',''))} {alarm.get('category','')}"
    results = retrieve_knowledge(query, k=5)
    state["knowledge"] = results
    state["trace"].append({"agent":"KnowledgeRetrieval","output":results,"ts":datetime.utcnow().isoformat()})
    return state


def sla_risk_agent(state: NOCState) -> NOCState:
    alarm  = state["normalized_alarm"] or state["alarm"]
    rca    = state.get("root_cause", {})
    prompt = (f"You are a Telecom SLA Risk Assessment Agent.\n"
              f"Alarm: {json.dumps(alarm)}\nRoot cause: {json.dumps(rca)}\n"
              f"Return ONLY JSON with keys:\n"
              f"sla_risk (Low/Medium/High/Critical), breach_probability (0-100),\n"
              f"breach_type (Availability/Latency/Packet_Loss), time_to_breach_minutes (int),\n"
              f"escalation_required (bool), escalation_level (NOC-L1/NOC-L2/NOC-L3/Management)")
    raw = _llm_call("You assess SLA breach risk. Return ONLY JSON.", prompt)
    try:
        sla = json.loads(raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip())
    except Exception:
        sev  = alarm.get("severity","Minor")
        prob = {"Critical":91,"Major":72,"Minor":35,"Warning":15}.get(sev, 50)
        sla  = {
            "sla_risk":               "High" if prob > 70 else "Medium" if prob > 40 else "Low",
            "breach_probability":     prob,
            "breach_type":            "Availability",
            "time_to_breach_minutes": 15 if prob > 70 else 60,
            "escalation_required":    prob > 70,
            "escalation_level":       "NOC-L2" if prob > 70 else "NOC-L1",
        }
    state["sla_risk"] = sla
    state["trace"].append({"agent":"SLARisk","output":sla,"ts":datetime.utcnow().isoformat()})
    return state


def parallel_analysis(state: NOCState) -> NOCState:
    state = root_cause_agent(state)
    state = knowledge_retrieval_agent(state)
    state = sla_risk_agent(state)
    return state


def remediation_agent(state: NOCState) -> NOCState:
    alarm_type = state["alarm"].get("alarm", "")
    alarm      = state["normalized_alarm"] or state["alarm"]
    kpi        = state["alarm"].get("kpi", {})
    rca        = state.get("root_cause", {})
    kb         = state.get("knowledge", [])
    kb_text    = "\n".join([k.get("content","") for k in kb[:3]])

    prompt = (f"You are a Telecom Remediation Planning Agent.\n"
              f"Alarm Type: {alarm_type}\n"
              f"Alarm Details: {json.dumps(alarm)}\n"
              f"Root Cause: {json.dumps(rca)}\n"
              f"KPI Metrics: {json.dumps(kpi)}\n"
              f"Relevant KB/SOP:\n{kb_text}\n\n"
              f"Generate a detailed remediation plan SPECIFIC to the alarm type '{alarm_type}'.\n"
              f"Do NOT generate BGP steps unless this is a BGP alarm.\n"
              f"Return ONLY JSON with keys:\n"
              f"steps (list: step_number int, action str, command str or null, automated bool),\n"
              f"risk_score (Low/Medium/High), success_probability (0-100),\n"
              f"estimated_duration_minutes (int), rollback_plan (str)")

    raw = _llm_call("You generate remediation plans specific to the alarm type. Return ONLY JSON.", prompt)
    rem = None
    try:
        clean = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        parsed = json.loads(clean)
        if parsed.get("steps") and len(parsed["steps"]) > 0:
            rem = parsed
        else:
            raise ValueError("LLM returned empty steps list")
    except Exception as e:
        log.warning(f"Remediation fallback triggered for '{alarm_type}': {e}")
        rem = _get_fallback_remediation(alarm_type, kpi)

    state["remediation"]    = rem
    state["human_approved"] = False
    state["trace"].append({"agent":"Remediation","output":rem,"ts":datetime.utcnow().isoformat()})
    return state


def automation_agent(state: NOCState) -> NOCState:
    if not state.get("human_approved", False):
        result = {"status":"Skipped","reason":"Awaiting human approval"}
        state["automation_result"] = result
        state["trace"].append({"agent":"Automation","output":result,"ts":datetime.utcnow().isoformat()})
        return state

    steps      = state.get("remediation", {}).get("steps", [])
    auto_steps = [s for s in steps if s.get("automated", False)]
    results    = []
    for s in auto_steps:
        results.append({
            "step":     s.get("step_number"),
            "action":   s.get("action"),
            "command":  s.get("command"),
            "status":   "Success",
            "response": f"Executed at {datetime.utcnow().strftime('%H:%M:%SZ')}",
        })
    auto_result = {
        "executed_steps": results,
        "overall_status": "Success",
        "timestamp":      datetime.utcnow().isoformat(),
    }
    state["automation_result"] = auto_result
    state["trace"].append({"agent":"Automation","output":auto_result,"ts":datetime.utcnow().isoformat()})
    return state


def itsm_agent(state: NOCState) -> NOCState:
    alarm     = state["alarm"]
    rca       = state.get("root_cause", {})
    rem       = state.get("remediation", {})
    approved  = state.get("human_approved", False)

    alarm_id      = alarm.get("alarm_id", "")
    existing_ticket = next(
        (t for t in _tickets_store if t.get("alarm_id") == alarm_id),
        None
    )

    if existing_ticket and approved:
        existing_ticket["status"]         = "Resolved"
        existing_ticket["resolved_at"]    = datetime.utcnow().isoformat()
        existing_ticket["automation_used"] = True
        existing_ticket["resolution"]     = rem.get("rollback_plan", "Automated remediation applied")
        ticket = existing_ticket
        log.info(f"ITSM: Updated existing ticket {ticket['ticket_id']} → Resolved")
    else:
        ticket_id = f"INC-{datetime.utcnow().year}-{random.randint(1000,9999)}"
        ticket = {
            "ticket_id":       ticket_id,
            "alarm_id":        alarm_id,
            "device":          alarm.get("device","Unknown"),
            "alarm":           alarm.get("alarm","Unknown"),
            "severity":        alarm.get("severity","Unknown"),
            "root_cause":      rca.get("root_cause","Pending RCA"),
            "status":          "Resolved" if approved else "In Progress",
            "priority":        "P1" if alarm.get("severity")=="Critical" else "P2",
            "sla_risk":        state.get("sla_risk",{}).get("sla_risk","Unknown"),
            "created_at":      datetime.utcnow().isoformat(),
            "resolved_at":     datetime.utcnow().isoformat() if approved else None,
            "resolution":      rem.get("rollback_plan","Pending"),
            "assigned_to":     "NOC-L2 Team",
            "automation_used": approved,
        }
        _tickets_store.append(ticket)
        log.info(f"ITSM: Created new ticket {ticket['ticket_id']}")

    state["ticket"] = ticket
    state["trace"].append({"agent":"ITSM","output":ticket,"ts":datetime.utcnow().isoformat()})
    return state


# ═══════════════════════════════════════════════════════════════════════════════
# 7. LANGGRAPH WORKFLOW
# ═══════════════════════════════════════════════════════════════════════════════
def build_graph():
    g = StateGraph(NOCState)
    g.add_node("alarm_ingestion",   alarm_ingestion_agent)
    g.add_node("event_correlation", event_correlation_agent)
    g.add_node("parallel_analysis", parallel_analysis)
    g.add_node("remediation",       remediation_agent)
    g.add_node("automation",        automation_agent)
    g.add_node("itsm",              itsm_agent)
    g.set_entry_point("alarm_ingestion")
    g.add_edge("alarm_ingestion",   "event_correlation")
    g.add_edge("event_correlation", "parallel_analysis")
    g.add_edge("parallel_analysis", "remediation")
    g.add_edge("remediation",       "automation")
    g.add_edge("automation",        "itsm")
    g.add_edge("itsm",              END)
    return g.compile()

_noc_graph = None
def get_graph():
    global _noc_graph
    if _noc_graph is None:
        _noc_graph = build_graph()
    return _noc_graph


# ═══════════════════════════════════════════════════════════════════════════════
# 8. MCP MOCK TOOLS
# ═══════════════════════════════════════════════════════════════════════════════
def tool_router_reset(device: str, action: str = "bgp_reset") -> dict:
    return {"tool":"RouterTool","device":device,"action":action,
            "endpoint":f"POST /router/{device}/{action}","status":"Success",
            "ts":datetime.utcnow().isoformat()}

def tool_ticket_create(payload: dict) -> dict:
    return {"tool":"TicketTool","ticket_id":f"INC-{uuid.uuid4().hex[:6].upper()}",
            "status":"Created","payload":payload,"ts":datetime.utcnow().isoformat()}

def tool_log_search(device: str, query: str) -> dict:
    return {"tool":"LogSearchTool","device":device,"query":query,"results":[
        f"{device} 2026-06-12T10:00:00Z %BGP-5-ADJCHANGE: neighbor Down",
        f"{device} 2026-06-12T09:55:00Z %SYS-2-MALLOCFAIL: Memory allocation failed",
        f"{device} 2026-06-12T09:50:00Z %CPU-3-HIGH: CPU utilization 96% for 1 minute",
    ],"count":3}

def tool_kpi_fetch(device: str) -> dict:
    return {"tool":"KPITool","device":device,"metrics":{
        "cpu":            random.randint(10,99),
        "memory":         random.randint(20,98),
        "packet_loss":    round(random.uniform(0,15),2),
        "latency_ms":     random.randint(1,300),
        "bandwidth_util": random.randint(5,99),
    },"ts":datetime.utcnow().isoformat()}


# ═══════════════════════════════════════════════════════════════════════════════
# 9. FASTAPI APPLICATION
# ═══════════════════════════════════════════════════════════════════════════════
@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("=== NOC Copilot startup ===")
    alarms = generate_alarms(100)
    generate_incidents(alarms, 50)
    generate_historical_incidents()
    generate_sop()
    generate_vendor_kb()
    generate_runbooks()
    generate_topology()
    _load_stores()
    build_vector_store()
    get_graph()
    log.info("=== Startup complete ===")
    yield
    log.info("NOC Copilot shutdown")

app = FastAPI(title="NOC Agentic Copilot API", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class AlarmRequest(BaseModel):
    device:       str
    alarm:        str
    severity:     str = "Major"
    timestamp:    Optional[str] = None
    kpi:          Optional[dict] = None
    auto_approve: bool = False

class IncidentRequest(BaseModel):
    incident_id: str
    description: str
    severity:    str = "Major"

class ExecuteRequest(BaseModel):
    incident_id: str
    action:      str
    device:      str

class KPIInput(BaseModel):
    device:         str
    cpu:            Optional[float] = None
    memory:         Optional[float] = None
    packet_loss:    Optional[float] = None
    latency_ms:     Optional[float] = None
    bandwidth_util: Optional[float] = None


@app.get("/health")
def health():
    return {"status":"ok","agents":8,"vector_docs":"loaded","ts":datetime.utcnow().isoformat()}

@app.get("/alarms")
def get_alarms(limit: int = 50, severity: Optional[str] = None):
    data = _alarms_store
    if severity:
        data = [a for a in data if a.get("severity","").lower() == severity.lower()]
    return {"alarms": data[:limit], "total": len(data)}

@app.post("/alarm")
async def process_alarm(req: AlarmRequest, background_tasks: BackgroundTasks):
    alarm = {
        "alarm_id":  f"ALM-{uuid.uuid4().hex[:8].upper()}",
        "device":    req.device,
        "alarm":     req.alarm,
        "severity":  req.severity,
        "timestamp": req.timestamp or datetime.utcnow().isoformat(),
        "kpi":       req.kpi or tool_kpi_fetch(req.device)["metrics"],
        "status":    "Open",
        "source":    "manual",   # ← tag so UI can highlight it
    }

    # ── FIX 4: Insert into memory AND persist to disk immediately ──────────
    _alarms_store.insert(0, alarm)
    _save_alarms()
    log.info(f"Manual alarm {alarm['alarm_id']} saved to disk. Store size: {len(_alarms_store)}")

    initial_state: NOCState = {
        "alarm":             alarm,
        "normalized_alarm":  None,
        "correlation":       None,
        "root_cause":        None,
        "knowledge":         None,
        "sla_risk":          None,
        "remediation":       None,
        "automation_result": None,
        "ticket":            None,
        "human_approved":    req.auto_approve,
        "trace":             [],
    }
    result = get_graph().invoke(initial_state)
    _traces_store.insert(0, {"alarm_id":alarm["alarm_id"],"alarm":alarm,
                              "result":result,"ts":datetime.utcnow().isoformat()})
    return {
        "alarm_id":          alarm["alarm_id"],
        "normalized_alarm":  result["normalized_alarm"],
        "correlation":       result["correlation"],
        "root_cause":        result["root_cause"],
        "knowledge":         result["knowledge"],
        "sla_risk":          result["sla_risk"],
        "remediation":       result["remediation"],
        "automation_result": result["automation_result"],
        "ticket":            result["ticket"],
        "trace":             result["trace"],
    }


@app.post("/alarm/approve/{alarm_id}")
def approve_alarm(alarm_id: str):
    trace = next((t for t in _traces_store if t["alarm_id"]==alarm_id), None)
    if not trace:
        raise HTTPException(404, f"No trace found for alarm {alarm_id}")
    state: NOCState = trace["result"].copy()
    state["human_approved"] = True
    state["trace"] = list(state.get("trace",[]))
    state = automation_agent(state)
    state = itsm_agent(state)

    # Update alarm status to Resolved in store and persist
    for a in _alarms_store:
        if a.get("alarm_id") == alarm_id:
            a["status"] = "Resolved"
            break
    _save_alarms()

    trace["result"] = state
    return {
        "approved":          True,
        "alarm_id":          alarm_id,
        "automation_result": state["automation_result"],
        "ticket":            state["ticket"],
    }

@app.post("/incident")
def create_incident(req: IncidentRequest):
    incident = {"incident_id":req.incident_id,"description":req.description,
                "severity":req.severity,"status":"Open","created_at":datetime.utcnow().isoformat()}
    _incidents_store.insert(0, incident)
    return {"created":True,"incident":incident}

@app.get("/incidents")
def get_incidents(limit: int = 50):
    return {"incidents":_incidents_store[:limit],"total":len(_incidents_store)}

@app.post("/execute")
def execute_action(req: ExecuteRequest):
    return {"executed":True,"result":tool_router_reset(req.device, req.action)}

@app.get("/ticket")
def get_tickets(limit: int = 50):
    return {"tickets":_tickets_store[:limit],"total":len(_tickets_store)}

@app.get("/topology")
def get_topology():
    path = DATA_DIR / "topology.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"error":"topology not found"}

@app.get("/knowledge")
def search_knowledge(q: str = "BGP down", k: int = 5):
    return {"query":q,"results":retrieve_knowledge(q, k)}

@app.get("/traces")
def get_traces(limit: int = 20):
    return {"traces":_traces_store[:limit],"total":len(_traces_store)}

@app.post("/kpi")
def submit_kpi(kpi: KPIInput):
    data   = kpi.model_dump(exclude_none=True)
    device = data.pop("device")
    for alarm in _alarms_store:
        if alarm.get("device") == device:
            alarm.setdefault("kpi", {}).update(data)
    _save_alarms()
    return {"updated":True,"device":device,"kpi":data}

@app.get("/mcp/router/{device}/{action}")
def mcp_router(device: str, action: str):
    return tool_router_reset(device, action)

@app.get("/mcp/logs/{device}")
def mcp_logs(device: str, q: str = "error"):
    return tool_log_search(device, q)

@app.get("/mcp/kpi/{device}")
def mcp_kpi(device: str):
    return tool_kpi_fetch(device)

@app.post("/mcp/ticket")
def mcp_ticket(payload: dict):
    return tool_ticket_create(payload)

@app.get("/sample/regenerate")
def regenerate_data():
    alarms = generate_alarms(100)
    generate_incidents(alarms, 50)
    generate_historical_incidents()
    generate_sop()
    generate_vendor_kb()
    generate_runbooks()
    generate_topology()
    _load_stores()
    build_vector_store()
    return {"regenerated":True,"alarms":100,"incidents":50}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend:app", host="0.0.0.0", port=8000, reload=True)