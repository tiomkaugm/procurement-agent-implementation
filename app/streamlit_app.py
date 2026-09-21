"""Dashboard MARL Procurement. Run: streamlit run app/streamlit_app.py"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from procurement_marl.agents.random_agent import RandomPolicy
from procurement_marl.agents.rule_based import RuleBasedPolicy
from procurement_marl.costs import cash_balances
from procurement_marl.env import AGENTS, ProcurementEnv
from procurement_marl.evaluate import run_episode
from procurement_marl.scenario import composite_scores, eligible_vendors, load_scenario, sample_scenario

RUNS = ROOT / "runs"
BLUE, ORANGE = "#2a78d6", "#eb6834"          # categorical slots 1 and 2
PLAN = "Rencana optimal satu putaran"
AGENT_NAMES = {"IRE": "IRE (Intake & Routing)", "VMI": "VMI (Vendor Matrix)",
               "DA": "DA (Deal Architect)", "SLM": "SLM (Settlement & Liquidity)"}
VIOLATION = {"kas_negatif": "kas negatif", "anggaran": "anggaran terlampaui",
             "unit_mendesak": "unit mendesak terlambat", "kas_minimum": "kas di bawah minimum"}
REASON = {"kapasitas": "kapasitas kurang", "skor": "gugur skor komposit",
          "lead_time": "lead time melewati tenggat", "dikecualikan": "dikecualikan"}

st.set_page_config(page_title="MARL Procurement", layout="wide")


def rp(x: int) -> str:
    return ("-" if x < 0 else "") + "Rp" + f"{abs(x):,}".replace(",", ".")


# ------------------------------------------------------------------ policies
@st.cache_resource
def load_policy(name: str, seed: int):
    if name == "Random":
        return RandomPolicy(seed)
    if name == "Rule-based (meniru laporan)":
        return RuleBasedPolicy()
    if name == "IQL":
        from procurement_marl.agents.iql import IQL
        return IQL.load(RUNS / "iql" / "checkpoint.json")
    if name == "CTDE actor-critic":
        from procurement_marl.agents.ctde_ac import CTDEActorCritic
        return CTDEActorCritic.load(RUNS / "ctde" / "checkpoint.pt")
    return None    # single-round plan is built per scenario


def available_policies() -> list[str]:
    names = ["Rule-based (meniru laporan)", "Random"]
    if (RUNS / "iql" / "checkpoint.json").exists():
        names.append("IQL")
    if (RUNS / "ctde" / "checkpoint.pt").exists():
        names.append("CTDE actor-critic")
    return names + [PLAN]


def play_episode(policy_name: str, scenario_kind: str, seed: int) -> dict:
    scenario = load_scenario() if scenario_kind == "Fixture laporan" else sample_scenario(seed)
    if policy_name == PLAN:
        from procurement_marl.oracle import PlanPolicy, best_single_round_plan
        _, plan = best_single_round_plan(scenario)
        policy = PlanPolicy(plan)
    else:
        policy = load_policy(policy_name, seed)
    env = ProcurementEnv(scenario)
    # Rencana optimal dinilai satu putaran: jika konflik, episode berhenti di situ (sama seperti oracle).
    result = run_episode(env, policy, seed=seed, options={"scenario": scenario},
                         stop_at_first_check=(policy_name == PLAN))
    return {"scenario": scenario, "result": result, "policy": policy_name}


# ----------------------------------------------------------------- describing
def describe(e: dict) -> tuple[str, str]:
    """(decision, detail) in Indonesian for one log entry."""
    a = e["agent"]
    if a == "IRE":
        return e["action"], "; ".join(f"{b['qty']} unit bulan {b['month']}" for b in e["batches"])
    if a == "VMI":
        masked = ", ".join(f"{v}: {REASON[r]}" for v, r in e["masked"].items()) or "tidak ada yang di-mask"
        return f"pilih {e['vendor']}", f"batch {e['batch'] + 1}. Di-mask: {masked}. Estimasi {rp(e['estimate'])}"
    if a == "DA":
        note = ""
        if e.get("accepted") is not None:
            note = " (diterima)" if e["accepted"] else " (ditolak)"
        if e.get("withdrew"):
            note += ", vendor mundur"
        price = f"harga {rp(e['price'])}/unit" if e["price"] else "kembali ke VMI"
        return e["action"] + note, f"batch {e['batch'] + 1}, {e['vendor']}, {price}"
    if a == "SLM":
        note = ""
        if e["term_accepted"] is not None:
            note = " (termin diterima)" if e["term_accepted"] else " (termin ditolak, jatuh tempo)"
        return e["action"] + note, (f"batch {e['batch'] + 1}: barang {rp(e['goods_value'])}, diskon {rp(e['discount'])}, "
                                    f"transport+risiko {rp(e['logistics'])}, total {rp(e['total'])}")
    names = ", ".join(VIOLATION[v] for v in e["violations"])
    return ("KONSENSUS" if e["consensus"] else "konflik",
            f"total {rp(e['total'])}, sisa anggaran {rp(e['budget_left'])}. " + (f"Pelanggaran: {names}." if names else ""))


def round_payments(log: list[dict], step: int) -> dict[int, int]:
    """Payments of the batches settled so far in the round of log[step - 1]."""
    rnd, payments = log[step - 1]["round"], {}
    for e in log[:step]:
        if e["round"] == rnd and e["agent"] == "SLM":
            for m, amount in e["schedule"].items():
                payments[m] = payments.get(m, 0) + amount
    return payments


def cash_chart(sc, cash: list[int]) -> go.Figure:
    months = [f"Bulan {m}" for m in range(1, len(cash) + 1)]
    fig = go.Figure(go.Bar(x=months, y=[k / 1e6 for k in cash], marker_color=BLUE, name="Kas akhir bulan",
                           text=[f"{k / 1e6:.1f}" for k in cash], textposition="outside",
                           hovertemplate="%{x}: Rp%{y:.2f} juta<extra></extra>"))
    fig.add_hline(y=0, line_color="#888888", line_width=1)
    fig.add_hline(y=sc.min_cash / 1e6, line_dash="dash", line_color=ORANGE,
                  annotation_text=f"batas minimum {sc.min_cash / 1e6:.0f} jt", annotation_position="top left")
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=10), yaxis_title="juta Rp",
                      showlegend=False, bargap=0.5)
    return fig


def curve_chart(column: str, title: str, curves: dict[str, pd.DataFrame]) -> go.Figure:
    fig = go.Figure()
    for (name, df), color in zip(curves.items(), (BLUE, ORANGE)):
        fig.add_trace(go.Scatter(x=df["episode"], y=df[column], name=name, mode="lines",
                                 line=dict(color=color, width=2)))
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=40, b=10), title=title, hovermode="x unified",
                      xaxis_title="episode", legend=dict(orientation="h", y=-0.25))
    return fig


# ---------------------------------------------------------------------- pages
def episode_tab() -> None:
    ep = st.session_state.get("episode")
    if ep is None:
        st.info("Pilih kebijakan dan skenario di panel kiri, lalu tekan **Jalankan episode**.")
        return
    sc, result, log = ep["scenario"], ep["result"], ep["result"]["log"]

    st.subheader("Skenario")
    c = st.columns(5)
    c[0].metric("Permintaan", f"{sc.quantity} unit")
    c[1].metric("Mendesak", f"{sc.urgent_quantity} unit")
    c[2].metric("Anggaran", rp(sc.budget))
    c[3].metric("Tenggat", f"{sc.deadline_days} hari")
    c[4].metric("Kas awal", rp(sc.initial_cash))
    scores, eligible = composite_scores(sc), eligible_vendors(sc)
    st.dataframe(pd.DataFrame([{
        "Vendor": v.name, "Harga list": rp(v.list_price), "Transport+risiko/unit": rp(v.transport + v.risk),
        "Kualitas": v.quality, "Lead time (hari)": v.lead_time, "Kapasitas": v.capacity,
        "Penawaran awal": rp(v.initial_offer), "Harga lantai": rp(v.floor_price),
        "Diskon bayar cepat": f"{v.discount_pct}%", "Skor komposit": round(scores[v.name], 2),
        "Lolos skor": "ya" if v.name in eligible else "gugur"} for v in sc.vendors]),
        hide_index=True, use_container_width=True)

    outcome = "KONSENSUS" if result["consensus"] else "TANPA KONSENSUS"
    st.subheader(f"Episode: {outcome} setelah {result['rounds']} putaran ({ep['policy']})")
    st.caption(f"Return tim {result['team_return']:.2f}. "
               "Reward diberikan di setiap putaran, jadi episode yang gagal tetap mengumpulkan reward positif per putaran.")

    if ep["policy"] == PLAN and not result["consensus"]:
        st.info("Rencana optimal satu putaran konflik di putaran 1 "
                f"({', '.join(result['violations']) or 'pelanggaran batasan'}). "
                "Oracle menganggap episode berhenti di sini dengan penalti tim, tanpa putaran revisi.")

    if "step" not in st.session_state or st.session_state.get("step_for") != id(ep):
        st.session_state.update(step=1, step_for=id(ep))
    b = st.columns([1, 1, 6])
    if b[0].button("◀ Mundur", disabled=st.session_state.step <= 1):
        st.session_state.step -= 1
        st.rerun()
    if b[1].button("Maju ▶", disabled=st.session_state.step >= len(log)):
        st.session_state.step += 1
        st.rerun()
    step = st.slider("Langkah", 1, len(log), key="step")
    entry = log[step - 1]

    st.subheader(f"Kartu agen (setelah langkah {step}, putaran {entry['round']})")
    cols = st.columns(4)
    for col, agent in zip(cols, AGENTS):
        past = [e for e in log[:step] if e["agent"] == agent and e["round"] == entry["round"]]
        with col.container(border=True):
            st.markdown(f"**{AGENT_NAMES[agent]}**")
            if not past:
                st.caption("belum bertindak di putaran ini")
            else:
                decision, detail = describe(past[-1])
                st.markdown(f"Keputusan: `{decision}`")
                st.caption(detail)
            if entry["agent"] == agent:
                st.markdown("🟢 giliran ini")

    left, right = st.columns(2)
    with left:
        st.subheader("Kas per bulan")
        if entry["agent"] == "ENV":
            cash = entry["cash"]
        else:
            cash = cash_balances(sc.initial_cash, list(sc.inflows), list(sc.other_needs), round_payments(log, step))
        st.plotly_chart(cash_chart(sc, cash), use_container_width=True)
        st.caption("Kas akhir bulan = kas sebelumnya + arus masuk - pembayaran - kebutuhan lain, "
                   "berdasarkan batch yang sudah dibayar sampai langkah ini.")
    with right:
        st.subheader("Rincian biaya putaran ini")
        slm = [e for e in log[:step] if e["round"] == entry["round"] and e["agent"] == "SLM"]
        if slm:
            df = pd.DataFrame([{"Batch": e["batch"] + 1, "Vendor": e["vendor"], "Harga/unit": rp(e["price"]),
                                "Barang": rp(e["goods_value"]), "Diskon": rp(e["discount"]),
                                "Transport+risiko": rp(e["logistics"]), "Total": rp(e["total"]),
                                "Cara bayar": e["mode"]} for e in slm])
            st.dataframe(df, hide_index=True, use_container_width=True)
            total = sum(e["total"] for e in slm)
            st.metric("Total pengadaan", rp(total), delta=f"sisa anggaran {rp(sc.budget - total)}",
                      delta_color="normal" if total <= sc.budget else "inverse")
        else:
            st.caption("Belum ada batch yang dibayar di putaran ini.")

    st.subheader("Log negosiasi")
    rows = []
    for i, e in enumerate(log[:step], start=1):
        decision, detail = describe(e)
        rows.append({"Langkah": i, "Putaran": e["round"], "Agen": e["agent"], "Keputusan": decision, "Rincian": detail})
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


def training_tab() -> None:
    curves = {name: pd.read_csv(RUNS / d / "curve.csv") for name, d in (("IQL", "iql"), ("CTDE", "ctde"))
              if (RUNS / d / "curve.csv").exists()}
    if not curves:
        st.info("Belum ada kurva training. Jalankan `python scripts/train.py --algo iql` dan `--algo ctde`.")
        return
    st.plotly_chart(curve_chart("mean_team_return", "Return tim rata-rata per blok", curves), use_container_width=True)
    st.plotly_chart(curve_chart("consensus_rate", "Tingkat konsensus per blok", curves), use_container_width=True)
    st.caption("Kurva adalah rata-rata per blok episode saat berlatih (dengan eksplorasi), bukan evaluasi greedy. "
               "Hasil evaluasi akhir ada di tab Perbandingan.")


def comparison_tab() -> None:
    path = RUNS / "comparison.csv"
    if not path.exists():
        st.info("Belum ada tabel. Jalankan `python scripts/evaluate.py`.")
        return
    st.subheader("Perbandingan kebijakan (500 skenario acak, seed 0-499)")
    st.dataframe(pd.read_csv(path), hide_index=True, use_container_width=True)
    st.warning("Rasio terhadap rencana optimal satu putaran boleh > 1 dan tidak berarti kebijakan lebih baik. "
               "Rencana optimal menilai kegagalan berhenti di satu putaran, sedangkan kebijakan multi-putaran "
               "mengumpulkan reward per putaran yang menutup sebagian penalti tim. Lihat tingkat konsensus: "
               "rencana optimal satu putaran 86,2% dibanding CTDE 80,4%.")
    sens = RUNS / "sensitivity" / "summary.csv"
    if sens.exists():
        st.subheader("Sensitivitas terhadap peluang penerimaan vendor")
        st.dataframe(pd.read_csv(sens), hide_index=True, use_container_width=True)
        st.caption("p_accept = p_termin dibuat tetap. Kolom terakhir: seberapa sering DA memilih penawaran_balik "
                   "dan SLM memilih revisi_termin.")


# ----------------------------------------------------------------------- main
st.title("MARL Procurement: pengadaan barang dengan empat agen")
st.caption("Kelompok 2, Agen Cerdas Enterprise. Empat agen (IRE, VMI, DA, SLM) bergiliran memutuskan pengadaan.")

with st.sidebar:
    st.header("Pengaturan episode")
    policy_name = st.selectbox("Kebijakan", available_policies())
    scenario_kind = st.radio("Skenario", ["Fixture laporan", "Acak (pilih seed)"])
    seed = st.number_input("Seed", min_value=0, value=0, step=1)
    if st.button("Jalankan episode", type="primary"):
        with st.spinner("Menjalankan episode..."):
            st.session_state["episode"] = play_episode(policy_name, scenario_kind, int(seed))

tab1, tab2, tab3 = st.tabs(["Episode", "Kurva training", "Perbandingan"])
with tab1:
    episode_tab()
with tab2:
    training_tab()
with tab3:
    comparison_tab()
