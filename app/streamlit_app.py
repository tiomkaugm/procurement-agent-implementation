"""Dashboard MARL Procurement. Run: streamlit run app/streamlit_app.py"""

import sys
import json
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
from procurement_marl.scenario import cash_diagnosis, composite_scores, eligible_vendors, load_scenario, sample_scenario
from procurement_marl.presentation import STOP_REASON, VIOLATION, describe, log_rows, round_payments, rp
from procurement_marl.report import COORDINATION_STAGES, run_report_episode

RUNS = ROOT / "runs"
BLUE, ORANGE = "#2a78d6", "#eb6834"          # categorical slots 1 and 2
PLAN = "Rencana optimal satu putaran"
AGENT_NAMES = {"IRE": "IRE (Intake & Routing)", "VMI": "VMI (Vendor Matrix)",
               "DA": "DA (Deal Architect)", "SLM": "SLM (Settlement & Liquidity)"}
POLICY_HELP = {
    "Rule-based (meniru laporan)": "Fixture: reproduksi enam tahap koordinasi Bab 6.5. Pada skenario acak: "
                                  "aturan tetap untuk pembanding, berhenti jika rencana dan keadaan berulang.",
    "Random": "Memilih tindakan valid secara acak. Berguna untuk menguji jalur proses dan sebagai pembanding dasar.",
    "IQL": "Setiap agen memakai tabel tindakan hasil latihan. Episode ini menjalankan model, tidak melatih ulang.",
    "CTDE actor-critic": "Actor memilih berdasarkan observasi lokal; critic bersama hanya dipakai saat latihan. "
                         "Episode ini menjalankan model yang sudah dilatih.",
    PLAN: "Mencari nilai harapan reward tim tertinggi dalam pilihan rencana satu putaran. "
          "Tidak menjamin biaya terendah atau konsensus; berhenti setelah pemeriksaan pertama.",
}

st.set_page_config(page_title="MARL Procurement", layout="wide")


# ------------------------------------------------------------------ policies
def load_policy(name: str, seed: int):
    # A fresh policy per episode prevents cached RNG/Q-table state leaking between runs.
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
    report_replay = policy_name == "Rule-based (meniru laporan)" and scenario_kind == "Fixture laporan"
    if report_replay:
        return {"scenario": scenario, "result": run_report_episode(seed), "policy": policy_name,
                "report_replay": True}
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
    return {"scenario": scenario, "result": result, "policy": policy_name, "report_replay": False}


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
        hide_index=True, width="stretch")

    diag = cash_diagnosis(sc)
    if diag["infeasible"]:
        st.warning(
            f"Diagnosis kas dalam batas harga dan horizon simulator: kas yang tersedia selama "
            f"horizon {rp(diag['available'])} (kas awal + arus masuk - kebutuhan lain), sedangkan biaya terendah "
            f"{rp(diag['lower_bound'])} (seluruh permintaan ke vendor {diag['vendor']} di harga lantai dengan diskon bayar cepat). "
            f"Kekurangan minimal {rp(diag['shortfall'])}. Perlu tambahan dana, arus kas masuk, atau permintaan yang lebih kecil.")
    else:
        st.caption("Diagnosis kas: batas bawah biaya masih tercakup kas horizon. Ini belum membuktikan skenario layak; "
                   "kegagalan di sini berarti solusi belum ditemukan, bukan pasti tidak mungkin.")

    report_replay = ep.get("report_replay", False)
    outcome = "KONSENSUS RENCANA" if result["consensus"] else "TANPA KONSENSUS"
    if result["stop_reason"] == "report_revision_required":
        outcome = "USULAN PERLU REVISI"
    elif result["truncated"]:
        outcome = "SIMULASI DIHENTIKAN"
    duration = (f"{result['coordination_steps']} tahap koordinasi" if report_replay
                else f"{result['rounds']} siklus simulasi")
    st.subheader(f"Episode: {outcome} setelah {duration} ({ep['policy']})")
    st.info(STOP_REASON[result["stop_reason"]])
    st.caption(f"Return tim tanpa diskonto: {result['team_return']:.2f}. "
               "Reward merupakan desain simulator dan diberikan per siklus; bukan angka ilustrasi tabel 6.6.")
    if report_replay:
        with st.expander("Alur enam tahap sesuai Bab 6.5"):
            for number, label in COORDINATION_STAGES.items():
                st.write(f"{'Persiapan' if number == 0 else str(number)}: {label}")
            st.write("Evaluasi bersama: IRE meninjau kebutuhan, VMI alternatif vendor, DA harga/termin, "
                     "dan SLM dampak kas. Skenario usulan perlu revisi sebelum persetujuan pengadaan.")

    if ep["policy"] == PLAN and not result["consensus"]:
        st.info("Rencana optimal satu putaran konflik di putaran 1 "
                f"({', '.join(VIOLATION[v] for v in result['violations']) or 'pelanggaran batasan'}). "
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

    position = entry.get("coordination_label", f"Siklus simulasi {entry['round']}")
    st.subheader(f"Langkah {step}: {position}")
    cols = st.columns(4)
    for col, agent in zip(cols, AGENTS):
        past = [e for e in log[:step] if e.get("reviewer", e["agent"]) in (agent, "BERSAMA")
                and e["round"] == entry["round"]]
        with col.container(border=True):
            st.markdown(f"**{AGENT_NAMES[agent]}**")
            if not past:
                st.caption("belum bertindak pada rencana ini")
            else:
                decision, detail = describe(past[-1])
                st.markdown(f"Keputusan: `{decision}`")
                st.caption(detail)
            if entry.get("reviewer", entry["agent"]) in (agent, "BERSAMA"):
                st.markdown("🟢 giliran ini")

    left, right = st.columns(2)
    with left:
        st.subheader("Kas per bulan")
        if "cash" in entry:
            cash = entry["cash"]
        else:
            cash = cash_balances(sc.initial_cash, list(sc.inflows), list(sc.other_needs), round_payments(log, step))
        st.plotly_chart(cash_chart(sc, cash), width="stretch")
        st.caption("Kas akhir bulan = kas sebelumnya + arus masuk - pembayaran - kebutuhan lain, "
                   "berdasarkan jadwal yang diusulkan sampai langkah ini. Belum ada pembayaran aktual.")
    with right:
        st.subheader("Rincian biaya rencana ini")
        slm = [e for e in log[:step] if e["round"] == entry["round"] and e.get("event") == "payment_plan"]
        if slm:
            df = pd.DataFrame([{"Batch": e["batch"] + 1, "Vendor": e["vendor"], "Harga/unit": rp(e["price"]),
                                "Barang": rp(e["goods_value"]), "Diskon": rp(e["discount"]),
                                "Transport+risiko": rp(e["logistics"]), "Total": rp(e["total"]),
                                "Cara bayar": e["mode"]} for e in slm])
            st.dataframe(df, hide_index=True, width="stretch")
            total = sum(e["total"] for e in slm)
            st.metric("Total pengadaan", rp(total), delta=f"sisa anggaran {rp(sc.budget - total)}",
                      delta_color="normal" if total <= sc.budget else "inverse")
        else:
            st.caption("Belum ada rekomendasi jadwal pembayaran untuk rencana ini.")

    st.subheader("Log negosiasi")
    st.dataframe(pd.DataFrame(log_rows(log[:step])), hide_index=True, width="stretch")
    st.download_button("Unduh log lengkap (CSV)", pd.DataFrame(log_rows(log)).to_csv(index=False).encode("utf-8-sig"),
                       file_name="log_negosiasi.csv", mime="text/csv")


def training_tab() -> None:
    curves = {name: pd.read_csv(RUNS / d / "curve.csv") for name, d in (("IQL", "iql"), ("CTDE", "ctde"))
              if (RUNS / d / "curve.csv").exists()}
    if not curves:
        st.info("Belum ada kurva training. Jalankan `python scripts/train.py --algo iql` dan `--algo ctde`.")
        return
    st.plotly_chart(curve_chart("mean_team_return", "Return tim rata-rata per blok", curves), width="stretch")
    st.plotly_chart(curve_chart("consensus_rate", "Tingkat konsensus per blok", curves), width="stretch")
    st.caption("Kurva adalah rata-rata per blok episode saat berlatih (dengan eksplorasi), bukan evaluasi greedy. "
               "Hasil evaluasi akhir ada di tab Perbandingan.")


def comparison_tab() -> None:
    path = RUNS / "comparison.csv"
    if not path.exists():
        st.info("Belum ada tabel. Jalankan `python scripts/evaluate.py`.")
        return
    meta_path = RUNS / "comparison.meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    if meta.get("evaluation_version") != 2:
        st.info("Tabel tersimpan berasal dari evaluasi sebelum perbaikan alur. "
                "Jalankan ulang evaluasi untuk memperbarui hasil penghentian Rule-based dan rencana satu putaran.")
    st.subheader("Perbandingan kebijakan — hasil evaluasi tersimpan")
    st.dataframe(pd.read_csv(path), hide_index=True, width="stretch")
    st.warning("Rasio terhadap rencana optimal satu putaran boleh > 1 dan tidak berarti kebijakan lebih baik. "
               "Rencana optimal menilai kegagalan berhenti di satu putaran, sedangkan kebijakan multi-putaran "
               "mengumpulkan reward per siklus. Bandingkan juga konsensus, pelanggaran, biaya, "
               "dan jumlah siklus pada tabel; return di sini tanpa diskonto.")
    sens = RUNS / "sensitivity" / "summary.csv"
    if sens.exists():
        st.subheader("Sensitivitas terhadap peluang penerimaan vendor")
        st.dataframe(pd.read_csv(sens), hide_index=True, width="stretch")
        st.caption("p_accept = p_termin dibuat tetap. Kolom terakhir: seberapa sering DA memilih penawaran_balik "
                   "dan SLM memilih revisi_termin. Ini hasil eksperimen tersimpan; "
                   "tidak dihitung ulang ketika menjalankan episode atau evaluasi utama.")


# ----------------------------------------------------------------------- main
st.title("MARL Procurement: pengadaan barang dengan empat agen")
st.caption("Simulasi rekomendasi pengadaan oleh IRE, VMI, DA, dan SLM. "
           "Pembayaran aktual memerlukan verifikasi tagihan, penerimaan barang, dan otorisasi di ERP.")

with st.sidebar:
    st.header("Pengaturan episode")
    policy_name = st.selectbox("Kebijakan", available_policies())
    st.caption(POLICY_HELP[policy_name])
    scenario_kind = st.radio("Skenario", ["Fixture laporan", "Acak (pilih seed)"])
    seed = st.number_input("Seed", min_value=0, value=0, step=1,
                           help="Angka awal pengacakan. Pengaturan sama menghasilkan episode yang sama. "
                                "Pada skenario acak, seed juga menentukan data skenario; tidak melatih ulang model.")
    if st.button("Jalankan episode", type="primary"):
        with st.spinner("Menjalankan episode..."):
            st.session_state["episode"] = play_episode(policy_name, scenario_kind, int(seed))

with st.expander("Cakupan simulasi dan asumsi"):
    st.write("VMI memakai empat komponen skor berbobot tetap. Nilai reputasi, normalisasi skor, ambang, "
             "harga lantai, peluang penerimaan vendor, dan termin 50:50 adalah asumsi simulator. "
             "Batas kas Rp60 juta pada fixture hanya peringatan.")
    st.write("IRE memakai jumlah kebutuhan mendesak yang sudah tersedia, belum melakukan klarifikasi dokumen. "
             "71 fitur vendor, klasifikasi ML historis, NLP kontrak, portal pemasok, dan ERP belum diimplementasikan.")
    st.write("CTDE menggunakan actor-critic dengan kumpulan episode baru saat training (on-policy). "
             "Belum menggunakan shared replay buffer. Revisi termin diproses sebagai usulan SLM, "
             "negosiasi DA dengan respons vendor tersimulasi, lalu rekomendasi jadwal oleh SLM.")

tab1, tab2, tab3 = st.tabs(["Episode", "Kurva training", "Perbandingan"])
with tab1:
    episode_tab()
with tab2:
    training_tab()
with tab3:
    comparison_tab()
