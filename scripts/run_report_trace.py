"""Print the six coordination stages of Bab 6.5. Run: python scripts/run_report_trace.py."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from procurement_marl.presentation import describe, rp
from procurement_marl.report import COORDINATION_STAGES, run_report_episode
from procurement_marl.scenario import load_scenario


def main() -> None:
    sc = load_scenario()
    result = run_report_episode(seed=0)
    print(f"Skenario laporan: {sc.quantity} unit ({sc.urgent_quantity} mendesak), "
          f"anggaran {rp(sc.budget)}, tenggat {sc.deadline_days} hari")
    print("Simulasi rekomendasi; belum ada pembayaran aktual atau otorisasi ERP.")
    for stage, title in COORDINATION_STAGES.items():
        print(f"\n{'Persiapan' if stage == 0 else f'Tahap {stage}'}: {title}")
        for e in result["log"]:
            if e["coordination_step"] != stage:
                continue
            decision, detail = describe(e)
            print(f"  {e.get('reviewer', e['agent'])}: {decision}. {detail}")
            if "cash" in e:
                for month, cash in enumerate(e["cash"][:2], 1):
                    print(f"  Kas akhir bulan {month}: {rp(cash)}")
    print("\nHasil: USULAN PERLU REVISI sebelum disetujui. "
          "IRE meninjau kebutuhan, VMI alternatif vendor, DA harga/termin, dan SLM dampak kas.")


if __name__ == "__main__":
    main()
