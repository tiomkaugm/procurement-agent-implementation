"""Print the rule-based trace on the report fixture (Bab 6), in Indonesian.

Usage: python scripts/run_report_trace.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from procurement_marl.agents.rule_based import RuleBasedPolicy
from procurement_marl.env import ProcurementEnv
from procurement_marl.evaluate import run_episode

VIOLATION = {"kas_negatif": "kas negatif", "anggaran": "anggaran terlampaui",
             "unit_mendesak": "unit mendesak terlambat", "kas_minimum": "kas di bawah minimum"}
REASON = {"kapasitas": "kapasitas kurang", "skor": "gugur skor komposit",
          "lead_time": "lead time melewati tenggat", "dikecualikan": "dikecualikan"}


def rp(x: int) -> str:
    return ("-" if x < 0 else "") + "Rp" + f"{abs(x):,}".replace(",", ".")


def show(entry: dict, n_batches: int) -> None:
    agent, tag = entry["agent"], f" (batch {entry.get('batch', 0) + 1})"
    if agent == "IRE":
        parts = [f"{b['qty']} unit bulan {b['month']}" for b in entry["batches"]]
        print(f"  IRE  : {entry['action']} -> {len(parts)} batch: " + "; ".join(parts))
    elif agent == "VMI":
        masked = ", ".join(f"{v} di-mask ({REASON[r]})" for v, r in entry["masked"].items()) or "tidak ada yang di-mask"
        print(f"  VMI{tag if n_batches > 1 else ''}: {masked}. Pilih {entry['vendor']}")
        print(f"        estimasi biaya {entry['vendor']} (harga terakhir yang disepakati, atau harga list): {rp(entry['estimate'])}")
    elif agent == "DA":
        print(f"  DA{tag if n_batches > 1 else ''}: {entry['action']}, harga {entry['vendor']} = {rp(entry['price'])}/unit")
    elif agent == "SLM":
        print(f"  SLM{tag if n_batches > 1 else ''}: {entry['mode']}. Barang {rp(entry['goods_value'])}, diskon {rp(entry['discount'])}, "
              f"setelah diskon {rp(entry['goods_after_discount'])}, transport+risiko {rp(entry['logistics'])}, "
              f"total {rp(entry['total'])}; dibayar " + ", ".join(f"bulan {m}" for m in entry["schedule"]))


def main() -> None:
    env = ProcurementEnv(scenario="report")
    result = run_episode(env, RuleBasedPolicy(), seed=0)
    sc = env.scenario
    print(f"Skenario laporan: {sc.quantity} unit ({sc.urgent_quantity} mendesak), anggaran {rp(sc.budget)}, tenggat {sc.deadline_days} hari")
    print("Kebijakan: rule-based (meniru laporan)")
    for rnd in range(1, result["rounds"] + 1):
        entries = [e for e in env.log if e["round"] == rnd]
        n_batches = len(entries[0]["batches"])
        print(f"\n=== Putaran {rnd} ===")
        for e in entries:
            if e["agent"] != "ENV":
                show(e, n_batches)
                continue
            for m, k in enumerate(e["cash"][:2], start=1):
                print(f"  Kas akhir bulan {m}: {rp(k)}")
            print(f"  Total pengadaan {rp(e['total'])}; sisa anggaran {rp(e['budget_left'])}")
            if e["warnings"]:
                print(f"  Peringatan: kas bulan 1 di bawah batas minimum {rp(sc.min_cash)} (bukan syarat keras)")
            if e["consensus"]:
                print("  Hasil: KONSENSUS")
            else:
                names = ", ".join(VIOLATION[v] for v in e["violations"])
                print(f"  Pelanggaran: {names}. " + ("Episode berakhir tanpa konsensus." if e["done"] else "Mulai putaran revisi."))
    print(f"\nHasil akhir: {'konsensus' if result['consensus'] else 'tanpa konsensus'} setelah {result['rounds']} putaran; "
          f"return tim {result['team_return']:.2f}")


if __name__ == "__main__":
    main()
