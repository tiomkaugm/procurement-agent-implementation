"""Sensitivity analysis: retrain IQL and CTDE with p_accept = p_termin fixed at 0.3, 0.5, 0.7, 0.9.

Reports team return, consensus rate, and how often DA picks penawaran_balik and SLM picks
revisi_termin. Shows whether the conclusions depend on our assumption about vendor acceptance.

Usage: python scripts/sensitivity.py [episodes]   (about 5 minutes per override value)
"""

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from train import train_ctde, train_iql

from procurement_marl.agents.random_agent import RandomPolicy
from procurement_marl.agents.rule_based import RuleBasedPolicy
from procurement_marl.evaluate import evaluate_policy

OVERRIDES = [0.3, 0.5, 0.7, 0.9]


def main(episodes: int = 50_000) -> None:
    out = ROOT / "runs" / "sensitivity"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for ov in OVERRIDES:
        agents = {"Random": RandomPolicy(0), "Rule-based": RuleBasedPolicy(),
                  "IQL": train_iql(episodes, 0, out / f"p{ov}" / "iql", override=ov),
                  "CTDE": train_ctde(episodes, 0, out / f"p{ov}" / "ctde", override=ov)}
        for name, policy in agents.items():
            m = evaluate_policy(policy, 500, override=ov)
            rows.append([ov, name, round(m["team_return"], 3), round(m["consensus"], 3),
                         round(m["counter_offer_rate"], 3), round(m["term_revision_rate"], 3)])
            print(rows[-1], flush=True)
    header = ["p_accept=p_termin", "kebijakan", "return_tim", "konsensus", "DA_penawaran_balik", "SLM_revisi_termin"]
    with open(out / "summary.csv", "w", newline="") as f:
        csv.writer(f).writerows([header] + rows)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 50_000)
