"""Sanity check of the random scenario generator before training.

Reports, over N seeds: share of feasible scenarios, share where the naive plan works
(teruskan + cheapest eligible vendor + penawaran_awal + bayar_cepat, in round 1),
and the consensus rate of the rule-based and random policies.

Usage: python scripts/check_scenarios.py [N] [config.yaml]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from procurement_marl.agents.random_agent import RandomPolicy
from procurement_marl.agents.rule_based import RuleBasedPolicy
from procurement_marl.env import ProcurementEnv
from procurement_marl.evaluate import run_episode
from procurement_marl.oracle import is_feasible
from procurement_marl.scenario import CONFIG_DIR, sample_scenario


def scenario_stats(n: int = 500, config: Path = CONFIG_DIR / "env_default.yaml") -> dict[str, float]:
    env = ProcurementEnv("report", config_path=config)
    feasible = naive = rule = rand = 0
    for seed in range(n):
        sc = sample_scenario(seed, config)
        opts = {"scenario": sc}
        feasible += is_feasible(sc)
        # the rule-based policy in round 1 is exactly the naive plan
        naive += run_episode(env, RuleBasedPolicy(), 0, opts, stop_at_first_check=True)["consensus"]
        rule += run_episode(env, RuleBasedPolicy(), 0, opts)["consensus"]
        rand += run_episode(env, RandomPolicy(seed), seed, opts)["consensus"]
    return {"feasible": feasible / n, "naive_success": naive / n,
            "rule_based_consensus": rule / n, "random_consensus": rand / n}


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    config = Path(sys.argv[2]) if len(sys.argv) > 2 else CONFIG_DIR / "env_default.yaml"
    stats = scenario_stats(n, config)
    print(f"{n} seed acak (seed 0..{n - 1})")
    print(f"  skenario layak                  : {stats['feasible']:.1%}   (target 80-90%)")
    print(f"  rencana naif gagal (putaran 1)  : {1 - stats['naive_success']:.1%}   (target 40-60%)")
    print(f"  konsensus rule-based            : {stats['rule_based_consensus']:.1%}")
    print(f"  konsensus random                : {stats['random_consensus']:.1%}")
