import numpy as np
import pytest
from pettingzoo.test import api_test

from procurement_marl.agents.random_agent import RandomPolicy
from procurement_marl.env import AGENTS, ProcurementEnv
from procurement_marl.evaluate import run_episode
from procurement_marl.oracle import best_single_round_plan, is_feasible
from procurement_marl.scenario import load_scenario, sample_scenario


def test_pettingzoo_api_report():
    api_test(ProcurementEnv("report"), num_cycles=300)


def test_pettingzoo_api_random_scenarios():
    api_test(ProcurementEnv("random"), num_cycles=300)


def test_observations_masks_and_state():
    env = ProcurementEnv()
    env.reset(seed=1)
    seen = set()
    for agent in env.agent_iter():
        obs, _, term, _, _ = env.last()
        assert env.observation_space(agent)["observation"].contains(obs["observation"])
        assert env.state_space.contains(env.state())
        d = env.discrete_obs(agent)
        assert len(d) == 4 and all(0 <= x <= 4 for x in d)
        seen.add(agent)
        if term:
            env.step(None)
            continue
        assert obs["action_mask"].sum() >= 1
        env.step(int(obs["action_mask"].argmax()))
    assert seen == set(AGENTS)


def test_partial_observability_no_cash_for_ire_and_vmi():
    env = ProcurementEnv()
    env.reset(seed=0)
    a = env.observe("IRE")["observation"].copy()
    b = env.observe("VMI")["observation"].copy()
    env.committed = {1: 99_000_000}       # change the cash picture
    assert np.array_equal(a, env.observe("IRE")["observation"])
    assert np.array_equal(b, env.observe("VMI")["observation"])
    assert not np.array_equal(env.observe("SLM")["observation"][3:7], np.zeros(4))


def test_seeded_reproducibility():
    r1 = run_episode(ProcurementEnv("random"), RandomPolicy(5), seed=11)
    r2 = run_episode(ProcurementEnv("random"), RandomPolicy(5), seed=11)
    assert r1["returns"] == r2["returns"] and r1["log"] == r2["log"]


def test_random_policy_episodes_terminate():
    env = ProcurementEnv("random")
    for seed in range(20):
        result = run_episode(env, RandomPolicy(seed), seed=seed)
        assert result["rounds"] <= 6


# ---- stochastic paths ----------------------------------------------------------
def play(env, actions):
    """Feed actions in order; returns after they are used up."""
    for a in actions:
        env.step(a)


def env_with(p_accept=None, p_termin=None, relation=None):
    env = ProcurementEnv()
    env.stoch["p_accept_override"] = p_accept
    env.stoch["p_termin_override"] = p_termin
    sc = load_scenario()
    if relation is not None:
        from dataclasses import replace
        sc = replace(sc, vendors=tuple(replace(v, relation=relation) for v in sc.vendors))
    env.reset(seed=0, options={"scenario": sc})
    return env


def test_counter_offer_always_accepted_and_term_accepted():
    env = env_with(1.0, 1.0)
    play(env, [0, 1, 1, 2])           # teruskan, vendor B, penawaran_balik, revisi_termin
    da = [e for e in env.log if e["agent"] == "DA"][0]
    slm = [e for e in env.log if e["agent"] == "SLM"][0]
    assert da["accepted"] is True and da["price"] == 89_000     # floor price
    assert slm["term_accepted"] is True and slm["mode"] == "revisi_termin"
    assert slm["schedule"] == {1: 12_000_000, 2: 44_500_000, 3: 44_500_000}


def test_counter_offer_rejected_lowers_relation_and_termin_falls_back():
    env = env_with(0.0, 0.0)
    play(env, [0, 1, 1, 2])
    da = [e for e in env.log if e["agent"] == "DA"][0]
    slm = [e for e in env.log if e["agent"] == "SLM"][0]
    assert da["accepted"] is False and da["price"] == 89_500     # back to initial offer
    assert da["relation"] == pytest.approx(0.4) and not da["withdrew"]
    assert slm["term_accepted"] is False and slm["mode"] == "bayar_jatuh_tempo"
    assert env.relation["B"] == pytest.approx(0.3)               # 0.6 - 0.2 - 0.1


def test_vendor_withdraws_when_relation_below_threshold():
    env = env_with(0.0, None, relation=0.45)
    play(env, [1, 1, 1])              # rejected: 0.45 - 0.2 = 0.25 < 0.3
    da = [e for e in env.log if e["agent"] == "DA"][0]
    assert da["withdrew"] is True
    assert env.stage == "VMI"          # batch goes back to VMI
    assert env.observe("VMI")["action_mask"][1] == 0     # B is masked for this batch


def test_alternative_request_once_per_batch():
    env = env_with()
    play(env, [0, 1])                 # teruskan, vendor B (only eligible one for 1000 units)
    assert env.observe("DA")["action_mask"][2] == 0      # no other vendor available
    env = env_with()
    play(env, [1, 1])                 # klarifikasi, batch 700: vendor B
    assert env.observe("DA")["action_mask"][2] == 1
    play(env, [2])                    # ask alternative -> VMI without B
    assert env.stage == "VMI" and env.observe("VMI")["action_mask"].tolist() == [1, 0, 0]
    play(env, [0])                    # vendor A
    assert env.observe("DA")["action_mask"][2] == 0      # already used once


def test_acceptance_frequency_matches_probability():
    env = env_with()
    p = env._p_accept("B")
    assert p == pytest.approx(0.56)
    assert env._p_termin("B") == pytest.approx(0.66)
    rate = np.mean([env._draw(p) for _ in range(10_000)])
    assert abs(rate - p) < 0.02


# ---- oracle -------------------------------------------------------------------
def test_report_fixture_is_infeasible():
    assert not is_feasible(load_scenario())


def test_random_scenarios_feasibility_and_naive_plan():
    from importlib import import_module
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    stats = import_module("check_scenarios").scenario_stats(500)
    assert 0.80 <= stats["feasible"] <= 0.90
    assert 0.40 <= 1 - stats["naive_success"] <= 0.60      # naive plan fails in 40-60% of scenarios


def test_single_round_plan_beats_naive_plan():
    from procurement_marl.oracle import expected_return
    sc = sample_scenario(3)
    env = ProcurementEnv(sc)
    best, _ = best_single_round_plan(sc)
    naive = (0, ((1, 0, 0),))         # teruskan, vendor B, penawaran_awal, bayar_cepat
    assert best >= expected_return(env, sc, naive) - 1e-9


# ---- step limits ---------------------------------------------------------------
def test_counter_offer_only_once_per_batch():
    env = env_with(0.0, None, relation=0.45)
    play(env, [1, 1, 1])              # batch 700 to B, counter rejected, B withdraws
    play(env, [0])                    # VMI picks A
    assert env.stage == "DA"
    assert env.observe("DA")["action_mask"][1] == 0      # counter already used for this batch


def test_10000_random_episodes_always_finish_within_step_limit():
    env = ProcurementEnv("random")
    most = 0
    for seed in range(10_000):
        result = run_episode(env, RandomPolicy(seed), seed=seed)
        assert not result["truncated"], f"seed {seed} hit the safety limit"
        most = max(most, result["steps"])
    assert most < env.max_steps


def test_truncation_safety_net():
    env = ProcurementEnv("random")
    env.max_steps = 3
    result = run_episode(env, RandomPolicy(0), seed=0)
    assert result["truncated"] and result["steps"] == 3


def test_insufficient_capacity_is_not_consensus():
    """When no vendor can fit a batch the VMI mask is relaxed, so the final check must catch it."""
    import dataclasses
    from procurement_marl.agents.rule_based import RuleBasedPolicy
    sc = load_scenario()
    sc = dataclasses.replace(sc, vendors=[dataclasses.replace(v, capacity=100) for v in sc.vendors],
                             initial_cash=10**10)
    result = run_episode(ProcurementEnv(sc), RuleBasedPolicy(), seed=0, options={"scenario": sc})
    assert not result["consensus"]
    assert "kapasitas" in result["violations"]


def _with_vendors(**changes):
    import dataclasses
    sc = load_scenario()
    vendors = tuple(dataclasses.replace(v, **changes) for v in sc.vendors)
    return dataclasses.replace(sc, vendors=vendors, initial_cash=10**10)


def test_no_vendor_fits_is_explicit():
    """Nobody fits the batch: VMI may take exactly one vendor, and the round ends in conflict with a clear reason."""
    from procurement_marl.agents.rule_based import RuleBasedPolicy
    sc = _with_vendors(capacity=100)
    env = ProcurementEnv(sc)
    result = run_episode(env, RuleBasedPolicy(), seed=0, options={"scenario": sc})
    vmi = next(e for e in env.log if e["agent"] == "VMI")
    assert vmi["no_vendor_fits"] and sum(vmi["mask"]) == 0        # logged mask is the real one, not the relaxed one
    assert not result["consensus"]
    assert {"tanpa_pemasok_layak", "kapasitas"} <= set(result["violations"])


def test_no_vendor_fits_random_policy_terminates():
    sc = _with_vendors(capacity=100)
    env = ProcurementEnv(sc)
    for seed in range(50):
        result = run_episode(env, RandomPolicy(seed), seed=seed, options={"scenario": sc})
        assert not result["truncated"] and not result["consensus"]


def test_all_vendors_below_score_threshold_keeps_best_and_logs_it():
    import dataclasses
    from procurement_marl.agents.rule_based import RuleBasedPolicy
    from procurement_marl.scenario import eligible_vendors, score_fallback
    sc = dataclasses.replace(load_scenario(), min_composite_score=99.0)
    assert score_fallback(sc) and eligible_vendors(sc) == ["B"]
    env = ProcurementEnv(sc)
    result = run_episode(env, RuleBasedPolicy(), seed=0, options={"scenario": sc})
    assert not result["truncated"]
    assert all(e["score_fallback"] for e in env.log if e["agent"] == "VMI")
    assert not score_fallback(load_scenario())


def test_score_is_a_hard_filter_even_when_the_only_eligible_vendor_is_too_small():
    """The one vendor above the threshold has too little capacity: a lower-scored vendor must not be used instead."""
    import dataclasses
    sc = load_scenario()
    vendors = tuple(dataclasses.replace(v, capacity=100) if v.name == "B" else v for v in sc.vendors)
    sc = dataclasses.replace(sc, vendors=vendors, initial_cash=10**10)     # only B passes the score, B is too small
    env = ProcurementEnv(sc)
    env.reset(seed=0, options={"scenario": sc})
    env.step(0)                                                              # IRE: teruskan
    # nobody fits, so only the largest-capacity vendor (A) is allowed; C (score below threshold) stays blocked
    assert list(map(int, env.observe("VMI")["action_mask"])) == [1, 0, 0]


def test_cash_diagnosis_fixture_numbers():
    from procurement_marl.scenario import cash_diagnosis
    d = cash_diagnosis(load_scenario())
    assert (d["vendor"], d["lower_bound"], d["available"], d["shortfall"]) == ("B", 99_220_000, 80_000_000, 19_220_000)
    assert d["infeasible"]


def test_cash_diagnosis_is_a_sound_bound():
    """If the cash bound says infeasible, no deterministic plan may reach consensus."""
    import dataclasses
    from procurement_marl.scenario import cash_diagnosis
    hits = 0
    for seed in range(200):
        base = sample_scenario(seed)
        sc = dataclasses.replace(base, initial_cash=base.initial_cash - 90_000_000)    # make the bound trigger often
        if cash_diagnosis(sc)["infeasible"]:
            hits += 1
            assert not is_feasible(sc), seed
    assert hits > 0


def test_ire_display_labels_do_not_promise_a_clarification():
    from procurement_marl.env import IRE_ACTIONS, IRE_LABELS
    assert set(IRE_LABELS) == set(IRE_ACTIONS)
    assert "klarifikasi" not in IRE_LABELS["minta_klarifikasi"]
