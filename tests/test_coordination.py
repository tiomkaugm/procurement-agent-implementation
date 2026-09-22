"""Regressions for payment handoff, stop reasons and one-cycle evaluation."""

import pytest

from procurement_marl.env import ProcurementEnv
from procurement_marl.evaluate import run_episode, single_round_plan_row
from procurement_marl.oracle import PlanPolicy, expected_return
from procurement_marl.presentation import log_rows, round_payments
from procurement_marl.scenario import load_scenario


@pytest.mark.parametrize("accepted", [False, True])
def test_terms_are_delegated_before_payment_schedule_is_committed(accepted):
    env = ProcurementEnv("report")
    env.stoch["p_termin_override"] = float(accepted)
    env.reset(seed=0)
    for action in (0, 1, 0):
        env.step(action)
    assert env.committed == {}
    env.step(2)
    events = [e for e in env.log if e.get("event") in ("term_request", "term_response", "payment_plan")]
    assert [(e["agent"], e["event"]) for e in events] == [
        ("SLM", "term_request"), ("DA", "term_response"), ("SLM", "payment_plan")]
    assert events[1]["accepted"] is accepted
    assert events[2]["term_accepted"] is accepted
    request_index = env.log.index(events[0]) + 1
    response_index = env.log.index(events[1]) + 1
    plan_index = env.log.index(events[2]) + 1
    assert round_payments(env.log, request_index) == {}
    assert round_payments(env.log, response_index) == {}
    payments = round_payments(env.log, plan_index)
    assert payments == events[2]["schedule"]
    assert sum(payments.values()) == 101_500_000
    assert payments == ({1: 12_000_000, 2: 44_750_000, 3: 44_750_000} if accepted
                        else {1: 12_000_000, 2: 89_500_000})
    assert all(row["Keputusan"] for row in log_rows(env.log))


def test_single_cycle_has_terminal_penalty_once_and_matches_oracle():
    scenario = load_scenario()
    plan = (0, ((1, 0, 0),))
    env = ProcurementEnv(scenario)
    result = run_episode(env, PlanPolicy(plan), seed=0, stop_at_first_check=True)
    assert result["rounds"] == 1 and result["stop_reason"] == "single_round_limit"
    assert result["log"][-1]["done"]
    assert result["team_return"] == pytest.approx(expected_return(ProcurementEnv(scenario), scenario, plan))
    # IRE +1, VMI +.9, DA saving 500/90000*10, SLM discount/budget-2;
    # no consensus adds -2 to each of the four agents, once.
    assert result["team_return"] == pytest.approx(1 + .9 + 500 / 90_000 * 10 + .0179 - 2 - 8)


def test_single_cycle_metrics_do_not_replay_failed_plan(monkeypatch):
    import procurement_marl.oracle as oracle
    import procurement_marl.scenario as scenario_module
    plan = (0, ((1, 0, 0),))
    monkeypatch.setattr(oracle, "best_single_round_plan", lambda scenario, env: (-8.0, plan))
    monkeypatch.setattr(scenario_module, "sample_scenario", lambda seed: load_scenario())
    row = single_round_plan_row(2)
    assert row["rounds"] == 1
    assert row["consensus"] == 0


def test_stochastic_policy_is_not_stopped_for_repeated_actions():
    # Identical successful term draws can recur. Only a deterministic baseline opts in.
    env = ProcurementEnv("report")
    env.stoch["p_termin_override"] = 1.0
    result = run_episode(env, PlanPolicy((0, ((1, 0, 2),))), seed=0)
    assert result["rounds"] == env.max_rounds
    assert result["stop_reason"] == "cycle_limit"


def test_step_limit_is_explicit_in_result_and_export():
    env = ProcurementEnv("report")
    env.max_steps = 3
    result = run_episode(env, PlanPolicy((0, ((1, 0, 0),))), seed=0)
    assert result["truncated"] and result["stop_reason"] == "step_limit"
    assert log_rows(result["log"])[-1]["Keputusan"] == "SIMULASI DIHENTIKAN"
