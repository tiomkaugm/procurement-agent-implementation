"""Rule-based policy on the report fixture must reproduce the trace of the report."""

import pytest

from procurement_marl.agents.rule_based import RuleBasedPolicy
from procurement_marl.env import ProcurementEnv
from procurement_marl.evaluate import run_episode
from procurement_marl.report import ReportReplayEnv


@pytest.fixture(scope="module")
def trace():
    env = ReportReplayEnv()
    result = run_episode(env, RuleBasedPolicy(), seed=0)
    return result, env.log


def pick(log, rnd, agent, batch=None):
    return [e for e in log if e["round"] == rnd and e["agent"] == agent
            and (batch is None or e.get("batch") == batch)]


def test_round1(trace):
    _, log = trace
    ire = pick(log, 1, "IRE")[0]
    assert ire["action"] == "teruskan"
    assert ire["batches"] == [{"qty": 1000, "month": 1}]
    vmi = pick(log, 1, "VMI")[0]
    assert vmi["masked"] == {"A": "kapasitas", "C": "skor"}
    assert vmi["vendor"] == "B" and vmi["estimate"] == 102_000_000     # no agreed price yet: list
    da = pick(log, 1, "DA")[0]
    assert da["action"] == "penawaran_awal" and da["price"] == 89_500
    slm = pick(log, 1, "SLM")[0]
    assert slm["action"] == "bayar_cepat"
    assert slm["goods_value"] == 89_500_000 and slm["discount"] == 1_790_000
    assert slm["goods_after_discount"] == 87_710_000 and slm["total"] == 99_710_000
    check = pick(log, 1, "ENV")[0]
    assert check["budget_left"] == 290_000
    assert check["cash"][0] == -19_710_000
    assert check["violations"] == ["kas_negatif"]
    assert not check["consensus"] and not check["done"]


def test_round2_batch1(trace):
    _, log = trace
    ire = pick(log, 2, "IRE")[0]
    assert ire["action"] == "minta_klarifikasi"
    assert ire["batches"] == [{"qty": 700, "month": 1}, {"qty": 300, "month": 2}]
    vmi = pick(log, 2, "VMI", 0)[0]
    assert vmi["vendor"] == "B" and vmi["estimate"] == 71_050_000   # last agreed price 89.500
    da = pick(log, 2, "DA", 0)[0]
    assert da["price"] == 89_500
    slm = pick(log, 2, "SLM", 0)[0]
    assert slm["goods_value"] == 62_650_000 and slm["discount"] == 1_253_000
    assert slm["goods_after_discount"] == 61_397_000
    assert slm["logistics"] == 8_400_000 and slm["total"] == 69_797_000
    assert slm["schedule"] == {1: 69_797_000}


def test_round2_batch2(trace):
    _, log = trace
    vmi = pick(log, 2, "VMI", 1)[0]
    assert vmi["vendor"] == "A" and vmi["estimate"] == 32_100_000   # no agreed price with A yet: list   # backup vendor
    da = pick(log, 2, "DA", 1)[0]
    assert da["price"] == 95_000
    slm = pick(log, 2, "SLM", 1)[0]
    assert slm["discount"] == 0 and slm["total"] == 30_600_000
    assert slm["schedule"] == {2: 30_600_000}


def test_round2_check(trace):
    _, log = trace
    check = pick(log, 2, "ENV")[0]
    assert check["cash"][:2] == [10_203_000, -20_397_000]
    assert check["cash"][0] < 60_000_000            # below minimum: only a warning
    assert "kas_di_bawah_minimum" in check["warnings"]
    assert check["total"] == 100_397_000
    assert check["total"] - 100_000_000 == 397_000
    assert check["violations"] == ["kas_negatif", "anggaran"]
    assert check["done"] and check["stop_reason"] == "report_revision_required"


def test_six_coordination_stages_end_with_revision_not_repeated_cycles(trace):
    result, log = trace
    checks = [e for e in log if e.get("event") == "cek_batasan"]
    assert len(checks) == 2
    assert [e["coordination_step"] for e in log] == [0, 0, 0, 0, 1, 2, 3, 3, 4, 4, 5, 5, 6]
    assert [e["agent"] for e in log if e["round"] == 2] == ["IRE", "VMI", "VMI", "DA", "DA", "SLM", "SLM", "ENV"]
    assert checks[-1]["done"] and not result["consensus"]
    assert result["rounds"] == 2 and result["coordination_steps"] == 6
    assert result["stop_reason"] == "report_revision_required"
    stage5 = [e for e in log if e["coordination_step"] == 5]
    assert stage5[0]["cash"][0] == 10_203_000
    assert stage5[-1]["cash"][:2] == [10_203_000, -20_397_000]


def test_general_rule_based_stops_when_plan_and_state_repeat():
    result = run_episode(ProcurementEnv("report"), RuleBasedPolicy(), seed=0)
    assert result["rounds"] == 3
    assert result["stop_reason"] == "no_progress"
    assert not result["consensus"]
