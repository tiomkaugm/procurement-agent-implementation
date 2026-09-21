"""Reproduces every number of Bab 6 from the report."""

from dataclasses import replace

import pytest

from procurement_marl.costs import (
    PAY_DUE, PAY_FAST, PAY_SPLIT, batch_total, cash_after, cash_balances,
    discount_amount, goods_value, logistics_cost, payment_schedule,
)
from procurement_marl.rewards import collective_objective
from procurement_marl.scenario import (
    composite_scores, eligible_vendors, load_scenario, sample_scenario,
)

S = load_scenario()
A, B, C = S.vendor("A"), S.vendor("B"), S.vendor("C")


def total(v, qty, price=None, disc=0):
    return batch_total(qty, price or v.list_price, v.transport, v.risk, disc)


# ---- 6.1 list-price comparison -------------------------------------------
def test_list_price_totals():
    assert total(A, 1000) == 107_000_000
    assert total(B, 1000) == 102_000_000
    assert total(A, 800) + total(B, 200) == 106_000_000


# ---- 6.5 single batch of 1.000 units to B --------------------------------
def test_single_batch_with_discount():
    cost = total(B, 1000, B.initial_offer, B.discount_pct)
    assert cost == 99_710_000
    assert S.budget - cost == 290_000
    cash = S.initial_cash - cost
    assert cash == 50_290_000
    assert cash - 70_000_000 == -19_710_000


def test_goods_only_shortfall():
    value = goods_value(1000, B.initial_offer)
    goods_after = value - discount_amount(value, B.discount_pct)
    assert S.initial_cash - goods_after - 70_000_000 == -7_710_000


# ---- 6.5 staged scenario: 700 B + 300 A ----------------------------------
def test_staged_estimates():
    assert batch_total(700, B.initial_offer, B.transport, B.risk) == 71_050_000
    assert total(A, 300) == 32_100_000


def test_batch_one_breakdown():
    value = goods_value(700, B.initial_offer)
    disc = discount_amount(value, B.discount_pct)
    assert value == 62_650_000
    assert disc == 1_253_000
    assert value - disc == 61_397_000
    assert total(B, 700, B.initial_offer, B.discount_pct) == 69_797_000


def test_staged_cash_and_budget():
    b1 = payment_schedule(PAY_FAST, 1, 700, B.initial_offer, B.transport, B.risk, B.discount_pct)
    b2 = payment_schedule(PAY_FAST, 2, 300, A.initial_offer, A.transport, A.risk, A.discount_pct)
    assert b1 == {1: 69_797_000}
    assert b2 == {2: 30_600_000}
    payments = {**b1, **b2}
    k = cash_balances(S.initial_cash, list(S.inflows), list(S.other_needs), payments)
    assert k[0] == 10_203_000      # K1
    assert k[1] == -20_397_000     # K2
    assert sum(payments.values()) == 100_397_000
    assert sum(payments.values()) - S.budget == 397_000


def test_cash_after_formula():
    assert cash_after(150_000_000, 0, 69_797_000, 70_000_000) == 10_203_000


# ---- payment modes -------------------------------------------------------
def test_payment_modes():
    args = (700, 89_500, 7_000, 5_000)
    logistics = logistics_cost(700, 7_000, 5_000)
    assert payment_schedule(PAY_DUE, 1, *args) == {1: logistics, 2: 62_650_000}
    split = payment_schedule(PAY_SPLIT, 2, *args)
    assert split == {2: logistics, 3: 31_325_000, 4: 31_325_000}


def test_discount_rounds_to_nearest_rupiah():
    assert discount_amount(1_250, 2) == 25       # 25.0
    assert discount_amount(1_275, 2) == 26       # 25.5 rounds up
    assert discount_amount(1_274, 2) == 25       # 25.48 rounds down


# ---- 6.6 collective objective --------------------------------------------
W = {"IRE": 1, "VMI": 1, "DA": 1, "SLM": 1}


def test_collective_objective_table_66():
    with_consensus = [{"IRE": 25, "VMI": 35, "DA": 40, "SLM": 50}]
    without = [{"IRE": 10, "VMI": 50, "DA": 30, "SLM": -100}]
    assert collective_objective(with_consensus, W, gamma=1.0) == 150
    assert collective_objective(without, W, gamma=1.0) == -10


def test_collective_objective_discounting():
    steps = [{"IRE": 1}, {"IRE": 1}]
    assert collective_objective(steps, {"IRE": 1}, gamma=0.5) == 1.5


# ---- 6.2 composite score --------------------------------------------------
def test_composite_scores_match_report():
    s = composite_scores(S)
    assert s["A"] == pytest.approx(6.60, abs=0.005)
    assert s["B"] == pytest.approx(7.19, abs=0.005)
    assert s["C"] == pytest.approx(5.41, abs=0.005)


def test_ranking_and_only_c_dropped():
    s = composite_scores(S)
    assert s["B"] > s["A"] > s["C"]
    assert eligible_vendors(S) == ["A", "B"]


def test_ranking_does_not_depend_on_reputation():
    flat = replace(S, vendors=tuple(replace(v, reputation=7.0) for v in S.vendors))
    s = composite_scores(flat)
    assert s["B"] > s["A"] > s["C"]


def test_fallback_keeps_one_vendor():
    strict = replace(S, min_composite_score=9.9)
    assert eligible_vendors(strict) == ["B"]


# ---- random generator -----------------------------------------------------
def test_sample_scenario_is_seeded_and_valid():
    assert sample_scenario(7) == sample_scenario(7)
    assert sample_scenario(7) != sample_scenario(8)
    sc = sample_scenario(3)
    assert 600 <= sc.quantity <= 1400
    assert 0 < sc.urgent_quantity < sc.quantity
    assert all(v.floor_price <= v.initial_offer <= v.list_price for v in sc.vendors)
