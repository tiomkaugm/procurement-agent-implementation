import csv
import sys
from pathlib import Path

import numpy as np

from procurement_marl.agents.iql import IQL
from procurement_marl.env import ProcurementEnv
from procurement_marl.evaluate import run_episode

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from train import train_iql   # noqa: E402


def test_q_update_formula():
    agent = IQL(alpha=0.5, gamma=0.9)
    agent.q["SLM"][(0, 0, 0, 0)] = np.array([1.0, 0.0, 0.0])
    agent.q["SLM"][(1, 1, 1, 1)] = np.array([0.0, 4.0, 10.0])
    # action 2 is masked in the next state, so the max is 4, not 10
    agent.update("SLM", (0, 0, 0, 0), 0, 2.0, (1, 1, 1, 1), np.array([1, 1, 0]))
    assert agent.q["SLM"][(0, 0, 0, 0)][0] == 1.0 + 0.5 * (2.0 + 0.9 * 4.0 - 1.0)
    # terminal update has no bootstrap
    agent.update("SLM", (0, 0, 0, 0), 1, 3.0, None, None)
    assert agent.q["SLM"][(0, 0, 0, 0)][1] == 0.5 * 3.0


def test_select_never_returns_masked_action():
    agent = IQL(seed=1)
    mask = np.array([0, 1, 0], dtype=np.int8)
    assert all(agent.select("DA", (0, 0, 0, 0), mask, epsilon=eps) == 1 for eps in (0.0, 1.0) for _ in range(50))


def test_training_writes_outputs_and_checkpoint_roundtrip(tmp_path):
    agent = train_iql(episodes=300, seed=0, out=tmp_path, block=100)
    rows = list(csv.DictReader(open(tmp_path / "curve.csv")))
    assert [r["episode"] for r in rows] == ["100", "200", "300"]
    assert (tmp_path / "config.json").exists()
    loaded = IQL.load(tmp_path / "checkpoint.json")
    for a in agent.q:
        assert agent.q[a].keys() == loaded.q[a].keys()
        for s in agent.q[a]:
            assert np.allclose(agent.q[a][s], loaded.q[a][s])


def test_training_is_reproducible(tmp_path):
    a = train_iql(episodes=200, seed=3, out=tmp_path / "a", block=100)
    b = train_iql(episodes=200, seed=3, out=tmp_path / "b", block=100)
    assert all(np.allclose(a.q[ag][s], b.q[ag][s]) for ag in a.q for s in a.q[ag])


def test_greedy_policy_plays_valid_full_episodes(tmp_path):
    agent = train_iql(episodes=200, seed=0, out=tmp_path, block=100)
    env = ProcurementEnv("random")
    for seed in range(20):
        result = run_episode(env, agent, seed=seed)     # env asserts every action is unmasked
        assert not result["truncated"]
