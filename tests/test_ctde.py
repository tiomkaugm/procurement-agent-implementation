import csv
import sys
from pathlib import Path

import numpy as np
import torch

from procurement_marl.agents.ctde_ac import CTDEActorCritic
from procurement_marl.agents.random_agent import RandomPolicy
from procurement_marl.agents.rule_based import RuleBasedPolicy
from procurement_marl.env import AGENTS, DIMS, ProcurementEnv
from procurement_marl.evaluate import comparison_table, evaluate_policy, run_episode

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from train import train_ctde   # noqa: E402


def test_masked_logits_never_choose_invalid_action():
    agent = CTDEActorCritic(seed=0, greedy=False)
    obs = np.zeros(DIMS["DA"], dtype=np.float32)
    mask = np.array([0, 1, 0], dtype=np.int8)
    assert all(agent._choose("DA", obs, mask) == 1 for _ in range(50))
    probs = torch.softmax(agent.masked_logits("DA", torch.from_numpy(obs), torch.from_numpy(mask)), -1)
    assert probs[1] > 0.999999


def test_actors_do_not_share_parameters_and_see_only_local_obs():
    agent = CTDEActorCritic(seed=0)
    ids = [id(p) for a in AGENTS for p in agent.actors[a].parameters()]
    assert len(ids) == len(set(ids))
    for a in AGENTS:
        assert agent.actors[a][0].in_features == DIMS[a]     # local observation only


def test_acting_needs_only_actors():
    agent = CTDEActorCritic(seed=0)
    agent.critic = None                                       # critic is training-only
    env = ProcurementEnv("random")
    result = run_episode(env, agent, seed=1)
    assert not result["truncated"]


def test_update_changes_actors_and_critic():
    agent = CTDEActorCritic(seed=0, greedy=False)
    env = ProcurementEnv("random")
    before = [p.detach().clone() for p in agent.parameters()]
    agent.update([agent.collect_episode(env, seed=s)[0] for s in range(8)])
    assert any(not torch.equal(b, p) for b, p in zip(before, agent.parameters()))


def test_training_outputs_and_checkpoint_roundtrip(tmp_path):
    agent = train_ctde(episodes=128, seed=0, out=tmp_path, block=64, batch=32)
    rows = list(csv.DictReader(open(tmp_path / "curve.csv")))
    assert len(rows) >= 1 and (tmp_path / "config.json").exists()
    loaded = CTDEActorCritic.load(tmp_path / "checkpoint.pt")
    env = ProcurementEnv("random")
    a = run_episode(env, agent, seed=5)
    b = run_episode(env, loaded, seed=5)
    assert a["returns"] == b["returns"]


def test_training_is_reproducible(tmp_path):
    a = train_ctde(episodes=64, seed=2, out=tmp_path / "a", block=64, batch=32)
    b = train_ctde(episodes=64, seed=2, out=tmp_path / "b", block=64, batch=32)
    assert all(torch.equal(x, y) for x, y in zip(a.parameters(), b.parameters()))


def test_evaluate_metrics_and_table():
    results = {"Random": evaluate_policy(RandomPolicy(0), 20), "Rule-based": evaluate_policy(RuleBasedPolicy(), 20)}
    for m in results.values():
        assert 0 <= m["consensus"] <= 1 and m["rounds"] >= 1
    results["Rencana"] = results["Rule-based"]
    header, body = comparison_table(results, "Rencana")
    assert len(header) == len(body[0]) and body[-1][-1] == "1.00"
