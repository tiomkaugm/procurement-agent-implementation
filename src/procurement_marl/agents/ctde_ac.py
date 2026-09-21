"""CTDE actor-critic: local actors, one centralized critic (PyTorch, CPU).

- Actors: one small MLP per agent (parameters are NOT shared). Each actor only sees the
  local observation of its own agent. Invalid actions are removed by masking the logits.
- Critic: one MLP V(s) on the full state. It is used only during training.
- Advantage: GAE over the joint sequence of turns. The reward of a step is the weighted sum of
  all agents' rewards, so all actors are pushed toward the same team objective.
- Execution: only the actors are needed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from ..env import AGENTS, DIMS, STATE_DIM, ProcurementEnv

NEG = -1e9


def mlp(n_in: int, hidden: int, n_out: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(n_in, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(),
                         nn.Linear(hidden, n_out))


class CTDEActorCritic:
    def __init__(self, hidden: int = 64, lr: float = 1e-3, gamma: float = 0.99, lam: float = 0.95,
                 entropy_coef: float = 0.01, value_coef: float = 0.5, seed: int = 0, greedy: bool = True):
        torch.manual_seed(seed)
        torch.set_num_threads(1)
        self.rng = np.random.default_rng(seed)
        self.hp = dict(hidden=hidden, lr=lr, gamma=gamma, lam=lam, entropy_coef=entropy_coef, value_coef=value_coef)
        self.gamma, self.lam, self.entropy_coef, self.value_coef = gamma, lam, entropy_coef, value_coef
        self.greedy = greedy
        self.actors = nn.ModuleDict({a: mlp(DIMS[a], hidden, 3) for a in AGENTS})
        self.critic = mlp(STATE_DIM, 2 * hidden, 1)
        self.optimizer = torch.optim.Adam(list(self.actors.parameters()) + list(self.critic.parameters()), lr=lr)

    # ---------------------------------------------------------------- acting
    def masked_logits(self, agent: str, obs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.actors[agent](obs).masked_fill(mask == 0, NEG)

    def act(self, env: ProcurementEnv, agent: str) -> int:
        """Policy interface: uses only the local observation of `agent`."""
        o = env.observe(agent)
        return self._choose(agent, o["observation"], o["action_mask"])

    def _choose(self, agent: str, obs: np.ndarray, mask: np.ndarray) -> int:
        with torch.no_grad():
            logits = self.masked_logits(agent, torch.from_numpy(obs), torch.from_numpy(mask))
            if self.greedy:
                return int(logits.argmax())
            probs = torch.softmax(logits, -1).numpy().astype(np.float64)
            return int(self.rng.choice(3, p=probs / probs.sum()))

    # ------------------------------------------------------------- learning
    def collect_episode(self, env: ProcurementEnv, seed: int) -> tuple[list[dict], bool]:
        """Play one episode with sampled actions. Returns (steps, consensus)."""
        env.reset(seed=seed)
        steps = []
        while True:
            agent = env.agent_selection
            o = env.observe(agent)
            state = env.state()
            action = self._choose(agent, o["observation"], o["action_mask"])
            env.step(action)
            reward = sum(env.weights[x] * env.rewards[x] for x in env.rewards)
            done = any(env.terminations.values()) or any(env.truncations.values())
            steps.append({"agent": agent, "obs": o["observation"], "mask": o["action_mask"], "action": action,
                          "state": state, "reward": reward, "done": done})
            if done:
                break
        last = next((e for e in reversed(env.log) if e.get("event") == "cek_batasan"), None)
        return steps, bool(last and last["consensus"])

    def update(self, episodes: list[list[dict]]) -> dict[str, float]:
        """One gradient step on a batch of episodes."""
        steps = [s for ep in episodes for s in ep]
        states = torch.from_numpy(np.stack([s["state"] for s in steps]))
        values = self.critic(states).squeeze(-1)
        v = values.detach().numpy()

        # GAE, computed backwards inside each episode (value after the last step is 0)
        adv = np.zeros(len(steps), dtype=np.float32)
        gae, next_v = 0.0, 0.0
        for t in reversed(range(len(steps))):
            if steps[t]["done"]:
                gae, next_v = 0.0, 0.0
            delta = steps[t]["reward"] + self.gamma * next_v - v[t]
            gae = delta + self.gamma * self.lam * gae
            adv[t], next_v = gae, v[t]
        returns = torch.from_numpy(adv + v)
        adv_t = torch.from_numpy((adv - adv.mean()) / (adv.std() + 1e-8))

        actor_loss, entropy = torch.zeros(()), torch.zeros(())
        for agent in AGENTS:
            idx = [i for i, s in enumerate(steps) if s["agent"] == agent]
            if not idx:
                continue
            obs = torch.from_numpy(np.stack([steps[i]["obs"] for i in idx]))
            mask = torch.from_numpy(np.stack([steps[i]["mask"] for i in idx]))
            actions = torch.tensor([steps[i]["action"] for i in idx])
            logp = torch.log_softmax(self.masked_logits(agent, obs, mask), -1)
            chosen = logp.gather(1, actions[:, None]).squeeze(1)
            actor_loss = actor_loss - (chosen * adv_t[idx]).sum() / len(steps)
            entropy = entropy - (logp.exp() * logp).sum() / len(steps)
        critic_loss = ((values - returns) ** 2).mean()
        loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), 0.5)
        self.optimizer.step()
        return {"actor_loss": float(actor_loss.detach()), "critic_loss": float(critic_loss.detach()),
                "entropy": float(entropy.detach())}

    def parameters(self):
        return list(self.actors.parameters()) + list(self.critic.parameters())

    # ------------------------------------------------------------ checkpoint
    def save(self, path: str | Path) -> None:
        torch.save({"hp": self.hp, "actors": self.actors.state_dict(), "critic": self.critic.state_dict()}, path)

    @classmethod
    def load(cls, path: str | Path, seed: int = 0) -> "CTDEActorCritic":
        raw = torch.load(path)
        agent = cls(seed=seed, **raw["hp"])
        agent.actors.load_state_dict(raw["actors"])
        agent.critic.load_state_dict(raw["critic"])
        return agent
