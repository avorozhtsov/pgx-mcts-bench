from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from torch import Tensor

from pgx_mcts_bench.config import SearchConfig
from pgx_mcts_bench.exploration import exploration_bonus
from pgx_mcts_bench.game import Go6x6
from pgx_mcts_bench.networks import AlphaZeroNet, MuZeroNet


def _observation_tensor(observation: np.ndarray, device: torch.device) -> Tensor:
    return (
        torch.from_numpy(observation)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(device=device, dtype=torch.float32)
    )


def _masked_softmax(logits: np.ndarray, legal: np.ndarray) -> np.ndarray:
    legal = np.asarray(legal, dtype=bool)
    if not legal.any():
        legal[-1] = True
    shifted = logits - np.max(logits[legal])
    weights = np.where(legal, np.exp(shifted), 0.0)
    return weights / weights.sum()


@dataclass
class Node:
    prior: float
    reward: float = 0.0
    visit_count: int = 0
    value_sum: float = 0.0
    children: dict[int, Node] = field(default_factory=dict)
    state: Any = None
    hidden: Tensor | None = None
    terminated: bool = False

    @property
    def value(self) -> float:
        return self.value_sum / self.visit_count if self.visit_count else 0.0


@dataclass(frozen=True)
class SearchResult:
    policy: np.ndarray
    action: int
    root_value: float
    visits: np.ndarray


class NeuralMCTS:
    def __init__(
        self,
        game: Go6x6,
        network: AlphaZeroNet | MuZeroNet,
        config: SearchConfig,
        device: str = "cpu",
    ):
        self.game = game
        self.network = network
        self.config = config
        self.device = torch.device(device)

    @torch.inference_mode()
    def run(
        self,
        state: Any,
        observation: np.ndarray,
        legal_actions: np.ndarray,
        rng: np.random.Generator,
        *,
        temperature: float,
        add_root_noise: bool,
    ) -> SearchResult:
        self.network.eval()
        root = Node(prior=1.0, state=state)
        if isinstance(self.network, AlphaZeroNet):
            value = self._expand_alphazero(root, observation, legal_actions)
        else:
            value = self._expand_muzero_root(root, observation, legal_actions)

        if add_root_noise:
            self._add_root_noise(root, rng)

        for _ in range(self.config.simulations):
            node = root
            path = [node]
            action = -1
            while node.children:
                action, node = self._select_child(path[-1])
                path.append(node)
                if node.visit_count == 0:
                    break

            parent = path[-2] if len(path) > 1 else None
            if parent is None:
                leaf_value = value
            elif isinstance(self.network, AlphaZeroNet):
                leaf_value = self._evaluate_alphazero_leaf(parent, node, action)
            else:
                leaf_value = self._evaluate_muzero_leaf(parent, node, action)
            self._backup(path, leaf_value)

        visits = np.zeros(self.game.config.action_size, dtype=np.int32)
        for action, child in root.children.items():
            visits[action] = child.visit_count
        policy = self._visit_policy(visits, legal_actions, temperature)
        action = int(rng.choice(len(policy), p=policy))
        return SearchResult(policy=policy, action=action, root_value=root.value, visits=visits)

    def _select_child(self, parent: Node) -> tuple[int, Node]:
        actions = np.fromiter(parent.children.keys(), dtype=np.int64)
        children = [parent.children[int(a)] for a in actions]
        priors = np.array([child.prior for child in children], dtype=np.float64)
        visits = np.array([child.visit_count for child in children], dtype=np.float64)
        q = np.array(
            [child.reward - self.config.discount * child.value for child in children],
            dtype=np.float64,
        )
        u = exploration_bonus(
            self.config.exploration, priors, parent.visit_count, visits, self.config
        )
        # The tiny prior term resolves the all-zero score at the first PUCT visit.
        index = int(np.argmax(q + u + 1e-12 * priors))
        return int(actions[index]), children[index]

    def _expand_children(self, node: Node, logits: np.ndarray, legal: np.ndarray) -> None:
        priors = _masked_softmax(logits, legal)
        node.children = {
            int(action): Node(prior=float(priors[action]))
            for action in np.flatnonzero(legal)
        }

    def _expand_alphazero(
        self, node: Node, observation: np.ndarray, legal_actions: np.ndarray
    ) -> float:
        obs = _observation_tensor(observation, self.device)
        logits, value = self.network(obs)
        self._expand_children(node, logits[0].cpu().numpy(), legal_actions.copy())
        return float(value.item())

    def _expand_muzero_root(
        self, node: Node, observation: np.ndarray, legal_actions: np.ndarray
    ) -> float:
        assert isinstance(self.network, MuZeroNet)
        obs = _observation_tensor(observation, self.device)
        hidden, logits, value, _ = self.network.initial_inference(obs)
        node.hidden = hidden.detach()
        self._expand_children(node, logits[0].cpu().numpy(), legal_actions.copy())
        return float(value.item())

    def _evaluate_alphazero_leaf(self, parent: Node, node: Node, action: int) -> float:
        transition = self.game.step(parent.state, action)
        node.state = transition.state
        node.reward = transition.reward
        node.terminated = transition.terminated
        if node.terminated:
            # The edge already carries the terminal reward, so there is no bootstrap value.
            return 0.0
        return self._expand_alphazero(node, transition.observation, transition.legal_actions)

    def _evaluate_muzero_leaf(self, parent: Node, node: Node, action: int) -> float:
        assert isinstance(self.network, MuZeroNet)
        assert parent.hidden is not None
        action_tensor = torch.tensor([action], dtype=torch.long, device=self.device)
        hidden, reward, logits, value, legal_logits = self.network.recurrent_inference(
            parent.hidden, action_tensor
        )
        node.hidden = hidden.detach()
        node.reward = float(reward.item())
        predicted_legal = legal_logits[0].cpu().numpy() >= 0.0
        predicted_legal[-1] = True  # Passing is always legal in Pgx Go.
        self._expand_children(node, logits[0].cpu().numpy(), predicted_legal)
        return float(value.item())

    def _backup(self, path: list[Node], leaf_value: float) -> None:
        value = leaf_value
        for index in range(len(path) - 1, -1, -1):
            node = path[index]
            node.visit_count += 1
            node.value_sum += value
            if index > 0:
                value = node.reward - self.config.discount * value

    def _add_root_noise(self, root: Node, rng: np.random.Generator) -> None:
        if not root.children or self.config.root_exploration_fraction <= 0:
            return
        actions = list(root.children)
        noise = rng.dirichlet(np.full(len(actions), self.config.root_dirichlet_alpha))
        fraction = self.config.root_exploration_fraction
        for action, sample in zip(actions, noise, strict=True):
            child = root.children[action]
            child.prior = (1.0 - fraction) * child.prior + fraction * float(sample)

    @staticmethod
    def _visit_policy(
        visits: np.ndarray, legal: np.ndarray, temperature: float
    ) -> np.ndarray:
        visits = visits.astype(np.float64)
        if temperature <= 1e-8:
            policy = np.zeros_like(visits)
            legal_visits = np.where(legal, visits, -1.0)
            policy[int(np.argmax(legal_visits))] = 1.0
            return policy
        weights = np.where(legal, visits ** (1.0 / temperature), 0.0)
        if weights.sum() == 0:
            weights = legal.astype(np.float64)
        return weights / weights.sum()
