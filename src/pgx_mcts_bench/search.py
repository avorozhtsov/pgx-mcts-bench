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


def _observation_batch(
    observations: list[np.ndarray], device: torch.device
) -> Tensor:
    array = np.stack(observations)
    return torch.from_numpy(array).permute(0, 3, 1, 2).to(
        device=device,
        dtype=torch.float32,
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
    consecutive_passes: int = 0
    move_count: int = 0

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
        return self.run_batch(
            states=[state],
            observations=[observation],
            legal_actions=[legal_actions],
            rngs=[rng],
            temperatures=[temperature],
            add_root_noise=add_root_noise,
        )[0]

    @torch.inference_mode()
    def run_batch(
        self,
        states: list[Any],
        observations: list[np.ndarray],
        legal_actions: list[np.ndarray],
        rngs: list[np.random.Generator],
        *,
        temperatures: list[float],
        add_root_noise: bool,
    ) -> list[SearchResult]:
        """Search several independent roots with batched network inference."""
        batch_size = len(states)
        if not (
            batch_size
            == len(observations)
            == len(legal_actions)
            == len(rngs)
            == len(temperatures)
        ):
            raise ValueError("All batched search inputs must have the same length")
        if batch_size == 0:
            return []

        self.network.eval()
        roots = [
            Node(
                prior=1.0,
                state=state,
                consecutive_passes=int(np.asarray(state._x.consecutive_pass_count)),
                move_count=int(np.asarray(state._x.step_count)),
            )
            for state in states
        ]
        if isinstance(self.network, AlphaZeroNet):
            root_values = self._expand_alphazero_batch(
                roots,
                observations,
                legal_actions,
            )
        else:
            root_values = self._expand_muzero_roots_batch(
                roots,
                observations,
                legal_actions,
            )

        if add_root_noise:
            for root, rng in zip(roots, rngs, strict=True):
                self._add_root_noise(root, rng)

        for _ in range(self.config.simulations):
            paths: list[list[Node]] = []
            actions: list[int] = []
            pending: list[int] = []
            leaf_values: list[float | None] = [None] * batch_size
            for index, root in enumerate(roots):
                node = root
                path = [node]
                action = -1
                while node.children:
                    action, node = self._select_child(path[-1])
                    path.append(node)
                    if node.visit_count == 0:
                        break
                paths.append(path)
                actions.append(action)
                if node.terminated:
                    leaf_values[index] = 0.0
                elif len(path) == 1:
                    leaf_values[index] = root_values[index]
                else:
                    pending.append(index)

            if isinstance(self.network, AlphaZeroNet):
                self._evaluate_alphazero_leaves_batch(
                    pending,
                    paths,
                    actions,
                    leaf_values,
                )
            else:
                self._evaluate_muzero_leaves_batch(
                    pending,
                    paths,
                    actions,
                    leaf_values,
                )
            for path, leaf_value in zip(paths, leaf_values, strict=True):
                assert leaf_value is not None
                self._backup(path, leaf_value)

        results = []
        for root, legal, temperature, rng in zip(
            roots,
            legal_actions,
            temperatures,
            rngs,
            strict=True,
        ):
            visits = np.zeros(self.game.config.action_size, dtype=np.int32)
            for action, child in root.children.items():
                visits[action] = child.visit_count
            policy = self._visit_policy(visits, legal, temperature)
            action = int(rng.choice(len(policy), p=policy))
            results.append(
                SearchResult(
                    policy=policy,
                    action=action,
                    root_value=root.value,
                    visits=visits,
                )
            )
        return results

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

    def _expand_alphazero_batch(
        self,
        nodes: list[Node],
        observations: list[np.ndarray],
        legal_actions: list[np.ndarray],
    ) -> list[float]:
        assert isinstance(self.network, AlphaZeroNet)
        logits, values = self.network(_observation_batch(observations, self.device))
        logits_np = logits.cpu().numpy()
        for index, node in enumerate(nodes):
            self._expand_children(
                node,
                logits_np[index],
                legal_actions[index].copy(),
            )
        return [float(value) for value in values.cpu().numpy()]

    def _expand_muzero_roots_batch(
        self,
        nodes: list[Node],
        observations: list[np.ndarray],
        legal_actions: list[np.ndarray],
    ) -> list[float]:
        assert isinstance(self.network, MuZeroNet)
        hidden, logits, values, _, _ = self.network.initial_inference(
            _observation_batch(observations, self.device)
        )
        logits_np = logits.cpu().numpy()
        for index, node in enumerate(nodes):
            node.hidden = hidden[index : index + 1].detach()
            self._expand_children(
                node,
                logits_np[index],
                legal_actions[index].copy(),
            )
        return [float(value) for value in values.cpu().numpy()]

    def _evaluate_alphazero_leaves_batch(
        self,
        pending: list[int],
        paths: list[list[Node]],
        actions: list[int],
        leaf_values: list[float | None],
    ) -> None:
        if not pending:
            return
        expand_indexes: list[int] = []
        observations: list[np.ndarray] = []
        legal_actions: list[np.ndarray] = []
        for index in pending:
            parent = paths[index][-2]
            node = paths[index][-1]
            transition = self.game.step(parent.state, actions[index])
            node.state = transition.state
            node.reward = transition.reward
            node.terminated = transition.terminated
            node.consecutive_passes = int(
                np.asarray(transition.state._x.consecutive_pass_count)
            )
            node.move_count = int(np.asarray(transition.state._x.step_count))
            if node.terminated:
                leaf_values[index] = 0.0
            else:
                expand_indexes.append(index)
                observations.append(transition.observation)
                legal_actions.append(transition.legal_actions)
        if not expand_indexes:
            return
        nodes = [paths[index][-1] for index in expand_indexes]
        values = self._expand_alphazero_batch(nodes, observations, legal_actions)
        for index, value in zip(expand_indexes, values, strict=True):
            leaf_values[index] = value

    def _evaluate_muzero_leaves_batch(
        self,
        pending: list[int],
        paths: list[list[Node]],
        actions: list[int],
        leaf_values: list[float | None],
    ) -> None:
        if not pending:
            return
        assert isinstance(self.network, MuZeroNet)
        parents = [paths[index][-2] for index in pending]
        assert all(parent.hidden is not None for parent in parents)
        exact_transitions = (
            [
                self.game.step(parent.state, actions[index])
                for parent, index in zip(parents, pending, strict=True)
            ]
            if self.config.muzero_exact_rules
            else None
        )
        parent_hidden = torch.cat(
            [parent.hidden for parent in parents if parent.hidden is not None],
            dim=0,
        )
        action_tensor = torch.tensor(
            [actions[index] for index in pending],
            dtype=torch.long,
            device=self.device,
        )
        hidden, rewards, logits, values, legal_logits, terminal_logits = (
            self.network.recurrent_inference(parent_hidden, action_tensor)
        )
        logits_np = logits.cpu().numpy()
        legal_np = legal_logits.cpu().numpy()
        terminal_np = torch.sigmoid(terminal_logits).cpu().numpy()
        for batch_index, index in enumerate(pending):
            parent = parents[batch_index]
            node = paths[index][-1]
            node.hidden = hidden[batch_index : batch_index + 1].detach()
            if exact_transitions is not None:
                transition = exact_transitions[batch_index]
                node.state = transition.state
                node.reward = transition.reward
                node.terminated = transition.terminated
                node.consecutive_passes = int(
                    np.asarray(transition.state._x.consecutive_pass_count)
                )
                node.move_count = int(np.asarray(transition.state._x.step_count))
                legal = transition.legal_actions
            else:
                node.reward = float(rewards[batch_index].item())
                action = actions[index]
                node.consecutive_passes = (
                    parent.consecutive_passes + 1
                    if action == self.game.config.board_size**2
                    else 0
                )
                node.move_count = parent.move_count + 1
                known_terminal = (
                    node.consecutive_passes >= 2
                    or node.move_count >= self.game.config.max_moves
                )
                node.terminated = known_terminal or bool(
                    terminal_np[batch_index] >= 0.5
                )
                legal = legal_np[batch_index] >= 0.0
                legal[-1] = True
            if node.terminated:
                leaf_values[index] = 0.0
                continue
            self._expand_children(node, logits_np[batch_index], legal)
            leaf_values[index] = float(values[batch_index].item())

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
