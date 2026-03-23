import pickle
from collections import deque
from pathlib import Path

import numpy as np
import torch
from torch import nn


class BaseAgent:
    def choose_action(self, state: np.ndarray) -> int:
        raise NotImplementedError

    def update(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        raise NotImplementedError

    def save(self, path: str) -> None:
        raise NotImplementedError

    @classmethod
    def load(cls, path: str) -> "BaseAgent":
        with open(path, "rb") as f:
            payload = pickle.load(f)

        agent_type = payload.get("agent_type")
        if agent_type == "random":
            return RandomAgent.from_payload(payload)
        if agent_type == "q_learning":
            return QLearningAgent.from_payload(payload)
        raise ValueError(f"Unsupported agent type in checkpoint: {agent_type}")

    @staticmethod
    def _write_payload(path: str, payload: dict) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            pickle.dump(payload, f)


class RandomAgent(BaseAgent):
    def __init__(self, n_actions: int = 4):
        self.n_actions = n_actions

    def choose_action(self, state: np.ndarray) -> int:
        return int(np.random.randint(0, self.n_actions))

    def update(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        pass

    def save(self, path: str) -> None:
        self._write_payload(path, {"agent_type": "random", "n_actions": self.n_actions})

    @classmethod
    def from_payload(cls, payload: dict) -> "RandomAgent":
        return cls(n_actions=payload["n_actions"])


def build_mlp(input_dim: int, output_dim: int, hidden_dims: list[int]) -> nn.Sequential:
    layers: list[nn.Module] = []
    in_features = input_dim
    for hidden_dim in hidden_dims:
        layers.extend([nn.Linear(in_features, hidden_dim), nn.ReLU()])
        in_features = hidden_dim
    layers.append(nn.Linear(in_features, output_dim))
    return nn.Sequential(*layers)


class CNNQNetwork(nn.Module):
    def __init__(
        self,
        input_shape: tuple[int, int, int],
        n_actions: int,
        conv_channels: list[int],
        head_hidden_dims: list[int],
    ):
        super().__init__()
        height, width, channels = input_shape

        conv_layers: list[nn.Module] = []
        in_channels = channels
        for out_channels in conv_channels:
            conv_layers.extend(
                [
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                ],
            )
            in_channels = out_channels
        self.features = nn.Sequential(*conv_layers)

        with torch.no_grad():
            sample = torch.zeros(1, channels, height, width)
            feature_dim = int(np.prod(self.features(sample).shape[1:]))

        self.head = build_mlp(feature_dim, n_actions, head_hidden_dims)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        return self.head(x)


class QLearningAgent(BaseAgent):
    def __init__(
        self,
        n_actions: int = 4,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        learning_rate: float = 1e-3,
        network_type: str = "mlp",
        hidden_dims: list[int] | None = None,
        conv_channels: list[int] | None = None,
        cnn_hidden_dims: list[int] | None = None,
        batch_size: int = 64,
        replay_buffer_size: int = 10000,
        min_replay_size: int = 1000,
        target_update_freq: int = 100,
        seed: int | None = None,
    ):
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.network_type = network_type
        self.hidden_dims = hidden_dims or [128, 128]
        self.conv_channels = conv_channels or [16, 32]
        self.cnn_hidden_dims = cnn_hidden_dims or [128]
        self.batch_size = batch_size
        self.replay_buffer_size = replay_buffer_size
        self.replay_buffer = deque(maxlen=replay_buffer_size)
        self.min_replay_size = min_replay_size
        self.target_update_freq = target_update_freq
        self.seed = seed

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.rng = np.random.default_rng(seed)
        self.training_steps = 0
        self.state_shape: tuple[int, ...] | None = None
        self.q_network: nn.Module | None = None
        self.target_network: nn.Module | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.loss_fn = nn.MSELoss()

    def _build_network(self, state: np.ndarray) -> nn.Module:
        state = np.asarray(state, dtype=np.float32)
        if self.network_type == "mlp":
            return build_mlp(int(state.size), self.n_actions, self.hidden_dims).to(self.device)
        return CNNQNetwork(
            input_shape=tuple(state.shape),
            n_actions=self.n_actions,
            conv_channels=self.conv_channels,
            head_hidden_dims=self.cnn_hidden_dims,
        ).to(self.device)

    def _ensure_initialized(self, state: np.ndarray) -> None:
        if self.q_network is not None:
            return
        self.state_shape = tuple(np.asarray(state).shape)
        self.q_network = self._build_network(state)
        self.target_network = self._build_network(state)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=self.learning_rate)

    def _format_state(self, state: np.ndarray) -> np.ndarray:
        state = np.asarray(state, dtype=np.float32)
        if self.network_type == "mlp":
            return state.reshape(-1)
        return np.transpose(state, (2, 0, 1))

    def _tensor(self, array: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(array, dtype=torch.float32, device=self.device)

    def choose_action(self, state: np.ndarray) -> int:
        self._ensure_initialized(state)
        if float(self.rng.random()) < self.epsilon:
            return int(self.rng.integers(0, self.n_actions))

        state_tensor = self._tensor(self._format_state(state)).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        return int(torch.argmax(q_values, dim=1).item())

    def update(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self._ensure_initialized(state)
        transition = (
            self._format_state(state),
            action,
            float(reward),
            self._format_state(next_state),
            bool(done),
        )
        self.replay_buffer.append(transition)

        if len(self.replay_buffer) < self.min_replay_size:
            if done:
                self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            return

        indices = self.rng.choice(len(self.replay_buffer), size=self.batch_size, replace=False)
        batch = [self.replay_buffer[index] for index in indices]

        states = self._tensor(np.stack([item[0] for item in batch]))
        actions = torch.as_tensor([item[1] for item in batch], dtype=torch.int64, device=self.device)
        rewards = self._tensor(np.array([item[2] for item in batch], dtype=np.float32))
        next_states = self._tensor(np.stack([item[3] for item in batch]))
        dones = self._tensor(np.array([item[4] for item in batch], dtype=np.float32))

        q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(dim=1).values
            targets = rewards + (1.0 - dones) * self.gamma * next_q_values

        loss = self.loss_fn(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.training_steps += 1
        if self.training_steps % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        if done:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path: str) -> None:
        payload = {
            "agent_type": "q_learning",
            "n_actions": self.n_actions,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "epsilon_min": self.epsilon_min,
            "epsilon_decay": self.epsilon_decay,
            "learning_rate": self.learning_rate,
            "network_type": self.network_type,
            "hidden_dims": self.hidden_dims,
            "conv_channels": self.conv_channels,
            "cnn_hidden_dims": self.cnn_hidden_dims,
            "batch_size": self.batch_size,
            "replay_buffer_size": self.replay_buffer_size,
            "min_replay_size": self.min_replay_size,
            "target_update_freq": self.target_update_freq,
            "seed": self.seed,
            "training_steps": self.training_steps,
            "state_shape": self.state_shape,
            "q_network_state_dict": None if self.q_network is None else self.q_network.state_dict(),
            "target_network_state_dict": None
            if self.target_network is None
            else self.target_network.state_dict(),
        }
        self._write_payload(path, payload)

    @classmethod
    def from_payload(cls, payload: dict) -> "QLearningAgent":
        agent = cls(
            n_actions=payload["n_actions"],
            gamma=payload["gamma"],
            epsilon=payload["epsilon"],
            epsilon_min=payload["epsilon_min"],
            epsilon_decay=payload["epsilon_decay"],
            learning_rate=payload["learning_rate"],
            network_type=payload.get("network_type", "mlp"),
            hidden_dims=list(payload.get("hidden_dims", [128, 128])),
            conv_channels=list(payload.get("conv_channels", [16, 32])),
            cnn_hidden_dims=list(payload.get("cnn_hidden_dims", [128])),
            batch_size=payload["batch_size"],
            replay_buffer_size=payload.get("replay_buffer_size", 10000),
            min_replay_size=payload["min_replay_size"],
            target_update_freq=payload["target_update_freq"],
            seed=payload.get("seed"),
        )
        agent.training_steps = payload.get("training_steps", 0)

        state_shape = payload.get("state_shape")
        if state_shape is not None:
            dummy_state = np.zeros(state_shape, dtype=np.float32)
            agent._ensure_initialized(dummy_state)
            agent.q_network.load_state_dict(payload["q_network_state_dict"])
            agent.target_network.load_state_dict(payload["target_network_state_dict"])

        return agent
