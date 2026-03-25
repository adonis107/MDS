from __future__ import annotations

import copy
import random
from abc import ABC, abstractmethod
from collections import deque

import numpy as np
import torch
from torch import nn, optim

from rl_snake.env import DOWN, LEFT, RIGHT, UP, SnakeEnv

# ---------------------------------------------------------------------------
# State extraction
# ---------------------------------------------------------------------------

_TURN_RIGHT = {UP: RIGHT, RIGHT: DOWN, DOWN: LEFT, LEFT: UP}
_TURN_LEFT = {UP: LEFT, RIGHT: UP, DOWN: RIGHT, LEFT: DOWN}
_DELTA = {UP: (-1, 0), RIGHT: (0, 1), DOWN: (1, 0), LEFT: (0, -1)}


def _is_dangerous(env: SnakeEnv, pos: tuple[int, int]) -> bool:
    r, c = pos
    if r < 0 or r >= env.height or c < 0 or c >= env.width:
        return True
    if pos in env.all_obstacle_positions:
        return True
    # tail will move away this step, so exclude it
    return pos in env.snake[:-1]


def get_state(env: SnakeEnv) -> np.ndarray:
    """Return the 11-feature state vector.

    Features (all binary floats):
        [0]  danger straight ahead
        [1]  danger to the right (relative turn)
        [2]  danger to the left  (relative turn)
        [3]  moving up
        [4]  moving right
        [5]  moving down
        [6]  moving left
        [7]  food is above head
        [8]  food is below head
        [9]  food is left  of head
        [10] food is right of head
    """
    hr, hc = env.snake[0]
    d = env.direction

    def ahead(direction: int) -> tuple[int, int]:
        dr, dc = _DELTA[direction]
        return (hr + dr, hc + dc)

    danger_straight = float(_is_dangerous(env, ahead(d)))
    danger_right = float(_is_dangerous(env, ahead(_TURN_RIGHT[d])))
    danger_left = float(_is_dangerous(env, ahead(_TURN_LEFT[d])))

    fr, fc = env.food or (hr, hc)

    return np.array(
        [
            danger_straight,
            danger_right,
            danger_left,
            float(d == UP),
            float(d == RIGHT),
            float(d == DOWN),
            float(d == LEFT),
            float(fr < hr),
            float(fr > hr),
            float(fc < hc),
            float(fc > hc),
        ],
        dtype=np.float32,
    )


def get_grid_state(env: SnakeEnv) -> np.ndarray:
    """Return the full grid as a 6-channel binary float array (C, H, W).

    Channels:
        0 = snake body
        1 = snake head
        2 = gold food
        3 = silver food
        4 = poison food
        5 = obstacle (static or dynamic)
    """
    obs = env._get_observation()  # (H, W) with values 0-6
    grid = np.zeros((6, env.height, env.width), dtype=np.float32)
    grid[0] = obs == 1  # body
    grid[1] = obs == 2  # head
    grid[2] = obs == 3  # gold food
    grid[3] = obs == 4  # silver food
    grid[4] = obs == 5  # poison food
    grid[5] = obs == 6  # obstacle
    return grid


# ---------------------------------------------------------------------------
# Replay Buffer
# ---------------------------------------------------------------------------


class ReplayBuffer:
    def __init__(self, capacity: int = 100_000) -> None:
        self._buf: deque = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self._buf.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self._buf, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch, strict=False)
        return (
            np.stack(states),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.stack(next_states),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self._buf)


# ---------------------------------------------------------------------------
# Base Agent
# ---------------------------------------------------------------------------


class BaseAgent(ABC):
    @abstractmethod
    def select_action(self, state: np.ndarray) -> int: ...

    @abstractmethod
    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None: ...

    @abstractmethod
    def learn(self) -> float | None: ...

    def save(self, path: str) -> None:
        raise NotImplementedError

    def load(self, path: str) -> None:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Networks
# ---------------------------------------------------------------------------


class _QNetwork(nn.Module):
    def __init__(self, state_dim: int, n_actions: int, hidden: tuple[int, ...] = (256, 128)) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = state_dim
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        layers.append(nn.Linear(in_dim, n_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _CNNQNetwork(nn.Module):
    """CNN Q-network that operates on the full grid observation (4-channel binary image)."""

    def __init__(
        self,
        height: int,
        width: int,
        n_actions: int,
        conv_channels: tuple[int, ...] = (32, 64),
        hidden: tuple[int, ...] = (512,),
    ) -> None:
        super().__init__()
        conv_layers: list[nn.Module] = []
        in_ch = 6  # body, head, gold, silver, poison, obstacle
        for ch in conv_channels:
            conv_layers += [nn.Conv2d(in_ch, ch, kernel_size=3, padding=1), nn.ReLU()]
            in_ch = ch
        self.conv = nn.Sequential(*conv_layers)

        fc_in = in_ch * height * width
        fc_layers: list[nn.Module] = []
        in_dim = fc_in
        for h in hidden:
            fc_layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        fc_layers.append(nn.Linear(in_dim, n_actions))
        self.fc = nn.Sequential(*fc_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = x.flatten(1)
        return self.fc(x)


# ---------------------------------------------------------------------------
# Shared DQN logic
# ---------------------------------------------------------------------------

_OPTIMIZERS = {
    "adam": optim.Adam,
    "rmsprop": optim.RMSprop,
}


class _BaseDQNAgent(BaseAgent):
    """Shared DQN training logic. Subclasses provide the Q-network architecture."""

    N_ACTIONS = 4

    def __init__(
        self,
        q_net: nn.Module,
        lr: float,
        gamma: float,
        epsilon_start: float,
        epsilon_end: float,
        epsilon_decay: float,
        batch_size: int,
        buffer_capacity: int,
        target_update_freq: int,
        optimizer_name: str,
        device: torch.device,
    ) -> None:
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.device = device

        self.q_net = q_net
        self.target_net = copy.deepcopy(q_net)
        self.target_net.eval()

        optimizer_cls = _OPTIMIZERS.get(optimizer_name)
        if optimizer_cls is None:
            msg = f"Unknown optimizer '{optimizer_name}'. Choose from: {list(_OPTIMIZERS)}"
            raise ValueError(msg)
        self.optimizer = optimizer_cls(self.q_net.parameters(), lr=lr)

        self.buffer = ReplayBuffer(buffer_capacity)
        self._steps = 0

    def select_action(self, state: np.ndarray) -> int:
        if random.random() < self.epsilon:
            return random.randrange(self.N_ACTIONS)
        with torch.no_grad():
            t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            return int(self.q_net(t).argmax(dim=1).item())

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.buffer.push(state, action, reward, next_state, done)

    def learn(self) -> float | None:
        if len(self.buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        s = torch.tensor(states, device=self.device)
        a = torch.tensor(actions, device=self.device)
        r = torch.tensor(rewards, device=self.device)
        s_ = torch.tensor(next_states, device=self.device)
        d = torch.tensor(dones, device=self.device)

        q_values = self.q_net(s).gather(1, a.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            max_next_q = self.target_net(s_).max(dim=1).values
            targets = r + self.gamma * max_next_q * (1.0 - d)

        loss = nn.functional.mse_loss(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        self._steps += 1
        if self._steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        return loss.item()

    def save(self, path: str) -> None:
        torch.save(
            {
                "q_net": self.q_net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
                "steps": self._steps,
            },
            path,
        )

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.q_net.load_state_dict(ckpt["q_net"])
        self.target_net.load_state_dict(ckpt["target_net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.epsilon = ckpt["epsilon"]
        self._steps = ckpt["steps"]


# ---------------------------------------------------------------------------
# Concrete agents
# ---------------------------------------------------------------------------


class DQNAgent(_BaseDQNAgent):
    """DQN agent with a fully-connected MLP operating on the 11-feature state vector."""

    STATE_DIM = 11

    def __init__(
        self,
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        batch_size: int = 64,
        buffer_capacity: int = 100_000,
        target_update_freq: int = 1_000,
        hidden: tuple[int, ...] = (256, 128),
        optimizer_name: str = "adam",
        device: str | None = None,
    ) -> None:
        dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        q_net = _QNetwork(self.STATE_DIM, self.N_ACTIONS, hidden).to(dev)
        super().__init__(
            q_net=q_net,
            lr=lr,
            gamma=gamma,
            epsilon_start=epsilon_start,
            epsilon_end=epsilon_end,
            epsilon_decay=epsilon_decay,
            batch_size=batch_size,
            buffer_capacity=buffer_capacity,
            target_update_freq=target_update_freq,
            optimizer_name=optimizer_name,
            device=dev,
        )


class CNNDQNAgent(_BaseDQNAgent):
    """DQN agent with a CNN operating on the full 4-channel grid observation."""

    def __init__(
        self,
        height: int,
        width: int,
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        batch_size: int = 64,
        buffer_capacity: int = 100_000,
        target_update_freq: int = 1_000,
        conv_channels: tuple[int, ...] = (32, 64),
        hidden: tuple[int, ...] = (512,),
        optimizer_name: str = "adam",
        device: str | None = None,
    ) -> None:
        dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        q_net = _CNNQNetwork(height, width, self.N_ACTIONS, conv_channels, hidden).to(dev)
        super().__init__(
            q_net=q_net,
            lr=lr,
            gamma=gamma,
            epsilon_start=epsilon_start,
            epsilon_end=epsilon_end,
            epsilon_decay=epsilon_decay,
            batch_size=batch_size,
            buffer_capacity=buffer_capacity,
            target_update_freq=target_update_freq,
            optimizer_name=optimizer_name,
            device=dev,
        )
