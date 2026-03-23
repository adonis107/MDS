import gymnasium as gym
import numpy as np


class BaseStateEncoder:
    def __init__(self):
        self.observation_space = None

    def encode(self, obs: np.ndarray, info: dict):
        raise NotImplementedError


class FullGridEncoder(BaseStateEncoder):
    def __init__(self, width: int = 20, height: int = 20, flatten: bool = True):
        super().__init__()
        self.flatten = flatten
        shape = (height * width * 3,) if flatten else (height, width, 3)
        self.observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=shape,
            dtype=np.float32,
        )

    def encode(self, obs, info):
        obs = obs.astype(np.float32)
        if self.flatten:
            return obs.flatten()
        return obs


class EgocentricEncoder(BaseStateEncoder):
    def __init__(self, window_radius: int = 3, flatten: bool = True):
        super().__init__()
        self.window_radius = window_radius
        self.flatten = flatten
        window_size = self.window_radius * 2 + 1
        shape = (window_size * window_size * 3,) if flatten else (window_size, window_size, 3)
        self.observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=shape,
            dtype=np.float32,
        )

    def encode(self, obs, info):
        head = info.get("head", (obs.shape[0] // 2, obs.shape[1] // 2))
        row = int(np.clip(head[0], 0, obs.shape[0] - 1))
        col = int(np.clip(head[1], 0, obs.shape[1] - 1))

        pad_width = (
            (self.window_radius, self.window_radius),
            (self.window_radius, self.window_radius),
            (0, 0),
        )
        padded_obs = np.pad(obs, pad_width, mode="constant", constant_values=0)
        window_size = self.window_radius * 2 + 1
        row += self.window_radius
        col += self.window_radius
        cropped_obs = padded_obs[
            row - self.window_radius : row + self.window_radius + 1,
            col - self.window_radius : col + self.window_radius + 1,
            :,
        ].astype(np.float32)
        if self.flatten:
            return cropped_obs.flatten()
        return cropped_obs


class FeatureVectorEncoder(BaseStateEncoder):
    def __init__(self):
        super().__init__()
        self.observation_space = gym.spaces.Box(
            low=-1,
            high=1,
            shape=(10,),
            dtype=np.float32,
        )

    @staticmethod
    def _is_body(obs: np.ndarray, row: int, col: int) -> bool:
        height, width = obs.shape[:2]
        if row < 0 or row >= height or col < 0 or col >= width:
            return True
        return bool(obs[row, col, 1] > 0.5 and obs[row, col, 0] < 0.5)

    @staticmethod
    def _infer_direction(obs: np.ndarray, head: tuple[int, int]) -> np.ndarray:
        row, col = head
        neighbors = {
            "up": (row - 1, col),
            "down": (row + 1, col),
            "left": (row, col - 1),
            "right": (row, col + 1),
        }

        if FeatureVectorEncoder._is_body(obs, *neighbors["down"]):
            return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        if FeatureVectorEncoder._is_body(obs, *neighbors["up"]):
            return np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
        if FeatureVectorEncoder._is_body(obs, *neighbors["right"]):
            return np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)
        if FeatureVectorEncoder._is_body(obs, *neighbors["left"]):
            return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        return np.zeros(4, dtype=np.float32)

    def encode(self, obs, info):
        height, width = obs.shape[:2]
        default_head = (height // 2, width // 2)
        default_food = (0, 0)

        head = info.get("head", default_head)
        food = info.get("food", default_food)
        head_row, head_col = int(head[0]), int(head[1])
        food_row, food_col = int(food[0]), int(food[1])

        danger_up = float(self._is_body(obs, head_row - 1, head_col))
        danger_down = float(self._is_body(obs, head_row + 1, head_col))
        danger_left = float(self._is_body(obs, head_row, head_col - 1))
        danger_right = float(self._is_body(obs, head_row, head_col + 1))

        food_vertical = float(np.clip(food_row - head_row, -(height - 1), height - 1)) / max(height - 1, 1)
        food_horizontal = float(np.clip(food_col - head_col, -(width - 1), width - 1)) / max(width - 1, 1)

        direction_one_hot = self._infer_direction(obs, (head_row, head_col))

        return np.array(
            [
                danger_up,
                danger_down,
                danger_left,
                danger_right,
                food_vertical,
                food_horizontal,
                direction_one_hot[0],
                direction_one_hot[1],
                direction_one_hot[2],
                direction_one_hot[3],
            ],
            dtype=np.float32,
        )
