from collections import deque


class BaseReward:
    """Base class for reward functions in the Snake RL environment."""

    def reset(self, info: dict | None = None):
        pass

    def compute(self, state, action, next_state, raw_reward, terminated, truncated, info):
        raise NotImplementedError


class Identity(BaseReward):
    def compute(self, state, action, next_state, raw_reward, terminated, truncated, info):
        return raw_reward


class SparseReward(BaseReward):
    def __init__(self, food_reward: float = 1.0, death_penalty: float = -1.0):
        self.food_reward = food_reward
        self.death_penalty = death_penalty
        self.previous_food = None

    def reset(self, info: dict | None = None):
        self.previous_food = None if info is None else info.get("food")

    def compute(self, state, action, next_state, raw_reward, terminated, truncated, info):
        total_reward = 0.0
        head = info.get("head")

        if self.previous_food is not None and head == self.previous_food:
            total_reward += self.food_reward

        if terminated:
            total_reward += self.death_penalty

        self.previous_food = info.get("food")
        return total_reward


class DenseReward(BaseReward):
    def __init__(
        self,
        food_reward: float = 10.0,
        death_penalty: float = -10.0,
        distance_alpha: float = 1.0,
        loop_penalty: float = -8.0,
        timeout_penalty: float = -5.0,
        loop_window: int = 20,
        loop_threshold: int = 4,
    ):
        self.food_reward = food_reward
        self.death_penalty = death_penalty
        self.distance_alpha = distance_alpha
        self.loop_penalty = loop_penalty
        self.timeout_penalty = timeout_penalty
        self.loop_threshold = loop_threshold
        self.position_history = deque(maxlen=loop_window)
        self.previous_food = None
        self.previous_distance = None

    def reset(self, info: dict | None = None):
        self.position_history.clear()
        self.previous_food = None if info is None else info.get("food")

        head = None if info is None else info.get("head")
        food = None if info is None else info.get("food")
        if head is not None:
            self.position_history.append(head)
        self.previous_distance = self._manhattan_distance(head, food)

    @staticmethod
    def _manhattan_distance(head, food):
        if head is None or food is None:
            return None
        return abs(head[0] - food[0]) + abs(head[1] - food[1])

    def compute(self, state, action, next_state, raw_reward, terminated, truncated, info):
        total_reward = 0.0
        head = info.get("head")
        food = info.get("food")

        food_eaten = self.previous_food is not None and head == self.previous_food
        if food_eaten:
            total_reward += self.food_reward

        if terminated:
            total_reward += self.death_penalty

        if truncated:
            total_reward += self.timeout_penalty

        if head is not None:
            self.position_history.append(head)
            if self.position_history.count(head) >= self.loop_threshold:
                total_reward += self.loop_penalty

        current_distance = self._manhattan_distance(head, food)
        if self.previous_distance is not None and current_distance is not None and not food_eaten:
            delta_distance = self.previous_distance - current_distance
            total_reward += delta_distance * self.distance_alpha

        self.previous_food = food
        self.previous_distance = current_distance
        return total_reward
