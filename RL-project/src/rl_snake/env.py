from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Actions
UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

# Grid encoding
# 0 = empty
# 1 = snake body
# 2 = snake head
# 3 = gold food  (high reward)
# 4 = silver food (medium reward)
# 5 = poison food (shrinks snake)
# 6 = obstacle   (static or dynamic)

_FOOD_GRID_VALUES = {"gold": 3, "silver": 4, "poison": 5}
_OPPOSITE = {UP: DOWN, DOWN: UP, LEFT: RIGHT, RIGHT: LEFT}


@dataclass
class StepResult:
    observation: np.ndarray
    reward: float
    done: bool
    info: dict


class SnakeEnv:
    """Snake environment with multiple food types and dynamic obstacles.

    Grid encoding in observation:
        0 = empty
        1 = snake body
        2 = snake head
        3 = gold food   (gold_reward on eat)
        4 = silver food (silver_reward on eat)
        5 = poison food (poison_reward + shrinks snake by poison_shrink)
        6 = obstacle    (static or dynamic — both lethal)

    Actions:
        0 = up, 1 = right, 2 = down, 3 = left

    Reward structure (additive):
        gold_reward           -- eating gold food
        silver_reward         -- eating silver food
        poison_reward         -- eating poison (typically 0 or negative)
        death_reward          -- collision with wall / self / obstacle
        step_reward               -- every step
        distance_reward_scale     -- potential-based shaping toward nearest valuable food
        body_proximity_reward_scale -- potential-based shaping away from own body (> 0 = reward distance)
    """

    ACTION_TO_DELTA = {UP: (-1, 0), RIGHT: (0, 1), DOWN: (1, 0), LEFT: (0, -1)}

    def __init__(
        self,
        height: int = 10,
        width: int = 10,
        init_length: int = 3,
        seed: int | None = None,
        # Rewards
        food_reward: float | None = None,  # backward-compat alias for gold_reward
        gold_reward: float = 1.0,
        silver_reward: float = 0.5,
        poison_reward: float = 0.0,
        death_reward: float = -1.0,
        step_reward: float = 0.0,
        distance_reward_scale: float = 0.0,
        body_proximity_reward_scale: float = 0.0,
        # Food counts
        n_gold: int = 1,
        n_silver: int = 0,
        n_poison: int = 0,
        poison_shrink: int = 2,  # body segments removed when poison is eaten
        # Steps
        max_steps: int | None = None,
        allow_reverse: bool = False,
        # Obstacles
        obstacles: list[tuple[int, int]] | None = None,
        n_dynamic_obstacles: int = 0,
        n_rand_obstacles: int = 0,
    ) -> None:
        self.height = height
        self.width = width
        self.init_length = init_length

        # food_reward is a backward-compat alias for gold_reward
        self.gold_reward = food_reward if food_reward is not None else gold_reward
        self.silver_reward = silver_reward
        self.poison_reward = poison_reward
        self.death_reward = death_reward
        self.step_reward = step_reward
        self.distance_reward_scale = distance_reward_scale
        self.body_proximity_reward_scale = body_proximity_reward_scale

        self.n_gold = n_gold
        self.n_silver = n_silver
        self.n_poison = n_poison
        self.poison_shrink = poison_shrink

        self.max_steps = max_steps
        self.allow_reverse = allow_reverse

        self.rng = np.random.default_rng(seed)

        self.obstacles: set[tuple[int, int]] = set(obstacles or [])
        self.n_dynamic_obstacles = n_dynamic_obstacles
        self.n_rand_obstacles = n_rand_obstacles
        self._rand_obstacles: set[tuple[int, int]] = set()

        self.snake: list[tuple[int, int]] = []
        self.direction = RIGHT
        # maps position -> food type ('gold', 'silver', 'poison')
        self.foods: dict[tuple[int, int], str] = {}
        self.done = False
        self.steps = 0

        # Dynamic obstacles: each entry is [row, col, direction]
        self.dynamic_obstacles: list[list] = []

        self.reset()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def length(self) -> int:
        """Current snake length."""
        return len(self.snake)

    @property
    def food(self) -> tuple[int, int] | None:
        """Backward-compat: position of the first gold food, or any food."""
        for pos, ftype in self.foods.items():
            if ftype == "gold":
                return pos
        return next(iter(self.foods), None)

    @property
    def dynamic_obstacle_positions(self) -> set[tuple[int, int]]:
        positions: set[tuple[int, int]] = set()
        for r, c, _d, ori, _phase in self.dynamic_obstacles:
            if ori == "v":
                positions.update((r + i, c) for i in range(3))
            else:
                positions.update((r, c + i) for i in range(3))
        return positions

    @property
    def all_obstacle_positions(self) -> set[tuple[int, int]]:
        return self.obstacles | self._rand_obstacles | self.dynamic_obstacle_positions

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> np.ndarray:
        self.done = False
        self.steps = 0
        self.direction = RIGHT

        center_row = self.height // 2
        center_col = self.width // 2
        self.snake = [(center_row, center_col - i) for i in range(self.init_length)]

        if any(pos in self.obstacles for pos in self.snake):
            raise ValueError("Initial snake overlaps a static obstacle.")

        self._rand_obstacles = set()
        self._spawn_rand_obstacles()

        self.dynamic_obstacles = []
        self._spawn_dynamic_obstacles()

        self.foods = {}
        for _ in range(self.n_gold):
            self._spawn_food("gold")
        for _ in range(self.n_silver):
            self._spawn_food("silver")
        for _ in range(self.n_poison):
            self._spawn_food("poison")

        return self._get_observation()

    def step(self, action: int) -> StepResult:
        if self.done:
            return StepResult(
                self._get_observation(), 0.0, True, {"length": self.length, "steps": self.steps}
            )

        self.steps += 1

        action = int(action)
        if action not in self.ACTION_TO_DELTA:
            raise ValueError(f"Invalid action: {action}")

        action = self._sanitize_action(action)
        self.direction = action

        head_row, head_col = self.snake[0]
        d_row, d_col = self.ACTION_TO_DELTA[action]
        new_head = (head_row + d_row, head_col + d_col)

        food_type = self.foods.get(new_head)
        ate_food = food_type is not None
        ate_poison = food_type == "poison"

        # Capture distances before move for potential shaping
        dist_before = self._manhattan_to_nearest_food(head_row, head_col)
        if self.body_proximity_reward_scale != 0.0 and len(self.snake) > 1:
            body_dist_before = self._manhattan_to_nearest_body(head_row, head_col)
            body_dist_after = self._manhattan_to_nearest_body(*new_head)
        else:
            body_dist_before = body_dist_after = 0.0

        # For collision: tail moves away unless the snake is growing (gold/silver)
        grows = ate_food and not ate_poison
        if self._is_collision(new_head, grow=grows):
            self.done = True
            return StepResult(
                self._get_observation(), self.death_reward, True, {"length": self.length, "steps": self.steps}
            )

        self.snake.insert(0, new_head)
        reward = self.step_reward

        if ate_food:
            del self.foods[new_head]
            if ate_poison:
                reward += self.poison_reward
                # Shrink — keep at least 1 segment
                shrink = min(self.poison_shrink, len(self.snake) - 1)
                for _ in range(shrink):
                    self.snake.pop()
                self._spawn_food("poison")
            else:
                reward += self.gold_reward if food_type == "gold" else self.silver_reward
                self._spawn_food(food_type)
        else:
            self.snake.pop()
            if self.distance_reward_scale != 0.0 and self.foods:
                dist_after = self._manhattan_to_nearest_food(*new_head)
                reward += self.distance_reward_scale * (dist_before - dist_after)

        if self.body_proximity_reward_scale != 0.0 and body_dist_before != 0.0:
            reward += self.body_proximity_reward_scale * (body_dist_after - body_dist_before)

        # Dynamic obstacles move after the snake acts
        self._move_dynamic_obstacles()

        if self.max_steps is not None and self.steps >= self.max_steps:
            self.done = True
            return StepResult(
                self._get_observation(),
                reward,
                True,
                {"length": self.length, "steps": self.steps, "termination_reason": "max_steps"},
            )

        return StepResult(
            self._get_observation(), reward, False, {"length": self.length, "steps": self.steps}
        )

    def render(self) -> None:
        obs = self._get_observation()
        symbols = {0: ".", 1: "o", 2: "H", 3: "G", 4: "S", 5: "P", 6: "#"}
        for row in obs:
            print(" ".join(symbols[int(cell)] for cell in row))
        print()

    def sample_action(self) -> int:
        return int(self.rng.integers(0, 4))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _sanitize_action(self, action: int) -> int:
        if self.allow_reverse:
            return action
        if action == _OPPOSITE[self.direction]:
            return self.direction
        return action

    def _is_collision(self, position: tuple[int, int], grow: bool) -> bool:
        row, col = position
        if row < 0 or row >= self.height or col < 0 or col >= self.width:
            return True
        if position in self.all_obstacle_positions:
            return True
        body_to_check = self.snake if grow else self.snake[:-1]
        return position in body_to_check

    def _manhattan_to_nearest_body(self, row: int, col: int) -> float:
        """Manhattan distance from (row, col) to the nearest body segment (excludes head)."""
        best = float("inf")
        for br, bc in self.snake[1:]:
            d = abs(br - row) + abs(bc - col)
            if d < best:
                best = d
        return best

    def _manhattan_to_nearest_food(
        self, row: int, col: int, food_types: tuple[str, ...] = ("gold", "silver")
    ) -> float:
        """Distance to nearest food of the given types (ignores poison by default)."""
        best = float("inf")
        for pos, ftype in self.foods.items():
            if ftype in food_types:
                d = abs(pos[0] - row) + abs(pos[1] - col)
                if d < best:
                    best = d
        return best if best != float("inf") else 0.0

    def _dynamic_obstacle_swept_cells(self) -> set[tuple[int, int]]:
        """All cells in rows/columns where dynamic obstacles travel (forbidden for food)."""
        blocked: set[tuple[int, int]] = set()
        for r, c, _d, ori, _phase in self.dynamic_obstacles:
            if ori == "h":
                blocked.update((r, col) for col in range(self.width))
            else:
                blocked.update((row, c) for row in range(self.height))
        return blocked

    def _spawn_food(self, food_type: str) -> None:
        occupied = set(self.snake) | self.all_obstacle_positions | set(self.foods) | self._dynamic_obstacle_swept_cells()
        free = [(r, c) for r in range(self.height) for c in range(self.width) if (r, c) not in occupied]
        if not free:
            if not self.foods:
                self.done = True
            return
        idx = int(self.rng.integers(len(free)))
        self.foods[free[idx]] = food_type

    def _spawn_rand_obstacles(self) -> None:
        """Randomly place n_rand_obstacles single-cell obstacles each episode."""
        occupied = set(self.snake) | self.obstacles | set(self.foods)
        free = [(r, c) for r in range(self.height) for c in range(self.width) if (r, c) not in occupied]
        indices = self.rng.permutation(len(free))
        self._rand_obstacles = {free[int(i)] for i in indices[: self.n_rand_obstacles]}

    def _spawn_dynamic_obstacles(self) -> None:
        """Spawn wall segments of 3 cells. Vertical walls move UP/DOWN, horizontal LEFT/RIGHT."""
        occupied = set(self.snake) | self.obstacles | self._rand_obstacles | set(self.foods)

        for _ in range(self.n_dynamic_obstacles):
            ori = "v" if self.rng.integers(2) == 0 else "h"

            if ori == "v":
                direction = UP if self.rng.integers(2) == 0 else DOWN
                # Ensure the oscillated position (1 step in direction) also stays in bounds.
                # UP shift: row-1 >= 0  →  r >= 1
                # DOWN shift: row+1+2 <= height-1  →  r <= height-4
                r_min = 1 if direction == UP else 0
                r_max = self.height - 3 if direction == UP else self.height - 4
                valid = [
                    (r, c)
                    for r in range(r_min, r_max + 1)
                    for c in range(self.width)
                    if (r, c) not in occupied and (r + 1, c) not in occupied and (r + 2, c) not in occupied
                ]
            else:
                direction = LEFT if self.rng.integers(2) == 0 else RIGHT
                # Ensure the oscillated position (1 step in direction) also stays in bounds.
                # LEFT shift: col-1 >= 0  →  c >= 1
                # RIGHT shift: col+1+2 <= width-1  →  c <= width-4
                c_min = 1 if direction == LEFT else 0
                c_max = self.width - 3 if direction == LEFT else self.width - 4
                valid = [
                    (r, c)
                    for r in range(self.height)
                    for c in range(c_min, c_max + 1)
                    if (r, c) not in occupied and (r, c + 1) not in occupied and (r, c + 2) not in occupied
                ]

            if not valid:
                break

            idx = int(self.rng.integers(len(valid)))
            r, c = valid[idx]
            self.dynamic_obstacles.append([r, c, direction, ori, 0])

            if ori == "v":
                occupied.update((r + i, c) for i in range(3))
            else:
                occupied.update((r, c + i) for i in range(3))

    def _move_dynamic_obstacles(self) -> None:
        """Move each wall obstacle 1 step in its direction, then back — pure oscillation."""
        for obs in self.dynamic_obstacles:
            r, c, d, ori, phase = obs
            # phase 0 → move in direction d; phase 1 → move back
            move_dir = d if phase == 0 else _OPPOSITE[d]
            dr, dc = self.ACTION_TO_DELTA[move_dir]
            obs[0] += dr
            obs[1] += dc
            obs[4] = 1 - phase

    def _get_observation(self) -> np.ndarray:
        grid = np.zeros((self.height, self.width), dtype=np.int8)

        for r, c in self.obstacles:
            grid[r, c] = 6
        for r, c in self._rand_obstacles:
            grid[r, c] = 6
        for r, c, _d, ori, _phase in self.dynamic_obstacles:
            if ori == "v":
                for i in range(3):
                    grid[r + i, c] = 6
            else:
                for i in range(3):
                    grid[r, c + i] = 6

        for r, c in self.snake[1:]:
            grid[r, c] = 1
        if self.snake:
            grid[self.snake[0][0], self.snake[0][1]] = 2

        for (r, c), ftype in self.foods.items():
            grid[r, c] = _FOOD_GRID_VALUES[ftype]

        return grid.copy()
