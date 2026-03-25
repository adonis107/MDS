from __future__ import annotations

from collections.abc import Sequence

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.axes import Axes
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure

# Indices match grid encoding: 0=empty 1=body 2=head 3=gold 4=silver 5=poison 6=obstacle
OBSERVATION_COLORS: Sequence[str] = (
    "#dfe6e9",  # 0  empty       — light grey-blue
    "#00b894",  # 1  snake body  — teal green
    "#00695c",  # 2  snake head  — dark teal
    "#fdcb6e",  # 3  gold food   — warm amber
    "#74b9ff",  # 4  silver food — sky blue
    "#fd79a8",  # 5  poison food — hot pink
    "#2d3436",  # 6  obstacle    — near-black charcoal
)

_LEGEND_ENTRIES = [
    ("#00695c", "head"),
    ("#00b894", "body"),
    ("#fdcb6e", "gold food"),
    ("#74b9ff", "silver food"),
    ("#fd79a8", "poison food"),
    ("#2d3436", "obstacle / wall"),
]


def _add_border(obs: np.ndarray) -> np.ndarray:
    """Wrap a grid observation in a 1-cell obstacle border (value 6)."""
    h, w = obs.shape
    bordered = np.full((h + 2, w + 2), 6, dtype=obs.dtype)
    bordered[1:-1, 1:-1] = obs
    return bordered


def _draw_legend(ax: Axes) -> None:
    patches = [mpatches.Patch(facecolor=color, edgecolor="#636e72", label=label) for color, label in _LEGEND_ENTRIES]
    ax.legend(handles=patches, loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0, framealpha=0.9)


def render_observation(
    observation: np.ndarray,
    ax: Axes | None = None,
    title: str | None = None,
    legend: bool = True,
) -> tuple[Figure, Axes]:
    """Render a single grid observation with Matplotlib."""
    observation = np.asarray(observation)
    if observation.ndim != 2:
        raise ValueError("Observation must be a 2D array.")

    bordered = _add_border(observation)

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 6))

    cmap = ListedColormap(OBSERVATION_COLORS)
    ax.imshow(bordered, cmap=cmap, vmin=0, vmax=len(OBSERVATION_COLORS) - 1)

    height, width = bordered.shape
    ax.set_xticks(np.arange(-0.5, width, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, height, 1), minor=True)
    ax.grid(which="minor", color="#b2bec3", linewidth=0.8)
    ax.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)

    if title is not None:
        ax.set_title(title)

    ax.set_aspect("equal")

    if legend:
        _draw_legend(ax)

    return ax.figure, ax


def make_animation(
    env,
    agent,
    get_state,
    n_frames: int = 1,
    max_frames: int | None = None,
    interval: int = 150,
    title: str = "Snake",
    legend: bool = True,
) -> FuncAnimation:
    """Generate a Matplotlib animation for one rollout.

    Args:
        n_frames: Number of frames to stack for the agent input (must match
                  how the agent was trained). The rendered video always shows
                  the raw single-frame grid regardless of this value.
    """
    from rl_snake.agent import FrameStack

    frame_stack = FrameStack(n_frames, get_state)

    first_obs = env.reset()
    observations = [first_obs]
    state = frame_stack.reset(env)
    done = False

    while not done and (max_frames is None or len(observations) < max_frames):
        action = agent.select_action(state)
        result = env.step(action)
        observations.append(result.observation)
        state = frame_stack.step(env)
        done = result.done

    bordered_frames = [_add_border(obs) for obs in observations]

    fig, ax = plt.subplots(figsize=(7, 6))
    cmap = ListedColormap(OBSERVATION_COLORS)
    image = ax.imshow(bordered_frames[0], cmap=cmap, vmin=0, vmax=len(OBSERVATION_COLORS) - 1)

    height, width = bordered_frames[0].shape
    ax.set_xticks(np.arange(-0.5, width, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, height, 1), minor=True)
    ax.grid(which="minor", color="#b2bec3", linewidth=0.8)
    ax.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)
    ax.set_aspect("equal")

    if legend:
        _draw_legend(ax)

    def update(frame_idx: int):
        image.set_data(bordered_frames[frame_idx])
        ax.set_title(f"{title} | step {frame_idx}")
        return (image,)

    animation = FuncAnimation(
        fig,
        update,
        frames=len(bordered_frames),
        interval=interval,
        blit=True,
        repeat=False,
    )
    return animation


def save_video(
    path: str,
    env,
    agent,
    get_state,
    n_frames: int = 1,
    fps: int = 8,
    max_frames: int = 400,
    title: str = "Snake",
) -> None:
    """Save a rollout video to disk.

    Supports .gif (Pillow) and .mp4 (ffmpeg). Falls back to GIF if the path
    extension is not recognised.

    Args:
        path:       Output file path (.gif or .mp4).
        env:        A SnakeEnv instance (will be reset internally).
        agent:      A trained agent with a select_action(state) method.
        get_state:  Base state-extraction function (get_state or get_grid_state).
        n_frames:   Frame-stack depth used during training.
        fps:        Frames per second.
        max_frames: Maximum number of game steps to record.
        title:      Title shown on the animation.
    """
    anim = make_animation(
        env,
        agent,
        get_state,
        n_frames=n_frames,
        max_frames=max_frames,
        interval=1000 // fps,
        title=title,
        legend=True,
    )
    try:
        if path.endswith(".mp4"):
            anim.save(path, writer="ffmpeg", fps=fps)
        else:
            anim.save(path, writer="pillow", fps=fps)
    finally:
        plt.close("all")
