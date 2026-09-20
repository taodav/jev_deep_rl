"""Convert observations to named facts, and enumerate legal discrete actions."""

from dataclasses import dataclass
from typing import Any, Protocol

import gymnasium as gym
import numpy as np

from .descriptions import COORDINATES, describe_breakout, describe_pong

DEFAULT_ATARI_FRAMESKIP = 2


@dataclass(frozen=True)
class Action:
    name: str
    value: int
    description: str


class ObservationAdapter(Protocol):
    goal: str

    def actions(self, env: gym.Env) -> tuple[Action, ...]: ...

    def reset(self) -> None: ...

    def encode(self, observation: Any, *, reward: float = 0.0) -> dict: ...


class CartPoleAdapter:
    goal = (
        "Keep the pole upright and the cart on the track for as long as possible. "
        "An episode fails beyond +/-2.4 meters or +/-12 degrees. "
        "Positive position, velocity, angle, and angular velocity point right."
    )

    def actions(self, env: gym.Env) -> tuple[Action, ...]:
        if not isinstance(env.action_space, gym.spaces.Discrete) or (
            env.action_space.n != 2 or env.action_space.start != 0
        ):
            raise ValueError("CartPole requires discrete actions 0 and 1.")
        return (
            Action("push_left", 0, "Apply a leftward force to the cart."),
            Action("push_right", 1, "Apply a rightward force to the cart."),
        )

    def reset(self) -> None:
        pass

    def encode(self, observation: Any, *, reward: float = 0.0) -> dict:
        values = np.asarray(observation, dtype=float)
        if values.shape != (4,) or not np.isfinite(values).all():
            raise ValueError("Expected four finite CartPole observation values.")
        x, velocity, angle, angular_velocity = values.tolist()
        return {
            "cart_position_m": x,
            "cart_velocity_m_per_s": velocity,
            "pole_angle_deg": float(np.degrees(angle)),
            "pole_angular_velocity_deg_per_s": float(np.degrees(angular_velocity)),
        }


class PongAdapter:
    """Palette-based object extraction from standard, unprocessed ALE Pong RGB.

    Only visible pixels are used. Colors/crop agree with OCAtari's Pong detector:
    https://github.com/k4ntz/OC_Atari/blob/master/ocatari/vision/pong.py
    The score display and gray borders are excluded from the ball search.
    """

    goal = (
        "Play Pong as the right-hand green paddle. Return the ball to avoid losing "
        "points and send it past the opponent to score. Coordinates increase right "
        "and down. A missing object means it is not visible in this frame. "
        "Motion is measured between observations, not a prediction of future bounces."
    )
    TOP = 34
    BOTTOM = 194  # exclusive

    def __init__(self, *, frameskip: int = DEFAULT_ATARI_FRAMESKIP) -> None:
        self.frameskip = frameskip
        self._previous_ball: dict | None = None

    def actions(self, env: gym.Env) -> tuple[Action, ...]:
        meanings = env.unwrapped.get_action_meanings()
        descriptions = {
            "NOOP": ("stay", "Keep the paddle at its current height."),
            "FIRE": ("fire", "Press fire without moving the paddle."),
            "RIGHT": ("up", "Move the right-hand paddle upward; y decreases."),
            "LEFT": ("down", "Move the right-hand paddle downward; y increases."),
            "RIGHTFIRE": ("up_fire", "Move the paddle upward while pressing fire."),
            "LEFTFIRE": ("down_fire", "Move the paddle downward while pressing fire."),
        }
        if not isinstance(env.action_space, gym.spaces.Discrete) or (
            env.action_space.start != 0
            or env.action_space.n != len(meanings)
            or set(meanings) != set(descriptions)
        ):
            raise ValueError("Pong adapter expects ALE's six-action minimal action set.")
        return tuple(
            Action(descriptions[name][0], index, descriptions[name][1])
            for index, name in enumerate(meanings)
        )

    def reset(self) -> None:
        self._previous_ball = None

    def _object(self, frame: np.ndarray, color: tuple[int, int, int]) -> dict | None:
        ys, xs = np.where(np.all(frame[self.TOP : self.BOTTOM] == color, axis=2))
        if not xs.size:
            return None
        left, right = int(xs.min()), int(xs.max())
        top, bottom = int(ys.min()) + self.TOP, int(ys.max()) + self.TOP
        return {
            "x": (left + right) / 2,
            "y": (top + bottom) / 2,
            "width": right - left + 1,
            "height": bottom - top + 1,
        }

    def encode(self, observation: Any, *, reward: float = 0.0) -> dict:
        frame = np.asarray(observation)
        if frame.shape != (210, 160, 3) or frame.dtype != np.uint8:
            raise ValueError("Pong needs unprocessed 210x160 RGB uint8 observations.")
        player = self._object(frame, (92, 186, 92))
        opponent = self._object(frame, (213, 130, 74))
        ball = self._object(frame, (236, 236, 236))
        state = self._encode_objects(
            player, opponent, ball, reward=reward, source="visible_rgb_palette_objects"
        )
        state["description"] = describe_pong(state, self.frameskip)
        return state

    def _encode_objects(
        self,
        player: dict | None,
        opponent: dict | None,
        ball: dict | None,
        *,
        reward: float,
        source: str,
    ) -> dict:
        motion = None
        # Scoring, disappearance and episode resets break the motion history.
        if ball is not None and self._previous_ball is not None and reward == 0:
            dx = ball["x"] - self._previous_ball["x"]
            dy = ball["y"] - self._previous_ball["y"]
            motion = {
                "dx_pixels_per_env_step": dx,
                "dy_pixels_per_env_step": dy,
                "horizontal": "right" if dx > 0 else "left" if dx < 0 else "stationary",
                "vertical": "down" if dy > 0 else "up" if dy < 0 else "stationary",
            }
        alignment = None
        if player is not None and ball is not None:
            gap = ball["y"] - player["y"]
            tolerance = (player["height"] + ball["height"]) / 2
            alignment = (
                "below" if gap > tolerance else "above" if gap < -tolerance else "overlapping"
            )
        self._previous_ball = ball
        return {
            "source": source,
            "coordinate_system": COORDINATES,
            "emulator_frames_per_action": self.frameskip,
            "playfield": {"width": 160, "top_y": self.TOP, "bottom_y": self.BOTTOM - 1},
            "player": player,
            "opponent": opponent,
            "ball": ball,
            "ball_motion": motion,
            "ball_relative_to_player": alignment,
        }


class PongRAMAdapter(PongAdapter):
    """Decode Pong's game-specific RAM registers into approximate screen geometry.

    Register meanings and coordinate offsets are documented by OCAtari:
    https://github.com/k4ntz/OC_Atari/blob/master/ocatari/ram/pong.py
    Sprites are clipped to the playfield; display rounding can differ by a pixel.
    """

    goal = (
        "Play Pong as the right-hand paddle. Return the ball to avoid losing points "
        "and send it past the opponent to score. Positions are decoded from game RAM "
        "into approximate screen coordinates: x increases right, y increases down. "
        "A missing object means its RAM coordinates do not indicate an active object "
        "in the playfield. Motion is measured between observations, not a prediction "
        "of future bounces."
    )

    def _box(self, left: int, top: int, width: int, height: int) -> dict | None:
        right, bottom = min(160, left + width), min(self.BOTTOM, top + height)
        left, top = max(0, left), max(self.TOP, top)
        if right <= left or bottom <= top:
            return None
        return {
            "x": (left + right - 1) / 2,
            "y": (top + bottom - 1) / 2,
            "width": right - left,
            "height": bottom - top,
        }

    def encode(self, observation: Any, *, reward: float = 0.0) -> dict:
        ram = np.asarray(observation)
        if ram.shape != (128,) or ram.dtype != np.uint8:
            raise ValueError("Pong RAM needs a 128-byte uint8 observation.")
        player = self._box(140, int(ram[51]) - 13, 4, 16)
        opponent = self._box(16, int(ram[50]) - 15, 4, 16)
        ball = (
            self._box(int(ram[49]) - 49, int(ram[54]) - 14, 2, 4)
            if ram[49] > 49 and ram[54] != 0
            else None
        )
        state = self._encode_objects(
            player, opponent, ball, reward=reward, source="decoded_atari_ram"
        )
        state["score"] = {"player": int(ram[14]), "opponent": int(ram[13])}
        state["description"] = describe_pong(state, self.frameskip)
        return state


class BreakoutRAMAdapter:
    """Decode default-mode Breakout RAM without consulting rendered pixels.

    The position/BCD score registers and brick bit layout are documented by:
    https://github.com/k4ntz/OC_Atari/blob/master/ocatari/ram/breakout.py
    Current paddle shrinkage is not decoded; expose nominal geometry explicitly.
    """

    goal = (
        "Play Breakout: keep the ball above the bottom paddle and clear the brick walls "
        "to score points before running out of lives. The paddle moves horizontally. "
        "Fire serves a ball when the game is waiting for a launch. Returning the ball "
        "from different parts of the paddle can change its direction. The brick map "
        "shows remaining obstacles and gaps. Position and motion are observed facts, "
        "not predicted interception points."
    )
    # Each column selects a bit at this RAM offset plus (5 - row).
    BRICK_COLUMNS = (
        (30, 6),
        (24, 6),
        (24, 4),
        (24, 2),
        (24, 0),
        (18, 0),
        (18, 2),
        (18, 4),
        (18, 6),
        (12, 4),
        (12, 6),
        (6, 6),
        (6, 4),
        (6, 2),
        (6, 0),
        (0, 0),
        (0, 2),
        (0, 4),
    )

    def __init__(self, *, frameskip: int = DEFAULT_ATARI_FRAMESKIP) -> None:
        self.frameskip = frameskip
        self._previous_ball: dict | None = None
        self._previous_lives: int | None = None

    def reset(self) -> None:
        self._previous_ball = None
        self._previous_lives = None

    def actions(self, env: gym.Env) -> tuple[Action, ...]:
        descriptions = {
            "NOOP": ("stay", "Keep the paddle at its current horizontal position."),
            "FIRE": (
                "launch",
                "Serve the next ball if the game is waiting for a launch; the paddle stays still.",
            ),
            "RIGHT": ("right", "Move the bottom paddle rightward; x increases."),
            "LEFT": ("left", "Move the bottom paddle leftward; x decreases."),
        }
        meanings = env.unwrapped.get_action_meanings()
        if not isinstance(env.action_space, gym.spaces.Discrete) or (
            env.action_space.start != 0
            or env.action_space.n != 4
            or set(meanings) != set(descriptions)
        ):
            raise ValueError("Breakout requires ALE's four-action minimal action set.")
        return tuple(
            Action(descriptions[name][0], index, descriptions[name][1])
            for index, name in enumerate(meanings)
        )

    @staticmethod
    def _bcd(value: int) -> int:
        tens, ones = value >> 4, value & 15
        if tens > 9 or ones > 9:
            raise ValueError("Invalid Breakout BCD score register.")
        return tens * 10 + ones

    def encode(self, observation: Any, *, reward: float = 0.0) -> dict:
        ram = np.asarray(observation)
        if ram.shape != (128,) or ram.dtype != np.uint8:
            raise ValueError("Breakout RAM needs a 128-byte uint8 observation.")
        lives = int(ram[57])
        # A center derived from the normal 16px paddle. Do not imply that its
        # collision width remains 16 after the game shrinks it.
        paddle = {"x": int(ram[72]) - 39.5, "y": 190.5, "nominal_width": 16, "width": None}
        ball = (
            {"x": int(ram[99]) - 48.5, "y": int(ram[101]) + 10.5, "width": 2, "height": 4}
            if 0 < int(ram[101]) <= 187
            else None
        )
        motion = None
        if ball is not None and self._previous_ball is not None and self._previous_lives == lives:
            dx = ball["x"] - self._previous_ball["x"]
            dy = ball["y"] - self._previous_ball["y"]
            motion = {
                "dx_pixels_per_env_step": dx,
                "dy_pixels_per_env_step": dy,
                "horizontal": "right" if dx > 0 else "left" if dx < 0 else "stationary",
                "vertical": "down" if dy > 0 else "up" if dy < 0 else "stationary",
            }
        self._previous_ball, self._previous_lives = ball, lives
        grid = np.array(
            [
                [(int(ram[offset + 5 - row]) >> bit) & 1 for offset, bit in self.BRICK_COLUMNS]
                for row in range(6)
            ],
            dtype=np.uint8,
        )
        state = {
            "source": "decoded_atari_ram",
            "coordinate_system": COORDINATES,
            "emulator_frames_per_action": self.frameskip,
            "playfield": {"left_x": 8, "right_x": 151, "top_y": 32, "bottom_y": 195},
            "paddle": paddle,
            "ball": ball,
            "ball_motion": motion,
            "ball_in_play": ball is not None,
            "lives_remaining": lives,
            "score": self._bcd(int(ram[76])) * 100 + self._bcd(int(ram[77])),
            "ball_offset_from_paddle": None
            if ball is None
            else {
                "dx_pixels": ball["x"] - paddle["x"],
                "dy_pixels": ball["y"] - paddle["y"],
            },
            "bricks": {
                "remaining": int(grid.sum()),
                "remaining_by_row": grid.sum(axis=1).tolist(),
                "remaining_by_column": grid.sum(axis=0).tolist(),
                "rows_top_to_bottom": ["".join(map(str, row)) for row in grid.tolist()],
                "legend": "1=brick, 0=gap; columns left to right; each cell is 8x6 pixels, origin (8,57).",
            },
        }
        state["description"] = describe_breakout(state, self.frameskip)
        return state


def make_environment(
    env_id: str,
    *,
    frameskip: int = DEFAULT_ATARI_FRAMESKIP,
    sticky_probability: float = 0.25,
    render_mode: str | None = None,
    pong_state: str = "rgb",
) -> tuple[gym.Env, ObservationAdapter]:
    if env_id == "CartPole-v1":
        return gym.make(env_id, render_mode=render_mode), CartPoleAdapter()
    if env_id in {"ALE/Pong-v5", "ALE/Breakout-v5", "ALE/MsPacman-v5", "ALE/MontezumaRevenge-v5"}:
        import ale_py

        if pong_state not in {"rgb", "ram"}:
            raise ValueError("Pong state must be rgb or ram.")
        if env_id == "ALE/MontezumaRevenge-v5":
            from .montezuma import MontezumaRAMAdapter

            observation_type = "ram"
            adapter = MontezumaRAMAdapter(frameskip=frameskip)
        elif env_id == "ALE/MsPacman-v5":
            from .mspacman import MsPacmanRAMAdapter

            observation_type = "ram"
            adapter = MsPacmanRAMAdapter(frameskip=frameskip)
        elif env_id == "ALE/Breakout-v5":
            observation_type = "ram"
            adapter = BreakoutRAMAdapter(frameskip=frameskip)
        else:
            observation_type = pong_state
            adapter = (
                PongRAMAdapter(frameskip=frameskip)
                if pong_state == "ram"
                else PongAdapter(frameskip=frameskip)
            )
        gym.register_envs(ale_py)
        return gym.make(
            env_id,
            obs_type=observation_type,
            frameskip=frameskip,
            repeat_action_probability=sticky_probability,
            full_action_space=False,
            render_mode=render_mode,
        ), adapter
    raise ValueError("No observation adapter for this environment.")
