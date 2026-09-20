"""Deterministic English summaries of decoded facts, without choosing an action."""

COORDINATES = (
    "Screen coordinates are in pixels, with origin at the top left: "
    "x increases rightward and y increases downward. Object x/y values are centers."
)


def motion_description(motion: dict | None) -> str:
    if motion is None:
        return "Ball motion is unknown because there is no valid consecutive position pair."
    dx = motion["dx_pixels_per_env_step"]
    dy = motion["dy_pixels_per_env_step"]
    return (
        f"Over the last environment step, the ball moved {abs(dx):.1f} pixels "
        f"{motion['horizontal']} and {abs(dy):.1f} pixels {motion['vertical']}. "
        "These are measured displacements, not a prediction after future collisions."
    )


def describe_pong(state: dict, frameskip: int) -> str:
    sentences = [
        "You control the right paddle, which moves up or down; the opponent controls the left paddle.",
        COORDINATES,
        f"The next action is held for {frameskip} emulator frames.",
    ]
    player, ball = state["player"], state["ball"]
    if player is None:
        sentences.append("The player paddle's position is unavailable.")
    else:
        sentences.append(
            f"Your paddle center is ({player['x']:.1f}, {player['y']:.1f}), "
            f"with height {player['height']} pixels."
        )
    if ball is None:
        sentences.append("The ball position is unavailable; its alignment is unknown.")
    else:
        sentences.append(f"The ball center is ({ball['x']:.1f}, {ball['y']:.1f}).")
        if player is not None:
            dy = ball["y"] - player["y"]
            relation = "below" if dy > 0 else "above" if dy < 0 else "level with"
            sentences.append(f"The ball is {abs(dy):.1f} pixels {relation} your paddle center.")
    sentences.append(motion_description(state["ball_motion"]))
    sentences.append(
        "Rightward ball motion goes toward your side of the court; leftward goes toward the opponent's side."
    )
    if "score" in state:
        sentences.append(
            f"Score: you {state['score']['player']}, opponent {state['score']['opponent']}."
        )
    return " ".join(sentences)


def describe_breakout(state: dict, frameskip: int) -> str:
    paddle, ball = state["paddle"], state["ball"]
    sentences = [
        "You control the bottom paddle, which moves left or right to return the ball toward the bricks.",
        COORDINATES,
        f"The next action is held for {frameskip} emulator frames.",
        f"Your paddle's approximate center is ({paddle['x']:.1f}, {paddle['y']:.1f}). "
        "Its nominal width is 16 pixels; the current width is not decoded and may be smaller.",
    ]
    if ball is None:
        sentences.append(
            "No active ball is indicated by RAM. Fire serves the next ball when the game "
            "is ready and lives remain."
        )
    else:
        dx, dy = ball["x"] - paddle["x"], ball["y"] - paddle["y"]
        horizontal = "right of" if dx > 0 else "left of" if dx < 0 else "aligned with"
        vertical = "below" if dy > 0 else "above" if dy < 0 else "level with"
        sentences.append(
            f"The ball center is ({ball['x']:.1f}, {ball['y']:.1f}), "
            f"{abs(dx):.1f} pixels {horizontal} the paddle center and "
            f"{abs(dy):.1f} pixels {vertical} it."
        )
    sentences.extend(
        [
            motion_description(state["ball_motion"]),
            "Downward ball motion goes toward the bottom of the screen; upward goes toward the brick wall. "
            "A ball that passes below the paddle is lost.",
            f"Lives remaining: {state['lives_remaining']}. Score: {state['score']}. "
            f"Bricks remaining: {state['bricks']['remaining']}.",
            f"Remaining bricks per row, from top to bottom: {state['bricks']['remaining_by_row']}. "
            "In the brick map, rows run top to bottom and columns left to right; 1 means brick, 0 means gap.",
        ]
    )
    return " ".join(sentences)
