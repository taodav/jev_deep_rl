import numpy as np
import pytest

from jev_rl.adapters import BreakoutRAMAdapter, PongRAMAdapter, make_environment


def test_breakout_state_describes_controls_coordinates_and_launch_phase():
    ram = np.zeros(128, dtype=np.uint8)
    ram[57], ram[72] = 5, 146
    adapter = BreakoutRAMAdapter(frameskip=8)
    initial = adapter.encode(ram)
    assert initial["ball"] is None
    assert initial["ball_in_play"] is False
    assert "No active ball" in initial["description"]
    assert "Fire serves the next ball" in initial["description"]
    assert "left or right" in initial["description"]
    assert "x increases rightward and y increases downward" in initial["description"]
    assert "8 emulator frames" in initial["description"]
    assert initial["paddle"]["width"] is None
    assert "current width is not decoded" in initial["description"]
    ram[99], ram[101], ram[76], ram[77] = 149, 129, 0x01, 0x23
    moving = adapter.encode(ram)
    assert moving["ball"] == {"x": 100.5, "y": 139.5, "width": 2, "height": 4}
    assert moving["score"] == 123
    assert "6.0 pixels left of" in moving["description"]
    assert "51.0 pixels above" in moving["description"]
    assert "motion is unknown" in moving["description"]
    ram[99] += 4
    ram[101] += 2
    # A brick reward must not clear motion history in Breakout.
    moving = adapter.encode(ram, reward=1)
    assert moving["ball_motion"]["dx_pixels_per_env_step"] == 4
    assert "4.0 pixels right" in moving["description"]
    assert "2.0 pixels down" in moving["description"]
    ram[57] -= 1
    assert adapter.encode(ram)["ball_motion"] is None
    adapter.reset()
    assert adapter.encode(ram)["ball_motion"] is None


def test_pong_text_describes_vertical_controls_and_missing_motion():
    ram = np.zeros(128, dtype=np.uint8)
    ram[[13, 14, 49, 50, 51, 54]] = [4, 7, 149, 105, 113, 94]
    adapter = PongRAMAdapter(frameskip=4)
    description = adapter.encode(ram)["description"]
    assert "up or down" in description
    assert "26.0 pixels above" in description
    assert "Score: you 7, opponent 4" in description
    assert "motion is unknown" in description
    ram[49] += 4
    ram[54] += 2
    description = adapter.encode(ram)["description"]
    assert "4.0 pixels right" in description
    assert "2.0 pixels down" in description


def test_breakout_brick_map_bit_order_and_precomputed_counts():
    ram = np.zeros(128, dtype=np.uint8)
    # Top-left and bottom-right cells use independent registers and bit positions.
    ram[35] = 1 << 6
    ram[0] = 1 << 4
    bricks = BreakoutRAMAdapter().encode(ram)["bricks"]
    assert bricks["remaining"] == 2
    assert bricks["rows_top_to_bottom"][0] == "1" + "0" * 17
    assert bricks["rows_top_to_bottom"][-1] == "0" * 17 + "1"
    assert bricks["remaining_by_row"] == [1, 0, 0, 0, 0, 1]
    assert bricks["remaining_by_column"] == [1] + [0] * 16 + [1]


def test_breakout_ram_against_real_game_controls_bricks_and_lives():
    pytest.importorskip("ale_py")
    env, adapter = make_environment(
        "ALE/Breakout-v5", render_mode="rgb_array", sticky_probability=0.0
    )
    try:
        actions = {action.name: action.value for action in adapter.actions(env)}
        assert actions == {"stay": 0, "launch": 1, "right": 2, "left": 3}
        paddle_x = {}
        for action in ["left", "right"]:
            env.reset(seed=7)
            ram, *_ = env.step(actions[action])
            paddle_x[action] = adapter.encode(ram)["paddle"]["x"]
        assert paddle_x["left"] < paddle_x["right"]
        ram, info = env.reset(seed=7)
        adapter.reset()
        state = adapter.encode(ram)
        assert state["bricks"]["remaining"] == 108
        assert state["lives_remaining"] == info["lives"] == 5
        colors = np.array(
            [
                [200, 72, 72],
                [198, 108, 58],
                [180, 122, 48],
                [162, 162, 42],
                [72, 160, 72],
                [66, 72, 200],
            ],
            dtype=np.uint8,
        )
        ys, xs = 60 + 6 * np.arange(6), 12 + 8 * np.arange(18)
        total_reward = 0.0
        observed_balls = 0
        for _ in range(500):
            # Test driver only: exercise serving and brick removal with a simple
            # reactive paddle. The runtime still uses the selected random/Jev policy.
            if state["ball"] is None:
                action = actions["launch"]
            elif state["ball"]["x"] > state["paddle"]["x"]:
                action = actions["right"]
            else:
                action = actions["left"]
            ram, reward, terminated, truncated, info = env.step(action)
            state = adapter.encode(ram, reward=reward)
            total_reward += reward
            assert state["score"] == total_reward
            assert state["lives_remaining"] == info["lives"]
            frame = env.render()
            visible = np.all(frame[ys[:, None], xs] == colors[:, None, :], axis=2)
            decoded = np.array(
                [[cell == "1" for cell in row] for row in state["bricks"]["rows_top_to_bottom"]]
            )
            unobscured = np.ones((6, 18), dtype=bool)
            if state["ball"] is not None:
                ball = state["ball"]
                unobscured = (abs(xs - ball["x"])[None, :] > 4) | (abs(ys - ball["y"])[:, None] > 5)
            assert np.array_equal(visible[unobscured], decoded[unobscured])
            if state["ball"] is not None:
                ball = state["ball"]
                # Ball color changes in brick rows and at walls. Compare the red
                # ball only in open lower court, clear of the paddle and borders.
                x, y = int(ball["x"]), int(ball["y"])
                if 10 < x < 149 and 96 < y < 186:
                    neighborhood = frame[y - 2 : y + 3, x - 2 : x + 3]
                    assert np.any(np.all(neighborhood == [200, 72, 72], axis=2))
                    observed_balls += 1
            if terminated or truncated:
                break
        assert observed_balls > 30
        assert total_reward > 0
    finally:
        env.close()
