import numpy as np
import pytest

from jev_rl.adapters import PongAdapter, PongRAMAdapter, make_environment


def test_ram_decoding_coordinates_scores_and_motion():
    ram = np.zeros(128, dtype=np.uint8)
    ram[[13, 14, 49, 50, 51, 54]] = [4, 7, 149, 105, 113, 94]
    adapter = PongRAMAdapter()
    state = adapter.encode(ram)
    assert state["source"] == "decoded_atari_ram"
    assert state["score"] == {"player": 7, "opponent": 4}
    assert state["player"] == {"x": 141.5, "y": 107.5, "width": 4, "height": 16}
    assert state["ball"] == {"x": 100.5, "y": 81.5, "width": 2, "height": 4}
    assert state["ball_relative_to_player"] == "above"
    assert state["ball_motion"] is None
    ram[49] += 4
    ram[54] += 2
    state = adapter.encode(ram)
    assert state["ball_motion"]["dx_pixels_per_env_step"] == 4
    assert state["ball_motion"]["dy_pixels_per_env_step"] == 2
    assert adapter.encode(ram, reward=-1)["ball_motion"] is None
    empty = adapter.encode(np.zeros(128, dtype=np.uint8))
    assert empty["ball"] is empty["player"] is empty["opponent"] is None
    assert adapter.encode(ram)["ball_motion"] is None
    adapter.reset()
    assert adapter.encode(ram)["ball_motion"] is None


@pytest.mark.parametrize("observation", [np.zeros(127, dtype=np.uint8), np.zeros(128)])
def test_ram_rejects_wrong_shape_or_dtype(observation):
    with pytest.raises(ValueError):
        PongRAMAdapter().encode(observation)


def test_ram_coordinates_agree_with_rendered_pong_objects():
    pytest.importorskip("ale_py")
    env, adapter = make_environment("ALE/Pong-v5", pong_state="ram", render_mode="rgb_array")
    pixel_adapter = PongAdapter()
    compared = {name: 0 for name in ["player", "opponent", "ball"]}
    try:
        env.reset(seed=7)
        env.action_space.seed(7)
        for _ in range(300):
            ram, reward, terminated, truncated, _ = env.step(env.action_space.sample())
            decoded = adapter.encode(ram, reward=reward)
            pixels = pixel_adapter.encode(env.render(), reward=reward)
            for name in compared:
                if pixels[name] is not None:
                    assert decoded[name] is not None
                    # RAM positions describe game geometry; framebuffer rounding
                    # and partially clipped sprites can shift a center by one pixel.
                    assert abs(decoded[name]["x"] - pixels[name]["x"]) <= 1
                    assert abs(decoded[name]["y"] - pixels[name]["y"]) <= 1
                    compared[name] += 1
            if terminated or truncated:
                break
        assert all(count > 100 for count in compared.values())
    finally:
        env.close()
