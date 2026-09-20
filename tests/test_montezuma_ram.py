import json

import httpx2
import numpy as np
import pytest
from typesafe_sdk import RetryPolicy, TypeSafeClient

from jev_rl.adapters import make_environment
from jev_rl.cli import main
from jev_rl.montezuma import MontezumaRAMAdapter
from jev_rl.policies import JevPolicy
from npy_to_gif import convert_recording


def ram_state():
    r = np.zeros(128, dtype=np.uint8)
    r[3], r[42], r[43], r[44], r[45], r[49] = 1, 77, 235, 14, 254, 4
    r[26], r[28], r[46], r[47], r[58], r[67], r[126] = 124, 124, 240, 58, 5, 31, 139
    return r


@pytest.fixture
def environment():
    pytest.importorskip("ale_py")
    env, adapter = make_environment(
        "ALE/MontezumaRevenge-v5", frameskip=1, sticky_probability=0.0, render_mode="rgb_array"
    )
    env.reset(seed=7)
    yield env, adapter
    env.close()


def test_common_coordinates_inventory_and_current_room_objects():
    ram = ram_state()
    ram[19:22] = [0x12, 0x34, 0x56]
    ram[65] = 128 | 64 | 32 | 16 | 4 | 1
    state = MontezumaRAMAdapter().encode(ram)
    assert state["score"] == 123456 and state["lives_remaining"] == 6
    assert state["inventory"] == dict(keys=2, swords=2, torch=True, amulet=True)
    assert (state["player"]["x"], state["player"]["y"]) == (80, 83)
    assert state["local_geometry"]["feet_y"] == 93
    assert state["local_geometry"]["aligned_surface_ids"] == ["platform_2"]
    assert state["items"][0]["kind"] == "key"
    assert (state["items"][0]["x"], state["items"][0]["y"]) == (16.5, 106.5)
    assert state["hazards"][0]["kind"] == "skull"
    assert all(door["locked"] for door in state["doors"])
    assert all(ex["destination"] is None for ex in state["geometry"]["exit_candidates"])
    json.dumps(state, allow_nan=False)


def test_room_changes_reset_motion_and_never_leak_old_objects_or_geometry():
    adapter = MontezumaRAMAdapter()
    ram = ram_state()
    first = adapter.encode(ram)
    ram[42] += 2
    assert adapter.encode(ram)["player"]["motion_per_decision"] == dict(dx=2, dy=0)
    ram[3], ram[49] = 2, 5
    second = adapter.encode(ram)
    assert not second["items"] and not second["doors"]
    assert second["hazards"][0]["kind"] == "skull"
    assert second["player"]["motion_per_decision"] is None
    assert second["geometry"] != first["geometry"]
    assert second["exploration"]["visited_rooms_this_level"] == [1, 2]
    ram[3] = 255
    unknown = adapter.encode(ram)
    assert unknown["geometry"] is unknown["local_geometry"] is None
    assert unknown["room"]["geometry_status"] == "unknown"
    adapter.reset()
    ram[3] = 1
    assert adapter.encode(ram)["exploration"]["visited_rooms_this_level"] == [1]


def test_life_change_death_and_teleport_do_not_create_motion():
    ram = ram_state()
    adapter = MontezumaRAMAdapter()
    adapter.encode(ram)
    for register, value in ((58, 4), (126, 102), (126, 139), (42, 20)):
        ram[register] = value
        assert adapter.encode(ram)["player"]["motion_per_decision"] is None
    ram[58], ram[126] = 0, 0x60
    final = adapter.encode(ram)
    assert final["game_over"] and final["lives_remaining"] == 0
    with pytest.raises(ValueError):
        adapter.encode(np.zeros((210, 160, 3), dtype=np.uint8))


def test_darkness_unknown_items_temporary_platforms_and_reset_are_explicit():
    ram = ram_state()
    adapter = MontezumaRAMAdapter()
    ram[3], ram[49], ram[34] = 18, 4, 214
    dark = adapter.encode(ram)
    assert dark["room"]["lighting"] == "dark" and dark["items"] is None
    assert not dark["geometry"]["temporary_platforms"][0]["present"]
    ram[65], ram[34] = 128, 232
    lit = adapter.encode(ram)
    assert lit["room"]["lighting"] == "lit" and lit["items"][0]["kind"] == "key"
    assert not lit["geometry"]["temporary_platforms"][0]["present"]
    ram[34] = 100
    assert adapter.encode(ram)["geometry"]["temporary_platforms"][0]["present"]
    assert not dark["geometry"]["temporary_platforms"][0][
        "present"
    ]  # No in-place edits to old states.


def test_exit_destinations_are_learned_only_from_crossings_and_reset_per_episode():
    ram = ram_state()
    adapter = MontezumaRAMAdapter()
    ram[42] = 148
    adapter.encode(ram)
    ram[3], ram[42] = 2, 5
    adapter.encode(ram)
    ram[3], ram[42] = 1, 145
    returned = adapter.encode(ram)
    right = next(ex for ex in returned["geometry"]["exit_candidates"] if ex["direction"] == "right")
    assert right["destination"] == dict(level=0, room=2)
    # The reverse crossing was observed near the left boundary, too.
    ram[3] = 2
    left = next(
        ex for ex in adapter.encode(ram)["geometry"]["exit_candidates"] if ex["direction"] == "left"
    )
    assert left["destination"] == dict(level=0, room=1)
    adapter.reset()
    assert all(
        ex["destination"] is None for ex in adapter.encode(ram)["geometry"]["exit_candidates"]
    )


def test_real_player_motion_ladder_jump_death_and_room_transition(environment):
    env, adapter = environment
    ale = env.unwrapped.ale
    base = ale.cloneState()
    initial = adapter.encode(ale.getRAM())
    assert initial["lives_remaining"] == env.unwrapped.ale.lives() == 6
    assert initial["local_geometry"]["aligned_surface_ids"] == ["platform_2"]
    for _ in range(6):
        ram, *_ = env.step(3)
    assert adapter.encode(ram)["player"]["motion_per_decision"] == dict(dx=6, dy=0)
    ale.restoreState(base)
    adapter.reset()
    adapter.encode(ale.getRAM())
    for _ in range(6):
        ram, *_ = env.step(5)
    assert adapter.encode(ram)["player"]["on_ladder"]
    ale.restoreState(base)
    adapter.reset()
    adapter.encode(ale.getRAM())
    ram, *_ = env.step(1)
    adapter.encode(ram)
    ram, *_ = env.step(1)
    assert adapter.encode(ram)["player"]["motion_per_decision"]["dy"] < 0
    ale.restoreState(base)
    adapter.reset()
    for _ in range(34):
        ram, *_ = env.step(3)
        state = adapter.encode(ram)
    assert state["lives_remaining"] == 5 and state["player"]["motion_per_decision"] is None
    ale.restoreState(base)
    adapter.reset()
    ale.setRAM(42, 148)
    adapter.encode(ale.getRAM())
    for _ in range(8):
        ram, *_ = env.step(3)
        state = adapter.encode(ram)
    assert state["room"]["id"] == 2 and state["items"] == []
    assert len(state["hazards"]) == 2


@pytest.mark.parametrize("room", range(24))
def test_all_room_templates_load_after_real_emulator_room_initialization(environment, room):
    env, adapter = environment
    ale = env.unwrapped.ale
    base = ale.cloneState()
    # Calibration only: position at a room boundary, let the ROM initialize the
    # next room. This is not policy play or evidence of successful exploration.
    for direction in ("right", "left"):
        if direction == "right" and room == 0:
            continue
        ale.restoreState(base)
        ale.setRAM(3, room - 1 if direction == "right" else room + 1)
        ale.setRAM(42, 150 if direction == "right" else 0)
        ale.setRAM(65, 128)
        for _ in range(5):
            ram, *_ = env.step(3 if direction == "right" else 4)
        if int(ram[3]) == room:
            break
    assert int(ram[3]) == room
    state = adapter.encode(ram)
    assert state["room"]["id"] == room
    geometry = state["geometry"]
    assert geometry is not None
    json.dumps(state, allow_nan=False)
    frame = env.render()
    # Check that reference rectangles coincide with rendered geometry. Room 15
    # has a uniform background hiding geometry; this test makes no visual claim
    # there. Pixel overlap does NOT establish traversal, precise collision boxes,
    # rope mechanics, or jump safety.
    colors = ((66, 158, 130), (24, 59, 157), (232, 204, 99), (236, 236, 236))
    if room != 15:
        for group in ("platforms", "walls", "ladders", "ropes"):
            for obj in geometry[group]:
                patch = frame[
                    max(48, obj["y_top"] - 1) : min(200, obj["y_bottom"] + 1),
                    max(0, obj["x_min"] - 1) : min(160, obj["x_max"] + 1),
                ]
                assert any(np.any(np.all(patch == color, axis=2)) for color in colors), (
                    room,
                    group,
                    obj,
                )


def test_offline_cli_recording_gif_and_real_sdk_request(environment, tmp_path):
    output = tmp_path / "montezuma.npy"
    assert (
        main(
            [
                "--env",
                "ALE/MontezumaRevenge-v5",
                "--policy",
                "random",
                "--max-steps",
                "12",
                "--save-npy",
                str(output),
            ]
        )
        == 0
    )
    run = np.load(output, allow_pickle=True).item()
    assert run["metadata"]["observation_source"] == "ram"
    assert run["episodes"][0]["frames"].shape[0] == 13
    assert convert_recording(output)[0]["source_frames"] == 13
    env, adapter = environment
    state = {
        "environment": "ALE/MontezumaRevenge-v5",
        "goal": adapter.goal,
        "step": 0,
        "observation": adapter.encode(env.unwrapped.ale.getRAM()),
        "previous_action": None,
        "previous_reward": None,
    }
    actions = adapter.actions(env)

    def respond(request):
        payload = json.loads(request.content)
        assert payload["state"] == state
        assert len(payload["questions"]["action"]["criteria"]) == 18
        return httpx2.Response(
            200,
            json={
                "model": "jev-1.13.0",
                "usage": {"input_tokens": 1, "output_tokens": 1},
                "answers": {
                    "action": {
                        "type": "choice",
                        "choice": "down",
                        "confidence": 1.0,
                        "probabilities": {a.name: float(a.name == "down") for a in actions},
                    }
                },
            },
        )

    with TypeSafeClient(
        api_key="test-placeholder",
        transport=httpx2.MockTransport(respond),
        retry=RetryPolicy(max_retries=0),
    ) as client:
        assert JevPolicy(client).decide(state, actions).action == "down"
