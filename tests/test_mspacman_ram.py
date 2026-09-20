import json

import httpx2
import numpy as np
import pytest
from typesafe_sdk import RetryPolicy, TypeSafeClient

from jev_rl.adapters import make_environment
from jev_rl.cli import main
from jev_rl.mspacman import MsPacmanRAMAdapter
from jev_rl.mspacman_maze import X, Y, maze_for, maze_position
from jev_rl.policies import JevPolicy
from jev_rl.recording import NpyRecorder
from jev_rl.responses import CaptureTransport, ResponseCapture
from jev_rl.runner import run_episode
from npy_to_gif import convert_recording


def ram_state():
    ram = np.zeros(128, dtype=np.uint8)
    ram[6:11] = 88
    ram[12:17] = [80, 80, 80, 50, 98]
    ram[19], ram[39], ram[56], ram[123] = 3, 255, 3, 2
    return ram


@pytest.fixture
def environment():
    pytest.importorskip("ale_py")
    env, adapter = make_environment(
        "ALE/MsPacman-v5", frameskip=1, sticky_probability=0.0, render_mode="rgb_array"
    )
    env.reset(seed=7)
    for _ in range(300):
        env.step(0)
    yield env, adapter
    env.close()


def test_score_pellet_bits_power_bits_and_text_are_unambiguous():
    ram = ram_state()
    ram[120:123] = [0x56, 0x34, 0x12]
    # Row 2 has no walls; sample both edge bits and each interleaved byte half.
    ram[65], ram[66], ram[67] = 80, 129, 66
    ram[117] = 32 | 4 | 3  # Low two bits are animation flags, not pellets.
    ram[60] = 2  # An unused pellet bit inside a wall in row 0 must be ignored.
    state = MsPacmanRAMAdapter().encode(ram)
    assert state["score"] == 123456 and state["lives_remaining"] == 3
    assert state["maze"]["pellets_remaining"] == 6
    assert [c for c, char in enumerate(state["maze"]["rows_top_to_bottom"][2]) if char == "."] == [
        0,
        1,
        8,
        12,
        13,
        17,
    ]
    rows = state["maze"]["rows_top_to_bottom"]
    assert [(r, c) for r, row in enumerate(rows) for c, char in enumerate(row) if char == "o"] == [
        (1, 0),
        (12, 17),
    ]
    assert state["ghosts"][0]["in_house"]
    assert state["ghosts"][0]["row"] == 6.5 and state["ghosts"][0]["column"] == 8.5
    assert set(state["navigation"]["corridors_from_player"]) == {"left", "right"}
    assert state["navigation"]["corridors_from_player"]["left"]["distance_cells"] == 1.5
    assert state["emulator_frames_per_action"] == 2 and "noop can continue" in state["description"]
    assert "Returning-eye status is unknown" in state["description"]
    assert "123456" not in state["description"] and "lives 3" not in state["description"]
    json.dumps(state, allow_nan=False)


def test_motion_resets_on_animation_life_or_layout_change_and_handles_tunnels():
    adapter = MsPacmanRAMAdapter()
    ram = ram_state()
    assert adapter.encode(ram)["player"]["motion_per_decision"] is None
    ram[10] -= 2
    assert adapter.encode(ram, reward=10)["player"]["motion_per_decision"]["columns"] == -0.167
    for register, value in [(39, 51), (39, 255), (123, 1), (0, 1)]:
        ram[register] = value
        assert adapter.encode(ram)["player"]["motion_per_decision"] is None
    adapter.reset()
    assert adapter.encode(ram)["player"]["motion_per_decision"] is None
    ram[0], ram[10], ram[16] = 0, 12, 50
    adapter.reset()
    adapter.encode(ram)
    ram[10] = 170
    motion = adapter.encode(ram)["player"]["motion_per_decision"]
    assert motion == {"rows": 0, "columns": -0.1}
    ram[123], ram[39] = 0, 83
    state = adapter.encode(ram)
    assert state["phase"] == "game_over" and state["lives_remaining"] == 0
    with pytest.raises(ValueError):
        adapter.encode(np.zeros((210, 160, 3), dtype=np.uint8))
    ram[0] = 4
    with pytest.raises(ValueError, match="maze"):
        adapter.encode(ram)


def test_fractional_coordinates_cover_every_cell_middle_gap_and_both_tunnel_boundaries():
    for row, y in enumerate(Y):
        for col, x in enumerate(X):
            assert maze_position(x, y) == (row, col)
    assert maze_position(9, 163.5) == (13, 0)  # The stuck bottom-left position.
    assert maze_position(79, 103.5) == (8, 8.5)  # Middle gap is 12px, not 8px.
    assert maze_position(13, 13.5) == (0.5, 0.5)
    assert maze_position(3, 55.5) == pytest.approx((4, 17.7))
    assert maze_position(161, 55.5) == pytest.approx((4, 17.6))

    adapter = MsPacmanRAMAdapter()
    ram = ram_state()
    ram[10], ram[16] = 18, 50
    adapter.encode(ram)
    ram[10] = 17
    crossing_zero = adapter.encode(ram)["player"]
    assert crossing_zero["column"] == 17.95
    assert crossing_zero["motion_per_decision"] == {"rows": 0, "columns": -0.05}
    ram[10] = 18
    assert adapter.encode(ram)["player"]["motion_per_decision"] == {"rows": 0, "columns": 0.05}
    ram[16] += 1
    assert adapter.encode(ram)["player"]["motion_per_decision"] == {"rows": 0.083, "columns": 0}


def test_stuck_heading_is_distinct_from_motion_and_all_route_distances_use_cells():
    adapter = MsPacmanRAMAdapter()
    ram = ram_state()
    ram[10], ram[16], ram[11], ram[17] = 18, 158, 18, 2
    adapter.encode(ram)
    state = adapter.encode(ram)
    assert state["player"] == {
        "row": 13,
        "column": 0,
        "heading": "left",
        "motion_per_decision": {"rows": 0, "columns": 0},
        "wall_up": False,
        "wall_right": False,
        "wall_down": True,
        "wall_left": True,
    }
    assert state["fruit"]["row"] == state["fruit"]["column"] == 0
    corridors = state["navigation"]["corridors_from_player"]
    assert set(corridors) == {"up", "right"}
    assert corridors["up"]["distance_cells"] == corridors["right"]["distance_cells"] == 3
    assert corridors["up"]["next_corner_or_junction"] == {"row": 10, "column": 0}
    assert corridors["right"]["next_corner_or_junction"] == {"row": 13, "column": 3}
    maze = maze_for(0)
    distances = maze.distances_in_cells(maze.locate(79, 103.5))
    assert distances[(8, 8)] == distances[(8, 9)] == 0.5
    # The wrap is one connection, including fractional progress along it.
    assert maze.distances_in_cells(maze.locate(9, 55.5))[(4, 17)] == 1
    assert maze.distances_in_cells(maze.locate(159, 55.5))[(4, 0)] == 0.5


@pytest.mark.parametrize(
    "x,y,expected",
    [
        (9, 7.5, (True, False, False, True)),  # Top-left: outer walls.
        (9, 163.5, (False, False, True, True)),  # Previously stuck corner.
        (9, 161.5, (False, True, False, True)),  # Just above it: cannot turn right yet.
        (9, 55.5, (True, False, True, False)),  # Left tunnel mouth wraps, not a wall.
        (149, 55.5, (True, False, True, False)),  # Right tunnel mouth wraps too.
        (161, 55.5, (True, False, True, False)),  # Inside the wrapping segment.
        (79, 103.5, (True, False, True, False)),  # Between two horizontal cells.
        (65, 103.5, (True, False, False, False)),  # Three-way junction.
        (79, 85.5, (None, None, None, None)),  # No valid player corridor in house.
    ],
)
def test_wall_flags_follow_current_geometry_without_rounding_into_a_junction(x, y, expected):
    ram = ram_state()
    ram[10], ram[16] = x + 9, int(y - 5.5)
    adapter = MsPacmanRAMAdapter()
    state = adapter.encode(ram)
    player = state["player"]
    fields = [f"wall_{direction}" for direction in ("up", "right", "down", "left")]
    assert tuple(player[field] for field in fields) == expected
    # Pellet removal, ghost vulnerability, and reward must not change static walls.
    ram[59:101] = 0
    ram[1:5] = 128
    after = adapter.encode(ram, reward=50)["player"]
    assert tuple(after[field] for field in fields) == expected


@pytest.mark.parametrize("layout", range(4))
def test_all_maze_walls_and_pellet_bits_match_the_real_renderer(environment, layout):
    env, adapter = environment
    ale = env.unwrapped.ale
    ale.setRAM(0, layout)  # Controlled calibration fixture, never a runtime action.
    for _ in range(4):
        ram, *_ = env.step(0)
    state, frame = adapter.encode(ram), env.render()
    maze = maze_for(layout)
    # Only this renderer check uses screen positions; the policy uses maze coordinates.
    actor_pixels = [
        (int(ram[x]) - 9, int(ram[y]) + 5.5)
        for x, y in [(10, 16), (6, 12), (7, 13), (8, 14), (9, 15)]
    ]
    checked = 0
    for row, col in maze.graph:
        if (row, col) in {(1, 0), (1, 17), (12, 0), (12, 17)}:
            continue  # Power pellets have their own bits and blink.
        x, y = X[col], Y[row]
        if any(abs(x - ax) < 7 and abs(y - ay) < 8 for ax, ay in actor_pixels):
            continue  # Actors can obscure a correctly decoded pellet.
        visible = bool(np.all(frame[int(y), x] == [228, 111, 111]))
        assert visible == (state["maze"]["rows_top_to_bottom"][row][col] == ".")
        checked += 1
    assert checked > 130
    # Remove food in this disposable emulator fixture to reveal corridor geometry.
    for register in range(59, 101):
        ale.setRAM(register, 0)
    ale.setRAM(117, int(ram[117]) & ~60)
    for _ in range(4):
        env.step(0)
    walls = np.all(env.render() == [228, 111, 111], axis=2)
    for row, y in enumerate(Y):
        for col, x in enumerate(X):
            floor = not walls[int(y) - 3 : int(y) + 4, x - 3 : x + 4].any()
            house = 64 < x < 96 and 64 < y < 96
            assert (floor and not house) == ((row, col) in maze.graph)
    # Tunnels must be openings at both physical screen edges.
    for row in range(14):
        opening = not walls[
            int(Y[row]) - 3 : int(Y[row]) + 4, list(range(13)) + list(range(146, 160))
        ].any()
        assert opening == (row in maze.tunnel_rows)
    assert len(maze.distances_in_cells(maze.locate(79, 103.5))) == len(maze.graph)
    # Check actual joystick travel at every directed graph edge, not just pixels.
    snapshot = ale.cloneState()
    action_values = {"up": 1, "right": 2, "left": 3, "down": 4}
    for (row, col), neighbors in maze.graph.items():
        for direction in neighbors:
            ale.restoreState(snapshot)
            ale.setRAM(10, X[col] + 9)
            ale.setRAM(16, int(Y[row] - 5.5))
            for ghost in range(4):
                ale.setRAM(6 + ghost, 88)
                ale.setRAM(12 + ghost, 80)
            for _ in range(8):
                ram, *_ = env.step(action_values[direction])
            dx = (int(ram[10]) - (X[col] + 9) + 80) % 160 - 80
            dy = int(ram[16]) - int(Y[row] - 5.5)
            assert {"up": dy < 0, "down": dy > 0, "right": dx > 0, "left": dx < 0}[direction]


def test_power_effect_individual_ghost_edibility_and_expiry(environment):
    env, adapter = environment
    ale = env.unwrapped.ale
    # Place the player just below the upper-left power pellet in an active game.
    ale.setRAM(10, 18)
    ale.setRAM(16, 18)
    for _ in range(40):
        ram, reward, *_ = env.step(1)
        state = adapter.encode(ram, reward=reward)
        if reward == 50:
            break
    else:
        pytest.fail("The power pellet was not consumed.")
    assert state["maze"]["rows_top_to_bottom"][1][0] != "o"
    assert all(ghost["frightened"] for ghost in state["ghosts"])
    assert 0 < state["power_effect"]["remaining_active_frames_upper_bound"] <= 512
    for _ in range(50):
        ram, *_ = env.step(1)
    assert np.any(np.all(env.render() == [66, 114, 194], axis=2))  # Frightened blue.
    # Eating one ghost clears its own flag while the shared effect continues.
    for _ in range(16):
        ale.setRAM(10, int(ram[9]))
        ale.setRAM(16, int(ram[15]))
        ram, reward, *_ = env.step(0)
        if reward >= 200:
            break
    state = adapter.encode(ram, reward=reward)
    assert reward >= 200 and state["ghosts"][3]["frightened"] is False
    assert any(ghost["frightened"] for ghost in state["ghosts"][:3])
    assert state["power_effect"]["remaining_active_frames_upper_bound"] <= 512
    ale.setRAM(10, 18)
    ale.setRAM(16, 2)
    for _ in range(600):
        ram, *_ = env.step(1)
        if not int(ram[116]) & 63:
            break
    state = adapter.encode(ram)
    assert not any(ghost["frightened"] for ghost in state["ghosts"])
    assert state["power_effect"]["remaining_active_frames_upper_bound"] == 0


def test_noop_does_not_brake_and_tunnel_really_wraps(environment):
    env, adapter = environment
    ale = env.unwrapped.ale
    actions = {action.name: action.value for action in adapter.actions(env)}
    assert actions == {
        "noop": 0,
        "up": 1,
        "right": 2,
        "left": 3,
        "down": 4,
        "upright": 5,
        "upleft": 6,
        "downright": 7,
        "downleft": 8,
    }
    ale.setRAM(10, 58)
    ale.setRAM(16, 50)
    for _ in range(8):
        ram, *_ = env.step(actions["noop"])
    assert ram[10] < 58
    ale.setRAM(10, 18)
    for _ in range(30):
        ram, *_ = env.step(actions["left"])
        if ram[10] > 150:
            break
    assert ram[10] > 150 and ram[16] == 50


def test_real_episode_ends_at_game_over_with_consistent_score_and_lives(environment):
    env, adapter = environment
    env.action_space.seed(7)
    score = adapter.encode(env.unwrapped.ale.getRAM())["score"]
    for step in range(10000):
        ram, reward, terminated, truncated, info = env.step(env.action_space.sample())
        state = adapter.encode(ram, reward=reward)
        score += reward
        assert state["score"] == score
        assert state["lives_remaining"] == (0 if terminated else info["lives"])
        if terminated or truncated:
            break
    assert terminated and state["phase"] == "game_over"


def test_mock_jev_records_exact_mspacman_states_and_responses(tmp_path, environment):
    env, adapter = environment
    actions = adapter.actions(env)
    requests = []

    def respond(request):
        requests.append(json.loads(request.content))
        return httpx2.Response(
            200,
            json={
                "model": "jev-1.13.0",
                "usage": {"input_tokens": 1000, "output_tokens": 30},
                "answers": {
                    "action": {
                        "type": "choice",
                        "choice": "left",
                        "confidence": 1.0,
                        "probabilities": {
                            action.name: float(action.name == "left") for action in actions
                        },
                    }
                },
            },
        )

    with NpyRecorder(tmp_path / "mock.npy") as recorder:
        with ResponseCapture(
            tmp_path / "mock.responses.jsonl", recorder.api_responses.append
        ) as capture:
            with TypeSafeClient(
                api_key="offline-test-key",
                retry=RetryPolicy(max_retries=0),
                transport=CaptureTransport(capture, httpx2.MockTransport(respond)),
            ) as client:
                result = run_episode(
                    env,
                    adapter,
                    JevPolicy(client, capture),
                    seed=7,
                    max_steps=4,
                    recording=recorder.start_episode(0, 7),
                )
    recording = np.load(tmp_path / "mock.npy", allow_pickle=True).item()
    episode = recording["episodes"][0]
    assert result.steps == len(requests) == len(recording["api_responses"]) == 4
    assert episode["observations"].shape == (5, 128)
    assert episode["frames"].shape == (5, 210, 160, 3)
    for i, request in enumerate(requests):
        assert request["state"] == episode["states"][i]
        assert "history" not in request["state"]["observation"]
        observation = request["state"]["observation"]
        assert observation["player"]["row"] == 8 and observation["player"]["column"] == 8.5
        assert {
            key: value for key, value in observation["player"].items() if key.startswith("wall_")
        } == {
            "wall_up": True,
            "wall_right": False,
            "wall_down": True,
            "wall_left": False,
        }
        # Old diagnostics and duplicate coordinates must not leak into actual SDK requests.
        old_fields = {
            "source",
            "layout_id",
            "power_pellet_cells",
            "returning_eyes",
            "x",
            "y",
            "nearest_cell",
            "motion_pixels_per_decision",
            "distance_px",
            "maze_distance_px",
        }

        def check_keys(value):
            if isinstance(value, dict):
                assert not old_fields & value.keys()
                for child in value.values():
                    check_keys(child)
            elif isinstance(value, list):
                for child in value:
                    check_keys(child)

        check_keys(observation)
    assert "offline-test-key" not in (tmp_path / "mock.responses.jsonl").read_text()


def test_cli_default_budget_recording_and_gif(tmp_path):
    pytest.importorskip("ale_py")
    destination = tmp_path / "random.npy"
    assert (
        main(["--env", "ALE/MsPacman-v5", "--policy", "random", "--save-npy", str(destination)])
        == 0
    )
    recording = np.load(destination, allow_pickle=True).item()
    assert recording["metadata"]["frameskip"] == 2
    assert recording["metadata"]["max_steps"] == 500
    assert recording["metadata"]["observation_source"] == "ram"
    assert recording["api_responses"] == []
    episode = recording["episodes"][0]
    assert len(episode["actions"]) <= 500
    assert len(episode["frames"]) == len(episode["actions"]) + 1
    results = convert_recording(destination)
    assert len(results) == 1 and destination.with_suffix(".gif").exists()


@pytest.mark.parametrize(
    "game,expected", [("mspacman", ["MsPacman"]), ("both", ["Pong", "Breakout"])]
)
def test_trial_maps_environment_and_keeps_both_to_two_games(monkeypatch, tmp_path, game, expected):
    import run_jev_trial

    calls = []
    monkeypatch.setenv("TYPESAFE_API_KEY", "offline-test-key")
    monkeypatch.setattr("jev_rl.cli.main", lambda args: calls.append(args) or 0)
    monkeypatch.setattr("npy_to_gif.convert_recording", lambda *args, **kwargs: [])
    assert (
        run_jev_trial.main(["--game", game, "--max-steps", "3", "--output-dir", str(tmp_path)]) == 0
    )
    assert [args[args.index("--env") + 1] for args in calls] == [
        f"ALE/{name}-v5" for name in expected
    ]
    assert all(args[args.index("--max-retries") + 1] == "0" for args in calls)
