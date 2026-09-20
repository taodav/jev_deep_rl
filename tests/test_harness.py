import io
import json

import gymnasium as gym
import httpx2
import numpy as np
import pytest
from typesafe_sdk import RetryPolicy, TypeSafeClient

from jev_rl.adapters import CartPoleAdapter, PongAdapter, make_environment
from jev_rl.cli import main
from jev_rl.policies import JevPolicy, PolicyError, RandomPolicy
from jev_rl.runner import run_episode


@pytest.mark.parametrize("request_id", [None, "request-test-123"])
def test_jev_sdk_request_and_response_without_network(request_id):
    requests = []

    def respond(request):
        requests.append(json.loads(request.content))
        assert request.url.path == "/v1/systemone"
        headers = {} if request_id is None else {"x-typesafe-request-id": request_id}
        return httpx2.Response(
            200,
            headers=headers,
            json={
                "model": "jev-1.13.0",
                "answers": {
                    "action": {
                        "type": "choice",
                        "choice": "push_right",
                        "confidence": 0.8,
                        "probabilities": {"push_left": 0.1, "push_right": 0.9},
                    }
                },
                "usage": {"input_tokens": 123, "output_tokens": 20},
            },
        )

    env, adapter = make_environment("CartPole-v1")
    try:
        with TypeSafeClient(
            api_key="test-placeholder",
            model="jev-1.13.0",
            transport=httpx2.MockTransport(respond),
            retry=RetryPolicy(max_retries=0),
        ) as client:
            log = io.StringIO()
            result = run_episode(env, adapter, JevPolicy(client), seed=3, max_steps=2, log=log)
        assert len(requests) == 2
        assert requests[0]["model"] == "jev-1.13.0"
        assert requests[0]["questions"]["action"]["type"] == "choice"
        assert set(requests[0]["questions"]["action"]["criteria"]) == {"push_left", "push_right"}
        events = [json.loads(line) for line in log.getvalue().splitlines()]
        transitions = [event for event in events if event["event"] == "transition"]
        assert requests[0]["state"] == transitions[0]["state"]
        assert requests[1]["state"] == transitions[0]["next_state"]
        assert transitions[0]["action_value"] == 1
        assert transitions[0]["decision"]["probabilities"]["push_right"] == 0.9
        assert transitions[0]["decision"]["latency_ms"] >= 0
        assert transitions[0]["decision"]["request_id"] == request_id
        assert result.input_tokens == 246
        assert result.output_tokens == 40
        assert "test-placeholder" not in log.getvalue()
    finally:
        env.close()


@pytest.mark.parametrize(
    "choice, probabilities",
    [
        ("fly", {"push_left": 0.5, "push_right": 0.5}),
        ("push_left", {"push_left": 1.0}),
        ("push_left", {"push_left": 0.2, "push_right": 0.2}),
        ("push_left", {"push_left": 0.5, "push_right": 0.47}),
    ],
)
def test_invalid_model_action_cannot_reach_environment(choice, probabilities):
    def respond(request):
        return httpx2.Response(
            200,
            json={
                "model": "jev-1.13.0",
                "usage": {"input_tokens": 1, "output_tokens": 1},
                "answers": {
                    "action": {
                        "type": "choice",
                        "choice": choice,
                        "confidence": 0.5,
                        "probabilities": probabilities,
                    }
                },
            },
        )

    env, adapter = make_environment("CartPole-v1")
    try:
        with TypeSafeClient(
            api_key="test-placeholder", transport=httpx2.MockTransport(respond)
        ) as client:
            log = io.StringIO()
            with pytest.raises(PolicyError) as failure:
                run_episode(env, adapter, JevPolicy(client), seed=1, max_steps=1, log=log)
        diagnostics = failure.value.diagnostics
        assert diagnostics["usage"] == {"input_tokens": 1, "output_tokens": 1}
        assert diagnostics["reason"] == (
            "invalid_choice" if choice == "fly" else "invalid_distribution"
        )
        if choice != "fly":
            assert diagnostics["known_option_probabilities"] == probabilities
            assert diagnostics["violations"] == (
                ["option_set"] if len(probabilities) == 1 else ["probability_sum"]
            )
        events = [json.loads(line) for line in log.getvalue().splitlines()]
        assert events[-1]["event"] == "policy_error"
        assert events[-1]["diagnostics"] == diagnostics
        assert "test-placeholder" not in log.getvalue()
        assert env._elapsed_steps == 0
    finally:
        env.close()


class CountingEnv(gym.Env):
    def __init__(self, ending):
        self.action_space = gym.spaces.Discrete(2)
        self.ending = ending
        self.steps = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        return np.zeros(4), {}

    def step(self, action):
        assert self.action_space.contains(action)
        self.steps += 1
        return (
            np.zeros(4),
            2.0,
            self.ending == "terminated" and self.steps == 2,
            self.ending == "truncated" and self.steps == 2,
            {},
        )


@pytest.mark.parametrize("ending", ["terminated", "truncated", "budget"])
def test_episode_boundaries_and_reward_accounting(ending):
    env = CountingEnv(ending)
    log = io.StringIO()
    result = run_episode(env, CartPoleAdapter(), RandomPolicy(), seed=7, max_steps=3, log=log)
    assert result.steps == (3 if ending == "budget" else 2)
    assert env.steps == result.steps
    assert result.total_reward == 2 * result.steps
    assert result.terminated == (ending == "terminated")
    assert result.truncated == (ending == "truncated")
    assert result.budget_exhausted == (ending == "budget")
    events = [json.loads(line) for line in log.getvalue().splitlines()]
    assert events[-1]["event"] == "episode_end"
    assert events[1]["state"]["previous_action"] is None


def test_seeded_baseline_replays_actions_and_observations():
    env, adapter = make_environment("CartPole-v1")
    policy = RandomPolicy()
    histories = []
    try:
        for _ in range(2):
            log = io.StringIO()
            run_episode(env, adapter, policy, seed=21, max_steps=25, log=log)
            histories.append(
                [
                    (event["state"], event["action_value"], event["reward"], event["next_state"])
                    for event in map(json.loads, log.getvalue().splitlines())
                    if event["event"] == "transition"
                ]
            )
        assert histories[0] == histories[1]
    finally:
        env.close()


def pong_frame(ball=None):
    frame = np.zeros((210, 160, 3), dtype=np.uint8)
    frame[100:116, 140:144] = [92, 186, 92]
    frame[90:106, 16:20] = [213, 130, 74]
    # HUD and border pixels share object colors but must not become objects.
    frame[5:15, 100:120] = [92, 186, 92]
    frame[0:30, :] = [236, 236, 236]
    if ball is not None:
        x, y = ball
        frame[y : y + 4, x : x + 2] = [236, 236, 236]
    return frame


def test_pong_visible_objects_motion_and_missing_observations():
    adapter = PongAdapter()
    first = adapter.encode(pong_frame((80, 70)))
    assert first["player"] == {"x": 141.5, "y": 107.5, "width": 4, "height": 16}
    assert first["ball"]["x"] == 80.5
    assert first["ball_relative_to_player"] == "above"
    assert first["ball_motion"] is None
    moved = adapter.encode(pong_frame((84, 72)))
    assert moved["ball_motion"]["dx_pixels_per_env_step"] == 4
    assert moved["ball_motion"]["vertical"] == "down"
    missing = adapter.encode(pong_frame())
    assert missing["ball"] is None
    assert missing["ball_motion"] is None
    assert adapter.encode(pong_frame((90, 75)))["ball_motion"] is None
    assert adapter.encode(pong_frame((80, 80)), reward=-1)["ball_motion"] is None
    adapter.reset()
    assert adapter.encode(pong_frame((70, 90)))["ball_motion"] is None


def test_cli_does_not_overwrite_existing_files(tmp_path):
    destination = tmp_path / "existing.jsonl"
    destination.write_text("keep this")
    assert main(["--max-steps", "1", "--log", str(destination)]) == 1
    assert destination.read_text() == "keep this"


def test_api_failure_does_not_print_service_payload(monkeypatch, capsys, tmp_path):
    def respond(request):
        return httpx2.Response(401, json={"detail": "do-not-print-this-payload"})

    def client_factory(**kwargs):
        kwargs["retry"] = RetryPolicy(max_retries=0)
        kwargs["transport"].inner.close()
        kwargs["transport"].inner = httpx2.MockTransport(respond)
        return TypeSafeClient(api_key="test-placeholder", **kwargs)

    monkeypatch.setattr("jev_rl.cli.TypeSafeClient", client_factory)
    assert (
        main(
            [
                "--policy",
                "jev",
                "--max-steps",
                "1",
                "--responses",
                str(tmp_path / "responses.jsonl"),
            ]
        )
        == 1
    )
    output = capsys.readouterr()
    assert "TypeSafe request failed" in output.err
    assert "do-not-print-this-payload" not in output.out + output.err
    assert "test-placeholder" not in output.out + output.err


def test_zero_retries_caps_attempts_and_aborts_on_retryable_error(monkeypatch, tmp_path, capsys):
    attempts = []

    def respond(request):
        attempts.append(request.url.path)
        return httpx2.Response(503, json={"detail": "private-service-detail"})

    def client_factory(**kwargs):
        kwargs["transport"].inner.close()
        kwargs["transport"].inner = httpx2.MockTransport(respond)
        return TypeSafeClient(api_key="test-placeholder", **kwargs)

    monkeypatch.setattr("jev_rl.cli.TypeSafeClient", client_factory)
    log = tmp_path / "no-retries.jsonl"
    assert (
        main(
            [
                "--policy",
                "jev",
                "--max-retries",
                "0",
                "--max-steps",
                "64",
                "--log",
                str(log),
            ]
        )
        == 1
    )
    assert attempts == ["/v1/systemone"]
    events = [json.loads(line) for line in log.read_text().splitlines()]
    assert events[0]["max_http_attempts_per_decision"] == 1
    assert not any(event["event"] == "transition" for event in events)
    captured = capsys.readouterr()
    assert "private-service-detail" not in captured.out + captured.err + log.read_text()


def test_atari_pong_pixels_and_action_directions():
    pytest.importorskip("ale_py")
    env, adapter = make_environment("ALE/Pong-v5", sticky_probability=0.0)
    try:
        actions = {action.name: action.value for action in adapter.actions(env)}
        assert set(actions) == {"stay", "fire", "up", "down", "up_fire", "down_fire"}
        paddle_y = {}
        for name in ("up", "down"):
            env.reset(seed=7)
            adapter.reset()
            for _ in range(3):
                frame, _, _, _, _ = env.step(actions[name])
            state = adapter.encode(frame)
            assert state["player"] is not None
            paddle_y[name] = state["player"]["y"]
        assert paddle_y["up"] < paddle_y["down"]
        result = run_episode(env, adapter, RandomPolicy(), seed=7, max_steps=100)
        assert result.steps == 100
        assert result.budget_exhausted
    finally:
        env.close()
