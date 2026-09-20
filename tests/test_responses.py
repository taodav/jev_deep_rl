import json

import httpx2
import numpy as np
import pytest
from typesafe_sdk import RetryPolicy, TypeSafeClient, TypeSafeError

from jev_rl.adapters import Action
from jev_rl.cli import main
from jev_rl.policies import JevPolicy, PolicyError
from jev_rl.responses import CaptureTransport, ReplayThenLiveTransport, ResponseCapture

ACTIONS = (Action("left", 0, "Move left."), Action("right", 1, "Move right."))
KEY = "capture-test-secret"


def answer(probabilities=None):
    return {
        "model": "jev-1.13.0",
        "usage": {"input_tokens": 123, "output_tokens": 20},
        "answers": {
            "action": {
                "type": "choice",
                "choice": "left",
                "confidence": 0.5,
                "probabilities": probabilities or {"left": 0.75, "right": 0.25},
            }
        },
        "future_server_field": {"echo": KEY, "detail": "retain this extra field"},
    }


@pytest.mark.parametrize(
    "kind", ["valid", "policy_rejected", "malformed_json", "bad_schema", "http_error"]
)
def test_every_http_response_is_saved_before_parsing_or_validation(tmp_path, kind):
    body = json.dumps(answer({"left": 0.5, "right": 0.47} if kind == "policy_rejected" else None))
    if kind == "malformed_json":
        body = '{"unfinished":'
    elif kind == "bad_schema":
        body = '{"unexpected": "shape"}'
    elif kind == "http_error":
        body = f"Upstream unavailable; echoed credential: {KEY}"
    status = 503 if kind == "http_error" else 200

    def respond(request):
        return httpx2.Response(
            status,
            text=body,
            headers={
                "x-typesafe-request-id": "request-123",
                "set-cookie": "private-cookie",
                "authorization": f"Bearer {KEY}",
            },
        )

    path = tmp_path / "responses.jsonl"
    copied = []
    with ResponseCapture(path, copied.append) as capture:
        with TypeSafeClient(
            api_key=KEY,
            transport=CaptureTransport(capture, httpx2.MockTransport(respond)),
            retry=RetryPolicy(max_retries=0),
        ) as client:
            policy = JevPolicy(client, capture)
            policy.reset(7)
            if kind == "valid":
                assert policy.decide({"step": 4, "test_echo": KEY}, ACTIONS).action == "left"
            else:
                with pytest.raises((PolicyError, TypeSafeError)):
                    policy.decide({"step": 4, "test_echo": KEY}, ACTIONS)
            # The journal must already be durable, before capture's context closes.
            records = [json.loads(line) for line in path.read_text().splitlines()]
            assert len(records) == 1
            event = records[0]
            assert event["status_code"] == status
            assert event["body_text"] == body.replace(KEY, "[REDACTED]")
            assert event["episode"] == 0 and event["state_step"] == 4 and event["decision"] == 1
            assert event["request_id"] == "request-123"
            assert KEY not in path.read_text()
            assert "private-cookie" not in path.read_text()
            assert "authorization" not in event
            assert copied == records


def test_retry_attempts_are_preserved_under_the_same_decision(tmp_path):
    attempts = []

    def respond(request):
        attempts.append(1)
        if len(attempts) == 1:
            return httpx2.Response(503, text="try later")
        return httpx2.Response(200, json=answer())

    path = tmp_path / "retry.jsonl"
    with (
        ResponseCapture(path) as capture,
        TypeSafeClient(
            api_key=KEY,
            transport=CaptureTransport(capture, httpx2.MockTransport(respond)),
            retry=RetryPolicy(max_retries=1, backoff_initial=0, backoff_max=0, backoff_jitter=0),
        ) as client,
    ):
        policy = JevPolicy(client, capture)
        policy.reset(7)
        policy.decide({"step": 0}, ACTIONS)
    events = [json.loads(line) for line in path.read_text().splitlines()]
    assert [event["status_code"] for event in events] == [503, 200]
    assert [event["attempt"] for event in events] == [1, 2]
    assert [event["decision"] for event in events] == [1, 1]


def test_credential_echoes_in_response_telemetry_are_also_redacted(tmp_path):
    def respond(request):
        body = answer()
        body["model"] = KEY
        return httpx2.Response(200, json=body, headers={"x-typesafe-request-id": KEY})

    path = tmp_path / "echo.jsonl"
    with (
        ResponseCapture(path) as capture,
        TypeSafeClient(
            api_key=KEY,
            transport=CaptureTransport(capture, httpx2.MockTransport(respond)),
            retry=RetryPolicy(max_retries=0),
        ) as client,
    ):
        decision = JevPolicy(client, capture).decide({"step": 0}, ACTIONS)
    assert decision.model == decision.request_id == "[REDACTED]"
    assert KEY not in path.read_text()


def test_transport_failure_saves_no_exception_text(tmp_path):
    def respond(request):
        raise httpx2.ConnectError(f"secret in exception: {KEY}", request=request)

    path = tmp_path / "connection.jsonl"
    with (
        ResponseCapture(path) as capture,
        TypeSafeClient(
            api_key=KEY,
            transport=CaptureTransport(capture, httpx2.MockTransport(respond)),
            retry=RetryPolicy(max_retries=0),
        ) as client,
    ):
        with pytest.raises(TypeSafeError):
            JevPolicy(client, capture).decide({"step": 0}, ACTIONS)
    event = json.loads(path.read_text())
    assert event["event"] == "transport_error"
    assert event["status_code"] is event["body_text"] is None
    assert event["error_type"] == "ConnectError"
    assert KEY not in path.read_text() and "secret in exception" not in path.read_text()


def test_failed_response_is_in_partial_npy_and_default_journal(monkeypatch, tmp_path, capsys):
    def respond(request):
        # This is outside CartPole's allowed action labels; retain the full reply.
        return httpx2.Response(200, json=answer())

    def client_factory(**kwargs):
        kwargs["transport"].inner.close()
        kwargs["transport"].inner = httpx2.MockTransport(respond)
        return TypeSafeClient(api_key=KEY, **kwargs)

    monkeypatch.setattr("jev_rl.cli.TypeSafeClient", client_factory)
    path = tmp_path / "failed.npy"
    assert (
        main(["--policy", "jev", "--max-steps", "1", "--max-retries", "0", "--save-npy", str(path)])
        == 1
    )
    run = np.load(path, allow_pickle=True).item()
    journal = path.with_suffix(".responses.jsonl")
    events = [json.loads(line) for line in journal.read_text().splitlines()]
    assert run["api_responses"] == events
    assert run["status"] == "incomplete" and run["error_type"] == "PolicyError"
    assert len(run["episodes"][0]["decisions"]) == 0
    assert len(events) == 1
    assert (
        json.loads(events[0]["body_text"])["future_server_field"]["detail"]
        == "retain this extra field"
    )
    assert KEY.encode() not in path.read_bytes()
    assert KEY not in journal.read_text()
    output = capsys.readouterr()
    assert KEY not in output.out + output.err


@pytest.mark.parametrize(
    "probabilities",
    [
        {"left": 0.5, "right": 0.49},
        {"left": 0.51, "right": 0.5},
    ],
)
def test_rounded_probabilities_are_normalized_without_changing_raw_reply(tmp_path, probabilities):
    body = answer(probabilities)
    path = tmp_path / "rounded.jsonl"
    with (
        ResponseCapture(path) as capture,
        TypeSafeClient(
            api_key=KEY,
            transport=CaptureTransport(
                capture, httpx2.MockTransport(lambda request: httpx2.Response(200, json=body))
            ),
            retry=RetryPolicy(max_retries=0),
        ) as client,
    ):
        decision = JevPolicy(client, capture).decide({"step": 0}, ACTIONS)
    assert decision.action == "left"
    assert sum(decision.probabilities.values()) == pytest.approx(1)
    assert decision.raw_probability_sum == pytest.approx(sum(probabilities.values()))
    event = json.loads(path.read_text())
    assert json.loads(event["body_text"])["answers"]["action"]["probabilities"] == probabilities


def make_journal(path):
    with (
        ResponseCapture(path) as capture,
        TypeSafeClient(
            api_key=KEY,
            transport=CaptureTransport(
                capture, httpx2.MockTransport(lambda request: httpx2.Response(200, json=answer()))
            ),
            retry=RetryPolicy(max_retries=0),
        ) as client,
    ):
        policy = JevPolicy(client, capture)
        policy.reset(7)
        policy.decide({"step": 0}, ACTIONS)


def test_replay_reuses_exact_prefix_and_caps_new_requests(tmp_path):
    original = tmp_path / "original.jsonl"
    make_journal(original)
    calls = []

    def respond(request):
        calls.append(json.loads(request.content))
        return httpx2.Response(200, json=answer())

    replay = ReplayThenLiveTransport(
        original, seed=7, max_steps=2, inner=httpx2.MockTransport(respond)
    )
    output = tmp_path / "resumed.jsonl"
    with (
        ResponseCapture(output) as capture,
        TypeSafeClient(
            api_key=KEY,
            transport=CaptureTransport(capture, replay),
            retry=RetryPolicy(max_retries=0),
        ) as client,
    ):
        policy = JevPolicy(client, capture)
        policy.reset(7)
        assert policy.decide({"step": 0}, ACTIONS).action == "left"
        assert not calls
        assert policy.decide({"step": 1}, ACTIONS).action == "left"
        assert len(calls) == 1
        with pytest.raises((ValueError, TypeSafeError)):
            policy.decide({"step": 2}, ACTIONS)
        assert len(calls) == 1
    events = [json.loads(line) for line in output.read_text().splitlines()]
    assert [event["source"] for event in events[:2]] == ["replay", "live"]
    assert events[0]["replay_origin"]["journal"] == str(original)
    assert events[0]["body_text"] == json.loads(original.read_text())["body_text"]
    assert KEY not in output.read_text()


def test_replay_mismatch_cannot_fall_back_to_live_api(tmp_path):
    original = tmp_path / "original.jsonl"
    make_journal(original)
    calls = []
    replay = ReplayThenLiveTransport(
        original,
        seed=7,
        max_steps=2,
        inner=httpx2.MockTransport(lambda request: calls.append(request)),
    )
    with TypeSafeClient(api_key=KEY, transport=replay, retry=RetryPolicy(max_retries=0)) as client:
        with pytest.raises((ValueError, TypeSafeError)):
            JevPolicy(client).decide({"step": 0, "changed": True}, ACTIONS)
    assert not calls and replay.live_requests == replay.replayed == 0


def test_replay_rejects_retry_journals_and_wrong_seed(tmp_path):
    original = tmp_path / "original.jsonl"
    make_journal(original)
    with pytest.raises(ValueError):
        ReplayThenLiveTransport(original, seed=8, max_steps=2)
    original.write_text(original.read_text() * 2)
    with pytest.raises(ValueError):
        ReplayThenLiveTransport(original, seed=7, max_steps=2)
