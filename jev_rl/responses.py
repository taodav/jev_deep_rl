"""Persist HTTP responses before SDK parsing, with runtime credentials redacted."""

import json
import os
from collections.abc import Callable
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import httpx2


class ResponseCapture:
    def __init__(self, path: Path, on_record: Callable[[dict], None] | None = None) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._stream = path.open("x", encoding="utf-8")
        self._on_record = on_record
        self._episode = -1
        self._seed = None
        self._state_step = None
        self._decision = 0
        self._attempt = 0
        self._secrets: set[str] = set()

    def __enter__(self) -> "ResponseCapture":
        return self

    def __exit__(self, *args) -> None:
        self._stream.close()

    def begin_episode(self, seed: int) -> None:
        self._episode += 1
        self._seed = seed
        self._decision = 0

    def begin_decision(self, state: dict) -> None:
        self._decision += 1
        self._state_step = state.get("step")

    def redact(self, text: str) -> str:
        secrets = self._secrets | {os.environ.get("TYPESAFE_API_KEY", "")}
        secrets.discard("")
        escaped = {json.dumps(value, ensure_ascii=True)[1:-1] for value in secrets}
        for secret in sorted(secrets | escaped, key=len, reverse=True):
            text = text.replace(secret, "[REDACTED]")
        return text

    def record(
        self,
        request: httpx2.Request,
        *,
        response: httpx2.Response | None,
        elapsed_ms: float,
        error_type: str | None = None,
    ) -> None:
        self._attempt += 1
        # Inspect credentials only inside the runtime, solely to remove echoes.
        # Header values are never included in the saved event.
        for name, value in request.headers.items():
            if name.lower() in {
                "authorization",
                "proxy-authorization",
                "cookie",
                "x-api-key",
                "api-key",
            }:
                self._secrets.add(value)
                if name.lower().endswith("authorization") and " " in value:
                    self._secrets.add(value.split(" ", 1)[1])

        # Full text preserves malformed JSON and unknown SDK fields for replay.
        # No request/response headers, cookies, query strings, or exception text.
        event = {
            "event": "http_response" if error_type is None else "transport_error",
            "schema_version": 1,
            "attempt": self._attempt,
            "source": request.extensions.get("jev_response_source", "live"),
            "replay_origin": request.extensions.get("jev_replay_origin"),
            "episode": self._episode if self._episode >= 0 else None,
            "seed": self._seed,
            "decision": self._decision,
            "state_step": self._state_step,
            "captured_at_utc": datetime.now(timezone.utc).isoformat(),
            "status_code": None if response is None else response.status_code,
            "elapsed_ms": elapsed_ms,
            "request_body_text": self.redact(request.content.decode("utf-8", errors="replace")),
            "body_text": None
            if response is None or error_type is not None
            else self.redact(response.text),
            "request_id": None
            if response is None
            else self.redact(response.headers.get("x-typesafe-request-id", "")) or None,
            "error_type": error_type,
        }
        self._stream.write(json.dumps(event, allow_nan=False) + "\n")
        self._stream.flush()
        if self._on_record is not None:
            self._on_record(deepcopy(event))


class CaptureTransport(httpx2.BaseTransport):
    def __init__(self, capture: ResponseCapture, inner: httpx2.BaseTransport | None = None) -> None:
        self.capture = capture
        self.inner = inner if inner is not None else httpx2.HTTPTransport(retries=0)

    def handle_request(self, request: httpx2.Request) -> httpx2.Response:
        started = perf_counter()
        response = None
        try:
            response = self.inner.handle_request(request)
            response.read()
        except Exception as error:
            self.capture.record(
                request,
                response=response,
                elapsed_ms=(perf_counter() - started) * 1000,
                error_type=type(error).__name__,
            )
            if response is not None:
                response.close()
            raise
        # This happens before the SDK can reject an HTTP status, JSON, or schema.
        try:
            self.capture.record(
                request, response=response, elapsed_ms=(perf_counter() - started) * 1000
            )
        except BaseException:
            response.close()
            raise
        return response

    def close(self) -> None:
        self.inner.close()


class ReplayThenLiveTransport(httpx2.BaseTransport):
    """Reuse an exact recorded request prefix, then spend only the remaining budget."""

    def __init__(
        self,
        journal: Path,
        *,
        seed: int,
        max_steps: int,
        inner: httpx2.BaseTransport | None = None,
    ) -> None:
        self.records = [json.loads(line) for line in journal.read_text().splitlines()]
        if not self.records or len(self.records) > max_steps:
            raise ValueError("Replay journal must fit the decision budget.")
        for index, record in enumerate(self.records):
            if not (
                record.get("schema_version") == 1
                and record.get("event") == "http_response"
                and record.get("status_code") == 200
                and record.get("episode") == 0
                and record.get("seed") == seed
                and record.get("state_step") == index
                and record.get("decision") == index + 1
                and isinstance(record.get("body_text"), str)
            ):
                raise ValueError(
                    "Replay requires one complete HTTP 200 response per consecutive decision."
                )
            json.loads(record["request_body_text"])
        self.journal = journal
        self.replayed = 0
        self.live_requests = 0
        self.max_live_requests = max_steps - len(self.records)
        self.inner = inner if inner is not None else httpx2.HTTPTransport(retries=0)

    def handle_request(self, request: httpx2.Request) -> httpx2.Response:
        if self.replayed < len(self.records):
            request.extensions["jev_response_source"] = "replay"
            record = self.records[self.replayed]
            # A changed state, model, or question must stop before any live call.
            if json.loads(request.content) != json.loads(record["request_body_text"]):
                raise ValueError("Replay request differs from its saved state, model, or question.")
            request.extensions["jev_replay_origin"] = {
                "journal": str(self.journal),
                "attempt": record["attempt"],
                "captured_at_utc": record["captured_at_utc"],
            }
            self.replayed += 1
            headers = {"content-type": "application/json"}
            if record.get("request_id"):
                headers["x-typesafe-request-id"] = record["request_id"]
            return httpx2.Response(200, text=record["body_text"], headers=headers)
        request.extensions["jev_response_source"] = "live"
        if self.live_requests >= self.max_live_requests:
            raise ValueError("Live request budget exhausted.")
        self.live_requests += 1
        return self.inner.handle_request(request)

    def close(self) -> None:
        self.inner.close()
