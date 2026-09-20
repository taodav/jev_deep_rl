"""A Jev Choice policy and a reproducible uniform-random baseline."""

import math
import random
from dataclasses import dataclass, field
from time import perf_counter
from typing import Protocol

from typesafe_sdk import Choice, TypeSafeClient

from .adapters import Action
from .responses import ResponseCapture


class PolicyError(RuntimeError):
    """Invalid model output; messages intentionally exclude raw service payloads."""

    def __init__(self, message: str, *, diagnostics: dict | None = None) -> None:
        super().__init__(message)
        self.diagnostics = diagnostics


@dataclass(frozen=True)
class Decision:
    action: str
    probabilities: dict[str, float]
    confidence: float | None = None
    latency_ms: float = 0.0
    model: str | None = None
    request_id: str | None = None
    usage: dict[str, int] = field(default_factory=dict)
    raw_probability_sum: float | None = None


class Policy(Protocol):
    def reset(self, seed: int) -> None: ...

    def decide(self, state: dict, actions: tuple[Action, ...]) -> Decision: ...


def action_question(actions: tuple[Action, ...]) -> Choice:
    names = [action.name for action in actions]
    if not 2 <= len(names) <= 255 or len(names) != len(set(names)):
        raise ValueError("Choice requires 2-255 distinct action names.")
    return Choice(
        instructions=(
            "Which single action should be applied for the next environment step "
            "to pursue `goal`, given `observation` and the previous action and reward? "
            "Use `observation.description` when present to interpret the coordinates, "
            "controlled object, motion, and game phase. "
            "Consider current motion as well as position. Select from the supplied actions. "
            "Missing observations are unknown, not zero."
        ),
        criteria={action.name: action.description for action in actions},
    )


class RandomPolicy:
    def __init__(self) -> None:
        self._rng = random.Random()

    def reset(self, seed: int) -> None:
        self._rng.seed(seed)

    def decide(self, state: dict, actions: tuple[Action, ...]) -> Decision:
        return Decision(
            action=self._rng.choice(actions).name,
            probabilities={action.name: 1 / len(actions) for action in actions},
        )


class JevPolicy:
    def __init__(self, client: TypeSafeClient, capture: ResponseCapture | None = None) -> None:
        self._client = client
        self._capture = capture

    def reset(self, seed: int) -> None:
        # Requests are stateless; the simulator seed does not seed the remote model.
        if self._capture is not None:
            self._capture.begin_episode(seed)

    def decide(self, state: dict, actions: tuple[Action, ...]) -> Decision:
        start = perf_counter()
        if self._capture is not None:
            self._capture.begin_decision(state)
        response = self._client.system_one(
            state=state, questions={"action": action_question(actions)}
        )
        latency = (perf_counter() - start) * 1000
        answer = response.choices.get("action")
        names = {action.name for action in actions}
        usage = {key: getattr(response.usage, key) for key in ("input_tokens", "output_tokens")}
        if answer is None or answer.choice not in names:
            raise PolicyError(
                "Jev did not return a legal action choice.",
                diagnostics={
                    "reason": "invalid_choice",
                    "answer_present": answer is not None,
                    "usage": usage,
                    "choice_matches_after_trimming": answer is not None
                    and answer.choice.strip() in names,
                },
            )
        probabilities = dict(answer.probabilities)
        total = sum(probabilities.values())
        # Live replies use two decimal places and can sum to 0.99. Allow the
        # rounding error of each option, capped at three percentage points.
        sum_tolerance = min(0.005 * len(names), 0.03) + 1e-9
        checks = {
            "option_set": set(probabilities) == names,
            "probability_range": all(
                math.isfinite(p) and 0 <= p <= 1 for p in probabilities.values()
            ),
            "probability_sum": math.isfinite(total) and abs(total - 1.0) <= sum_tolerance,
            "confidence_range": math.isfinite(answer.confidence) and 0 <= answer.confidence <= 1,
        }
        if not all(checks.values()):
            raise PolicyError(
                "Jev returned an invalid action probability distribution.",
                diagnostics={
                    "reason": "invalid_distribution",
                    "violations": [name for name, passed in checks.items() if not passed],
                    "choice": answer.choice,  # Already checked against our legal labels.
                    "known_option_probabilities": {
                        name: probabilities[name] if math.isfinite(probabilities[name]) else None
                        for name in sorted(names & set(probabilities))
                    },
                    "missing_options": sorted(names - set(probabilities)),
                    "unexpected_option_count": len(set(probabilities) - names),
                    "invalid_probability_count": sum(
                        not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities.values()
                    ),
                    "probability_sum": total if math.isfinite(total) else None,
                    "confidence": answer.confidence if math.isfinite(answer.confidence) else None,
                    "usage": usage,
                },
            )
        model = response.model
        request_id = response.raw_http_response.headers.get("x-typesafe-request-id")
        if self._capture is not None:
            model = self._capture.redact(model)
            request_id = self._capture.redact(request_id) if request_id is not None else None
        return Decision(
            action=answer.choice,
            probabilities={
                name: probability / total for name, probability in probabilities.items()
            },
            confidence=answer.confidence,
            latency_ms=latency,
            model=model,
            # A proxy or test transport can omit this optional telemetry header.
            request_id=request_id,
            usage=response.usage.model_dump(),
            raw_probability_sum=total,
        )
