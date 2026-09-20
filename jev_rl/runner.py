"""Synchronous evaluation: the simulator pauses while a policy is deciding."""

import json
from dataclasses import asdict, dataclass
from time import perf_counter
from typing import TextIO

import gymnasium as gym

from .adapters import ObservationAdapter
from .policies import Policy, PolicyError
from .recording import EpisodeRecording

DEFAULT_MAX_STEPS = 500


def write_event(stream: TextIO | None, event: dict) -> None:
    if stream is not None:
        stream.write(json.dumps(event, allow_nan=False) + "\n")
        stream.flush()


@dataclass(frozen=True)
class EpisodeResult:
    episode: int
    seed: int
    steps: int
    total_reward: float
    terminated: bool
    truncated: bool
    budget_exhausted: bool
    elapsed_s: float
    mean_decision_latency_ms: float
    input_tokens: int
    output_tokens: int


def run_episode(
    env: gym.Env,
    adapter: ObservationAdapter,
    policy: Policy,
    *,
    seed: int,
    max_steps: int = DEFAULT_MAX_STEPS,
    episode: int = 0,
    log: TextIO | None = None,
    recording: EpisodeRecording | None = None,
) -> EpisodeResult:
    if max_steps < 1:
        raise ValueError("max_steps must be positive.")
    actions = adapter.actions(env)
    by_name = {action.name: action.value for action in actions}
    if len(by_name) != len(actions) or any(
        not env.action_space.contains(action.value) for action in actions
    ):
        raise ValueError("Adapter actions must have distinct names and legal values.")
    adapter.reset()
    policy.reset(seed)
    observation, info = env.reset(seed=seed)
    env.action_space.seed(seed)
    env_id = env.spec.id if env.spec is not None else type(env).__name__
    state = {
        "environment": env_id,
        "goal": adapter.goal,
        "step": 0,
        "observation": adapter.encode(observation),
        "previous_action": None,
        "previous_reward": None,
    }
    write_event(log, {"event": "episode_start", "episode": episode, "seed": seed})
    if recording is not None:
        recording.initial(env.render(), observation, info, state)
    started = perf_counter()
    total_reward = total_latency = 0.0
    input_tokens = output_tokens = 0
    terminated = truncated = False
    for step in range(1, max_steps + 1):
        try:
            decision = policy.decide(state, actions)
        except PolicyError as error:
            write_event(
                log,
                {
                    "event": "policy_error",
                    "episode": episode,
                    "step": step,
                    "diagnostics": error.diagnostics,
                },
            )
            raise
        if decision.action not in by_name:
            raise PolicyError("Policy chose an action outside the adapter's legal set.")
        action_value = by_name[decision.action]
        observation, reward, terminated, truncated, next_info = env.step(action_value)
        reward = float(reward)
        terminated, truncated = bool(terminated), bool(truncated)
        next_state = {
            "environment": env_id,
            "goal": adapter.goal,
            "step": step,
            "observation": adapter.encode(observation, reward=reward),
            "previous_action": decision.action,
            "previous_reward": reward,
        }
        total_reward += reward
        total_latency += decision.latency_ms
        input_tokens += decision.usage.get("input_tokens", 0)
        output_tokens += decision.usage.get("output_tokens", 0)
        if recording is not None:
            recording.step(
                env.render(),
                observation,
                next_info,
                next_state,
                decision=decision,
                action=action_value,
                reward=reward,
                terminated=terminated,
                truncated=truncated,
            )
        write_event(
            log,
            {
                "event": "transition",
                "episode": episode,
                "step": step,
                "state": state,
                "decision": asdict(decision),
                "action_value": action_value,
                "reward": reward,
                "next_state": next_state,
                "terminated": terminated,
                "truncated": truncated,
            },
        )
        state = next_state
        info = next_info
        if terminated or truncated:
            break
    result = EpisodeResult(
        episode=episode,
        seed=seed,
        steps=step,
        total_reward=total_reward,
        terminated=terminated,
        truncated=truncated,
        budget_exhausted=not (terminated or truncated),
        elapsed_s=perf_counter() - started,
        mean_decision_latency_ms=total_latency / step,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )
    write_event(log, {"event": "episode_end", **asdict(result)})
    if recording is not None:
        recording.summary = asdict(result)
    return result
