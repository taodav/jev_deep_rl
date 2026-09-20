"""NumPy run recordings with aligned observations and transitions."""

from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from .policies import Decision, PolicyError


class EpisodeRecording:
    """N actions connect N+1 observations, including reset and the final frame."""

    def __init__(self, episode: int, seed: int) -> None:
        self.episode = episode
        self.seed = seed
        self.summary: dict | None = None
        self._started = perf_counter()
        self._snapshots: list[dict] = []
        self._transitions: list[dict] = []

    def _snapshot(self, frame: np.ndarray, observation: Any, info: dict, state: dict) -> dict:
        frame = np.array(frame, copy=True)
        if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
            raise ValueError("Recording requires an RGB uint8 render frame.")
        # Pong's raw observation is the rendered frame. Preserve both field names
        # but share the array when equal so the .npy doesn't store the pixels twice.
        same_pixels = (
            isinstance(observation, np.ndarray)
            and observation.dtype == frame.dtype
            and np.array_equal(observation, frame)
        )
        return {
            "frame": frame,
            "observation": frame if same_pixels else deepcopy(observation),
            "info": deepcopy(info),
            "state": deepcopy(state),
            "elapsed_s": perf_counter() - self._started,
        }

    def initial(self, frame: np.ndarray, observation: Any, info: dict, state: dict) -> None:
        if self._snapshots:
            raise ValueError("The initial observation can only be recorded once.")
        self._snapshots.append(self._snapshot(frame, observation, info, state))

    def step(
        self,
        frame: np.ndarray,
        observation: Any,
        info: dict,
        state: dict,
        *,
        decision: Decision,
        action: int,
        reward: float,
        terminated: bool,
        truncated: bool,
    ) -> None:
        if not self._snapshots:
            raise ValueError("Record the initial observation before transitions.")
        snapshot = self._snapshot(frame, observation, info, state)
        transition = {
            "decision": asdict(decision),
            "action": action,
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
        }
        self._snapshots.append(snapshot)
        self._transitions.append(transition)

    def to_dict(self) -> dict:
        snapshots = self._snapshots
        frames = (
            np.stack([item["frame"] for item in snapshots])
            if snapshots
            else np.empty((0, 0, 0, 3), dtype=np.uint8)
        )
        if snapshots and all(item["observation"] is item["frame"] for item in snapshots):
            observations = frames
        elif snapshots:
            observations = np.stack([item["observation"] for item in snapshots])
        else:
            observations = np.empty(0)
        transitions = self._transitions
        return {
            "episode": self.episode,
            "seed": self.seed,
            "summary": self.summary,
            "frames": frames,
            "observations": observations,
            "infos": [item["info"] for item in snapshots],
            "states": [item["state"] for item in snapshots],
            "observation_elapsed_s": np.array([item["elapsed_s"] for item in snapshots]),
            "actions": np.array([item["action"] for item in transitions], dtype=np.int64),
            "rewards": np.array([item["reward"] for item in transitions], dtype=np.float64),
            "terminated": np.array([item["terminated"] for item in transitions], dtype=np.bool_),
            "truncated": np.array([item["truncated"] for item in transitions], dtype=np.bool_),
            "decisions": [item["decision"] for item in transitions],
        }


class NpyRecorder:
    """Reserve a new file and save completed or partial runs when leaving context."""

    def __init__(self, path: Path) -> None:
        if path.suffix.lower() != ".npy":
            raise ValueError("Recording filenames must end in .npy.")
        path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = path.open("xb")
        self.metadata: dict = {}
        self.episodes: list[EpisodeRecording] = []
        self.api_responses: list[dict] = []
        self._created_at = datetime.now(timezone.utc).isoformat()

    def __enter__(self) -> "NpyRecorder":
        return self

    def start_episode(self, episode: int, seed: int) -> EpisodeRecording:
        recording = EpisodeRecording(episode, seed)
        self.episodes.append(recording)
        return recording

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        try:
            payload = {
                "format": "jev-rl-recording",
                "schema_version": 1,
                "created_at_utc": self._created_at,
                "status": "complete" if exc_type is None else "incomplete",
                # Exception text can contain HTTP payloads; retain only its class name.
                "error_type": None if exc_type is None else exc_type.__name__,
                "error_details": exc_value.diagnostics
                if isinstance(exc_value, PolicyError)
                else None,
                "metadata": self.metadata,
                "api_responses": self.api_responses,
                "episodes": [episode.to_dict() for episode in self.episodes],
            }
            np.save(self._stream, payload, allow_pickle=True)
        finally:
            self._stream.close()
