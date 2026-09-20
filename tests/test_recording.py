from dataclasses import asdict

import gymnasium as gym
import numpy as np
import pytest

from jev_rl.adapters import CartPoleAdapter, make_environment
from jev_rl.cli import main
from jev_rl.policies import RandomPolicy
from jev_rl.recording import NpyRecorder
from jev_rl.runner import run_episode


class ReusingBuffersEnv(gym.Env):
    """An environment that deliberately mutates previous observation/info buffers."""

    render_mode = "rgb_array"

    def __init__(self, ending):
        self.action_space = gym.spaces.Discrete(2)
        self.ending = ending
        self.observation = np.zeros(4, dtype=np.float32)
        self.frame = np.zeros((8, 10, 3), dtype=np.uint8)
        self.info = {"nested": {"value": np.zeros(1, dtype=np.int32)}}
        self.steps = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        self.observation.fill(0)
        self.info["nested"]["value"].fill(0)
        return self.observation, self.info

    def step(self, action):
        self.steps += 1
        self.observation.fill(self.steps)
        self.info["nested"]["value"].fill(self.steps)
        return (
            self.observation,
            float(action),
            self.ending == "terminated" and self.steps == 2,
            self.ending == "truncated" and self.steps == 2,
            self.info,
        )

    def render(self):
        self.frame.fill(self.steps)
        return self.frame


@pytest.mark.parametrize("ending", ["terminated", "truncated", "budget"])
def test_recording_preserves_every_step_and_copies_reused_buffers(tmp_path, ending):
    path = tmp_path / "run.npy"
    env = ReusingBuffersEnv(ending)
    with NpyRecorder(path) as recorder:
        recorder.metadata = {"test_run": True}
        summaries = []
        for episode in range(2):
            summaries.append(
                run_episode(
                    env,
                    CartPoleAdapter(),
                    RandomPolicy(),
                    seed=7 + episode,
                    max_steps=3,
                    episode=episode,
                    recording=recorder.start_episode(episode, 7 + episode),
                )
            )
    run = np.load(path, allow_pickle=True).item()
    assert run["status"] == "complete"
    assert run["metadata"] == {"test_run": True}
    assert len(run["episodes"]) == 2
    for episode, summary in zip(run["episodes"], summaries):
        steps = summary.steps
        assert episode["summary"] == asdict(summary)
        assert episode["frames"].shape == (steps + 1, 8, 10, 3)
        assert episode["observations"].shape == (steps + 1, 4)
        assert len(episode["infos"]) == len(episode["states"]) == steps + 1
        assert len(episode["decisions"]) == len(episode["actions"]) == steps
        assert np.array_equal(episode["rewards"], episode["actions"])
        assert episode["terminated"][-1] == (ending == "terminated")
        assert episode["truncated"][-1] == (ending == "truncated")
        assert np.all(np.diff(episode["observation_elapsed_s"]) >= 0)
        for t in range(steps + 1):
            assert np.all(episode["frames"][t] == t)
            assert np.all(episode["observations"][t] == t)
            assert episode["infos"][t]["nested"]["value"][0] == t
            assert episode["states"][t]["step"] == t
            if t < steps:
                assert (
                    episode["states"][t + 1]["previous_action"] == episode["decisions"][t]["action"]
                )


def test_incomplete_recording_retains_data_without_exception_payload(tmp_path):
    class FailingPolicy(RandomPolicy):
        def decide(self, state, actions):
            if state["step"] == 1:
                raise RuntimeError("private-error-payload")
            return super().decide(state, actions)

    path = tmp_path / "partial.npy"
    with pytest.raises(RuntimeError):
        with NpyRecorder(path) as recorder:
            run_episode(
                ReusingBuffersEnv("budget"),
                CartPoleAdapter(),
                FailingPolicy(),
                seed=7,
                max_steps=3,
                recording=recorder.start_episode(0, 7),
            )
    run = np.load(path, allow_pickle=True).item()
    assert run["status"] == "incomplete"
    assert run["error_type"] == "RuntimeError"
    assert run["episodes"][0]["frames"].shape[0] == 2
    assert len(run["episodes"][0]["decisions"]) == 1
    assert run["episodes"][0]["summary"] is None
    assert b"private-error-payload" not in path.read_bytes()


def test_recording_destination_is_never_overwritten(tmp_path):
    path = tmp_path / "existing.npy"
    path.write_bytes(b"keep this")
    assert main(["--save-npy", str(path)]) == 1
    assert path.read_bytes() == b"keep this"


@pytest.mark.parametrize("pong_state", ["rgb", "ram"])
def test_pong_recording_matches_observation_source_and_unrecorded_run(tmp_path, pong_state):
    pytest.importorskip("ale_py")
    path = tmp_path / "pong.npy"
    env, adapter = make_environment("ALE/Pong-v5", render_mode="rgb_array", pong_state=pong_state)
    try:
        expected = run_episode(env, adapter, RandomPolicy(), seed=7, max_steps=20)
        with NpyRecorder(path) as recorder:
            result = run_episode(
                env,
                adapter,
                RandomPolicy(),
                seed=7,
                max_steps=20,
                recording=recorder.start_episode(0, 7),
            )
    finally:
        env.close()
    episode = np.load(path, allow_pickle=True).item()["episodes"][0]
    assert result.total_reward == expected.total_reward
    assert episode["frames"].shape == (21, 210, 160, 3)
    assert episode["frames"].dtype == np.uint8
    if pong_state == "rgb":
        assert episode["frames"] is episode["observations"]
    else:
        assert episode["observations"].shape == (21, 128)
        assert episode["observations"].dtype == np.uint8
        assert episode["states"][0]["observation"]["source"] == "decoded_atari_ram"
    assert all("episode_frame_number" in info for info in episode["infos"])


def test_cartpole_recording_has_separate_rgb_frames_and_numeric_observations(tmp_path):
    pytest.importorskip("pygame")
    path = tmp_path / "cartpole.npy"
    assert main(["--max-steps", "2", "--save-npy", str(path)]) == 0
    run = np.load(path, allow_pickle=True).item()
    episode = run["episodes"][0]
    assert episode["frames"].shape == (3, 400, 600, 3)
    assert episode["observations"].shape == (3, 4)
    assert episode["observations"].dtype == np.float32
    assert run["metadata"]["policy"] == "random"
    assert run["metadata"]["questions"]["action"]["type"] == "choice"
