# Contributing

Use Python 3.14 and the committed `uv.lock`:

```bash
uv sync --locked --extra atari --extra recording
uv run --locked --no-env-file ruff check .
uv run --locked --no-env-file ruff format --check .
uv run --locked --no-env-file --extra atari --extra recording python -m pytest -q
```

Format changes with `uv run --no-env-file ruff format .`. Tests use local
environments and mock HTTP transports; they do not require an API key or spend
TypeSafe credits. GitHub Actions runs the same checks on Linux. For local
headless rendering, set `SDL_VIDEODRIVER=dummy` and `SDL_AUDIODRIVER=dummy`.

## Changes to policies and observations

Keep model prompts, observation semantics, action mappings, and frame skipping
explicit. They affect experiment comparability. Preserve the source recordings
when making changes; response replay requires exactly matching requests.

To add an environment:

1. Implement `ObservationAdapter`: `goal`, `actions`, `reset`, and `encode`.
2. Register it in `jev_rl.adapters.make_environment` and the CLI choices.
3. Verify decoded state and controls against actual emulator behavior.
4. Add an offline integration test using the SDK mock transport and recording path.
5. Document approximations, unknown fields, and any privileged RAM information.

The installed [TypeSafe skill](.agents/skills/typesafe-ai/SKILL.md) and
[AGENTS.md](AGENTS.md) guide agent-assisted changes. Consult the current official
TypeSafe documentation before changing the API integration.

## Credentials and artifacts

Keep credentials in the ignored `.env` or process environment. Never include
keys, authorization headers, cookies, raw exception payloads, or debug HTTP logs
in commits or issues. Do not inspect `.env` through an agent; runtime loaders may
read it for explicitly requested API runs.

Keep experiment outputs under `runs/`, which is ignored. Recordings include raw
RAM, RGB frames, exact prompts, and redacted API responses. Review any selected
artifact before sharing it. Only load trusted `.npy` files: the recording format
contains a pickled dictionary.
