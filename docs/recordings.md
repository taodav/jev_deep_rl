# Recordings and replay

Use `--save-npy` to save an entire run, including all its episodes, to one new `.npy`
file. CartPole rendering needs the optional `recording` extra (`pygame-ce`); Atari
can render through ALE directly. No video encoders are needed.

```bash
uv sync --extra atari --extra recording

# Decoded RAM drives the policy; renders are saved only for inspection.
# Uses the default 500-step limit, saving up to 501 frames including reset.
uv run --no-env-file --extra atari main.py --env ALE/Pong-v5 --policy random \
  --pong-state ram --save-npy runs/pong-ram-described-500.npy --log runs/pong-ram-described-500.jsonl

# Breakout always uses decoded RAM for policy inputs.
uv run --no-env-file --extra atari main.py --env ALE/Breakout-v5 --policy random \
  --save-npy runs/breakout-ram-random-500.npy --log runs/breakout-ram-random-500.jsonl

uv run --no-env-file --extra recording main.py --env CartPole-v1 --policy random \
  --save-npy runs/cartpole-random.npy
```

The flag also works with `--policy jev`. For live runs, load credentials into the
process environment with `uv` as described above. Existing output files are never
overwritten. Recordings are buffered in memory and written at the end; failures
and interrupts save captured data with `status="incomplete"`, when possible.

Load a recording with NumPy:

```python
import numpy as np

run = np.load("runs/pong-ram-described-500.npy", allow_pickle=True).item()
episode = run["episodes"][0]
print(run["metadata"])  # policy, model, seeds, versions, question, settings
print(episode["frames"].shape)  # (501, 210, 160, 3), RGB uint8 for 500 Pong steps
print(episode["observations"].shape)  # (501, 128), raw uint8 RAM with --pong-state ram
frame = episode["frames"][10]
state = episode["states"][10]  # exact structured state used for decision 10
decision = episode["decisions"][10]  # selected label, probabilities, confidence, timing, usage
next_frame = episode["frames"][11]  # result of actions[10]
```

The `.npy` contains a dictionary, so it uses NumPy's object/pickle format; load only
trusted recordings. `metadata` includes the action mapping, full question, requested
model, dependency versions, frame skip and sticky-action settings. Each episode has:

| Field | Contents and alignment |
| --- | --- |
| `frames` | N+1 RGB frames: reset, then one render after each of N environment steps |
| `observations` | N+1 raw observations; Pong RGB or RAM, Breakout/Ms. Pac-Man/Montezuma RAM, or CartPole's four numeric values |
| `infos` | N+1 original Gymnasium info dictionaries, including reset info |
| `states` | N+1 structured states; `states[t]` is the input to `decisions[t]` |
| `observation_elapsed_s` | N+1 elapsed wall-clock timestamps for the captured observations |
| `actions`, `rewards` | N environment action indices and resulting rewards |
| `terminated`, `truncated` | N environment ending flags |
| `decisions` | N records with labels, probabilities, confidence, latency, model, request ID, token usage |
| `summary` | Episode seed, length, return, ending reason, elapsed time and token totals |

`actions[t]` connects observations/frames `t` and `t+1`; the last observation has no
subsequent decision. Raw observations and renders share storage when identical.
With the harness's default Atari frame skip of 2, these are decision-boundary frames, not every
internal emulator frame. The `infos` retain ALE's episode/global frame numbers.
Early termination saves fewer than 501 frames, as expected.

## Save every API response

Every CLI run with `--policy jev` creates a response journal automatically:
`NAME.responses.jsonl` beside `--log` or `--save-npy`, or a unique file under `runs/`
when neither is given. Override the destination with `--responses NEW.jsonl`.
With `--save-npy`, the same records are also stored in the top-level
`run["api_responses"]` list.

Each HTTP attempt is captured and flushed before SDK parsing and policy validation,
including rejected decisions, HTTP errors, malformed JSON, and retries. Records
include episode/seed, decision number, zero-based `state_step`, global attempt
number, HTTP status, request ID, latency, `request_body_text`, and `body_text`.
The full decoded response text preserves unknown fields and malformed content.
Runtime credentials and their JSON-escaped forms are redacted before anything is
saved. Authorization headers, cookies, other headers, and exception messages are
never stored. A connection or body-read failure records its exception class;
`body_text` is null because a complete response was not available.

```python
import json
import numpy as np

run = np.load("runs/pong-trial/pong-ram-jev-trial-64.npy", allow_pickle=True).item()
response = run["api_responses"][-1]
print(response["state_step"], response["status_code"])
body = json.loads(response["body_text"])  # when the captured body is valid JSON
```

Response records align with HTTP attempts, not environment steps: a rejected reply
can exist without a transition, and retries can yield several replies for one
decision. Existing recordings made before response capture have no such list.

If a run stopped on a validator error, reuse its saved replies after fixing the
validator instead of paying for the same requests again:

```bash
uv run --env-file .env --extra atari --extra recording run_jev_trial.py \
  --game pong --output-dir runs/pong-resumed \
  --replay-responses runs/pong-original/pong-ram-jev-trial-64.responses.jsonl
```

The environment restarts with the same seed and must reproduce every saved request
exactly (state, model, and questions). A mismatch stops the run before any new API
call. The trial spends at most `64 - saved_reply_count` new requests. The source
journal must contain consecutive HTTP 200 replies, one per decision, with no
retries. CLI replay requires one episode and `--max-retries 0`. Captures label
`source` as `replay` or `live` and retain the replay's original journal, attempt,
and timestamp. Episode token totals include reused replies; use `source == "live"`
when counting new usage, and do not count the original and replay twice.
Replay requires the same prompt, state format, model, emulator configuration,
and seed as the source run. Changed requests are rejected before any live call.

## Export recordings as GIFs

`npy_to_gif.py` converts the saved RGB frames into looping GIFs using Pillow from
the `recording` extra. It reads only the recordings you specify; no environment
simulation, API calls, or credentials are needed.

```bash
uv run --no-env-file --extra recording npy_to_gif.py \
  runs/pong-ram-described-500.npy runs/breakout-ram-random-500.npy --scale 2
```

Each GIF is saved beside its `.npy` with the same stem. Multiple episodes produce
separate files named `NAME-episode-0.gif`, `NAME-episode-1.gif`, etc. Existing GIFs
are never overwritten; use `--output-dir` to export another set.

Atari playback uses recorded emulator frame counters and assumes the standard
60 Hz NTSC games: with the current frame skip of 2, this is about 30 saved frames
per second. Existing recordings retain their recorded timing, including older
trials that used frame skip 4 (about 15 saved frames per second).
Other environments use recorded `render_fps`. Playback follows game time, excluding
API latency, and holds the final frame for one additional nominal interval. GIF
delays round to 10 ms increments without cumulative drift. Use `--fps 10` to
override playback speed, or `--scale 2` for crisp 2x pixel enlargement (default 1x).
Only load trusted `.npy` recordings, since this recording format uses pickle.


## Combine GIFs

Join GIFs in left-to-right order, preserving source playback timing and stopping
when the shortest clip ends:

```bash
uv run --no-env-file --extra recording join_gifs.py \
  runs/pong.gif runs/breakout.gif runs/mspacman.gif \
  --output runs/combined.gif
```

Use `--crop-bottom 28` to remove a known 28-pixel bottom border from the combined
canvas. Cropping is explicit; choose a value appropriate to the source frames.
Inputs are kept at their original sizes, aligned at the top, and padded below
shorter inputs. Existing output files are never overwritten. Encoding buffers
frames in memory, so long or large animations can consume substantial RAM.

## Inspect Pong offline

```bash
uv run --no-env-file --extra atari debug_pong.py runs/pong.npy --report runs/pong-debug.json
```

The debugger replays the recorded seed, frame skip, sticky actions, and selected
actions. It compares observations, rendered frames, decoded geometry, rewards,
and ending flags. Saved decisions pass through the SDK and validator using a
mock HTTP transport. No live API calls occur. Use `--episode INDEX` to select an
episode. The current adapter must match the recording's observation format.
