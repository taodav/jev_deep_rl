"""Run the harness with credential-safe logging and error handling."""

import sys

from jev_rl.runtime import configure_runtime

configure_runtime()

# SDK logging configuration is read at import time.
from jev_rl.cli import main  # noqa: E402

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        print("Run stopped unexpectedly. Raw error details are suppressed.", file=sys.stderr)
        raise SystemExit(1) from None
