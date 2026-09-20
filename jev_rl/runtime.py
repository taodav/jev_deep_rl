"""Credential-safe runtime setup for command-line entry points."""

import logging
import os


def configure_runtime() -> None:
    """Apply before SDK imports; never open a credential file or print its value."""
    os.environ["TYPESAFE_LOG_LEVEL"] = "off"
    os.environ["TYPESAFE_BASE_URL"] = "https://api.typesafe.ai"
    logging.disable(logging.CRITICAL)
