"""Logging configuration (container-friendly)

Defaults:
- Logs to stdout (so Docker can collect logs).
- Optional rotating file log when LOG_TO_FILE is enabled.

Env vars (override config dict):
  LOG_LEVEL=INFO|DEBUG|WARNING|ERROR
  LOG_TO_FILE=0|1|true|false
  LOG_DIR=/app/logs
  LOG_FILENAME=agentic_ai.log
  LOG_JSON=0|1  (when 1, emits structured JSON lines)
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, Any
from logging.handlers import RotatingFileHandler


def _to_bool(val: Any, default: bool = False) -> bool:
    if val is None:
        return default
    if isinstance(val, bool):
        return val
    s = str(val).strip().lower()
    return s in ("1", "true", "yes", "y", "on")


def setup_logging(config: Dict[str, Any] | None = None) -> logging.Logger:
    """
    Idempotent logging setup safe for containers:
    - Always attaches a StreamHandler to stdout.
    - Optionally adds a RotatingFileHandler if LOG_TO_FILE is enabled.
    - Avoids duplicate handlers on repeated calls.
    """
    cfg = config or {}

    # Resolve level (env overrides config)
    level_name = str(os.getenv("LOG_LEVEL", cfg.get("level", "INFO"))).upper()
    level = getattr(logging, level_name, logging.INFO)

    # Formatting: JSON or plain text
    json_enabled = _to_bool(os.getenv("LOG_JSON", cfg.get("json", False)))
    if json_enabled:
        fmt = (
            '{"t":"%(asctime)s","lvl":"%(levelname)s","name":"%(name)s",'
            '"msg":"%(message)s"}'
        )
        datefmt = "%Y-%m-%dT%H:%M:%S%z"
    else:
        fmt = "%(asctime)s %(levelname)s %(name)s: %(message)s"
        datefmt = "%Y-%m-%d %H:%M:%S"

    formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)

    # Root logger
    root = logging.getLogger()
    root.setLevel(level)

    # Prevent duplicate handlers if setup_logging is called more than once
    # Remove existing handlers but keep their level/filters out of the way
    for h in list(root.handlers):
        root.removeHandler(h)
        try:
            h.close()
        except Exception:
            pass

    # Always add stdout handler
    sh = logging.StreamHandler()
    sh.setLevel(level)
    sh.setFormatter(formatter)
    root.addHandler(sh)

    # Optional rotating file log
    log_to_file = _to_bool(os.getenv("LOG_TO_FILE", cfg.get("log_to_file", False)))
    if log_to_file:
        # Determine path from env first, then config
        log_dir = os.getenv("LOG_DIR", cfg.get("dir")) or os.path.dirname(
            cfg.get("file", "./logs/agentic_ai.log")
        )
        log_filename = os.getenv("LOG_FILENAME", cfg.get("filename", "agentic_ai.log"))
        if not log_dir:
            log_dir = "./logs"

        logfile = os.path.join(log_dir, log_filename)

        try:
            Path(log_dir).mkdir(parents=True, exist_ok=True)
            fh = RotatingFileHandler(
                filename=logfile,
                maxBytes=10 * 1024 * 1024,  # 10 MB
                backupCount=5,
                encoding="utf-8",
                delay=True,  # don't open file until first emit
            )
            fh.setLevel(level)
            fh.setFormatter(formatter)
            root.addHandler(fh)
        except PermissionError:
            root.warning(
                "LOG_TO_FILE is enabled but no permission to write %s. "
                "Continuing without file logging.",
                logfile,
            )
        except OSError as e:
            root.warning(
                "LOG_TO_FILE is enabled but failed to create/open %s (%s). "
                "Continuing without file logging.",
                logfile,
                e,
            )

    # Capture warnings as logs
    logging.captureWarnings(True)

    return root