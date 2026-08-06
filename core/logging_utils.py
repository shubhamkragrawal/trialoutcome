"""Domain-agnostic structured (JSON) logging setup for TrialOutcome's batch/
monitoring entrypoints (M9-17) -- drift_job.py, register_model.py,
retrain_trigger.py, and dataset_builder.py all configure logging through
this module instead of each defining their own formatter. No pharma-specific
strings or column names belong here -- see those four files for the concrete
call sites and log messages.
"""

from __future__ import annotations

import json
import logging
import sys


class JsonFormatter(logging.Formatter):
    """
    Purpose: Render each log record as one JSON object per line -- the
        {ts, level, logger, msg} shape a log aggregator (or a human running
        `make drift | jq`) can parse without a custom grok pattern.
    Leakage guard: N/A.
    Failure mode: N/A -- json.dumps over plain str/bool/int fields never
        raises for the record attributes this formatter reads.
    """

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload)


class _StdoutProxy:
    """
    Purpose: A stream-like object whose write()/flush() resolve `sys.stdout`
        at CALL time, not at construction time.
    Leakage guard: N/A.
    Failure mode: `logging.StreamHandler(sys.stdout)` (the naive version of
        this) binds whatever `sys.stdout` IS at handler-construction time --
        which for `configure_json_logging()` (called at each module's import
        time, see below) is long before pytest's `capsys` fixture
        monkeypatches `sys.stdout` for an individual test. The handler would
        then keep writing to the ORIGINAL stdout object forever, invisible
        to `capsys.readouterr()` -- discovered because
        tests/test_version_mismatch.py's `assert "WARNING" in captured.out`
        started failing the moment retrain_trigger.py's warning moved from
        `print()` to `logger.warning()`. Looking up `sys.stdout` fresh on
        every write (via the module-level `sys` reference, not a captured
        local) is what makes this handler observe capsys's per-test swap.
    """

    def write(self, message: str) -> None:
        sys.stdout.write(message)

    def flush(self) -> None:
        sys.stdout.flush()


def configure_json_logging(level: int = logging.INFO) -> None:
    """
    Purpose: Attach one JSON-formatted handler to the root logger, so every
        module-level `logging.getLogger(__name__)` call site across this
        project's batch entrypoints emits structured output without each
        one re-implementing handler setup.
    Leakage guard: N/A.
    Failure mode: Idempotent by design -- safe to call from more than one
        entrypoint in the same process (e.g. retrain_trigger.py imports
        register_model.py, and both call this) without stacking duplicate
        handlers and duplicating every log line. Writes to stdout via
        _StdoutProxy (see its docstring for why not a plain
        `logging.StreamHandler(sys.stdout)`), not the logging module's
        stderr default, so `capsys`-based tests (see
        tests/test_version_mismatch.py) and shell pipelines (`make drift |
        jq`) both see it the same way a `print()` call would have.
    """
    root = logging.getLogger()
    if any(isinstance(h.formatter, JsonFormatter) for h in root.handlers):
        return
    handler = logging.StreamHandler(_StdoutProxy())
    handler.setFormatter(JsonFormatter())
    root.addHandler(handler)
    root.setLevel(level)
