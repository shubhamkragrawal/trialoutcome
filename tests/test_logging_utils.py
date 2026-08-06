"""Fast, dependency-free unit tests for M9-17's core/logging_utils.py.
No DB, no MLflow required.
"""

from __future__ import annotations

import json
import logging

from core.logging_utils import JsonFormatter, configure_json_logging


def test_configure_json_logging_is_idempotent():
    root = logging.getLogger()
    before = len(root.handlers)

    configure_json_logging()
    configure_json_logging()
    configure_json_logging()

    json_handlers = [h for h in root.handlers if isinstance(h.formatter, JsonFormatter)]
    assert len(json_handlers) == 1
    assert len(root.handlers) == before or len(root.handlers) == before + 1


def test_json_formatter_emits_valid_json_with_expected_keys():
    record = logging.LogRecord(
        name="test.logger",
        level=logging.WARNING,
        pathname=__file__,
        lineno=1,
        msg="something %s happened",
        args=("bad",),
        exc_info=None,
    )
    formatted = JsonFormatter().format(record)
    payload = json.loads(formatted)

    assert payload["level"] == "WARNING"
    assert payload["logger"] == "test.logger"
    assert payload["msg"] == "something bad happened"
    assert "ts" in payload


def test_configured_logging_is_visible_to_capsys(capsys):
    """Regression test (M9-17): a naive `logging.StreamHandler(sys.stdout)`
    captures sys.stdout at handler-construction time, which pytest's capsys
    fixture has not yet monkeypatched for THIS test the first time
    configure_json_logging() ever runs in the process (usually triggered by
    an earlier test module's import). That bug meant warnings logged via
    this setup were invisible to `capsys.readouterr()` -- see
    tests/test_version_mismatch.py's `assert "WARNING" in captured.out`,
    which broke silently in the field until traced back to this. The fix
    (_StdoutProxy) resolves sys.stdout at write time instead."""
    configure_json_logging()
    logger = logging.getLogger("test.capsys_visibility")

    logger.warning("this warning must reach capsys")

    captured = capsys.readouterr()
    assert "this warning must reach capsys" in captured.out
    assert "WARNING" in captured.out
