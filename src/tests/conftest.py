"""
Pytest session configuration.

Keep the test suite quiet and deterministic:
- Disable logging-internal exception tracebacks (some third-party handlers flush
  after pytest has already closed the capture stream on Windows).
"""

from __future__ import annotations


def _quiet_logging_errors() -> None:
    import logging

    # Prevent "Logging error" tracebacks from polluting test output.
    logging.raiseExceptions = False

    # Best-effort patch: `ascii_colors.ConsoleHandler.close()` may flush a closed stream
    # at interpreter shutdown under pytest capture on Windows.
    try:
        import ascii_colors
    except Exception:
        return

    ConsoleHandler = getattr(ascii_colors, "ConsoleHandler", None)
    if ConsoleHandler is None:
        return

    # Silence ascii_colors's own traceback printer during pytest shutdown.
    # This library prints errors directly to stderr even when exceptions are handled.
    BaseHandler = getattr(ascii_colors, "Handler", None)
    if BaseHandler is not None:
        BaseHandler.handle_error = lambda self, message: None  # type: ignore[assignment]

    orig_close = getattr(ConsoleHandler, "close", None)
    if not callable(orig_close):
        return

    def _safe_close(self):
        try:
            orig_close(self)
        except ValueError:
            # I/O operation on closed file (ignored during test shutdown)
            return

    ConsoleHandler.close = _safe_close  # type: ignore[assignment]


def pytest_configure():  # pragma: no cover
    _quiet_logging_errors()
    # Ensure this is the last thing to run before Python shuts down logging handlers.
    import atexit

    atexit.register(_quiet_logging_errors)


def pytest_sessionfinish():  # pragma: no cover
    # Re-apply right before interpreter shutdown; some plugins mutate logging settings.
    _quiet_logging_errors()
