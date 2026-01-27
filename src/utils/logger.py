#!/usr/bin/env python

import contextvars
import logging
import os
import sys
import time

from src.core.settings import settings

# Per-request id (set by FastAPI middleware). Defaults to "-" when not in a request context.
request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")


class RequestIdFilter(logging.Filter):
    """Inject `request_id` into every record so formatters never KeyError."""

    def filter(self, record: logging.LogRecord) -> bool:  # pragma: no cover
        record.request_id = request_id_var.get()
        return True


class HourlyFileHandler(logging.Handler):
    """Write logs to `logs/YYYY-MM-DD/YYYY-MM-DD_HH.log` with hourly rotation."""

    def __init__(self, log_directory: str):
        super().__init__()
        self.log_directory = log_directory
        self.current_key: str | None = None
        self.file_handler: logging.FileHandler | None = None

        self.formatter = logging.Formatter(
            "[%(asctime)s] [req:%(request_id)s] [%(filename)s|%(funcName)s] [line:%(lineno)d] "
            "%(levelname)-8s: %(message)s",
            datefmt="%Y-%m-%d %H:%M",
        )
        self.setFormatter(self.formatter)

    def emit(self, record: logging.LogRecord) -> None:
        try:
            # Defensive: some libraries log before filters run; never let formatting fail.
            if not hasattr(record, "request_id"):
                record.request_id = request_id_var.get()

            ts = time.time()
            date_str = time.strftime("%Y-%m-%d", time.localtime(ts))
            hour_str = time.strftime("%H", time.localtime(ts))

            key = f"{date_str}_{hour_str}"
            if self.current_key != key or self.file_handler is None:
                self._rotate(date_str, hour_str, key)

            if self.file_handler:
                self.file_handler.emit(record)
        except Exception:
            self.handleError(record)

    def _rotate(self, date_str: str, hour_str: str, key: str) -> None:
        if self.file_handler:
            self.file_handler.close()

        date_folder = os.path.join(self.log_directory, date_str)
        os.makedirs(date_folder, exist_ok=True)

        log_filename = f"{date_str}_{hour_str}.log"
        log_file = os.path.join(date_folder, log_filename)

        self.file_handler = logging.FileHandler(log_file, encoding="utf-8")
        self.file_handler.setFormatter(self.formatter)
        self.current_key = key

    def close(self) -> None:
        if self.file_handler:
            self.file_handler.close()
        super().close()


_setup_done = False


def _parse_log_level(value: str | None, default: int = logging.DEBUG) -> int:
    if not value:
        return default
    s = str(value).strip().upper()
    if not s:
        return default
    return int(getattr(logging, s, default))


def setup_global_logging() -> None:
    """Configure root logging once (console + hourly files)."""

    global _setup_done
    if _setup_done:
        return

    # During tests, avoid noisy "Logging error" tracebacks from third-party handlers.
    # pytest captures/tears down streams aggressively; logging should never fail tests.
    if "pytest" in sys.modules or os.getenv("PYTEST_CURRENT_TEST"):
        logging.raiseExceptions = False

    root_logger = logging.getLogger()
    log_level = _parse_log_level(os.getenv("LOG_LEVEL"), default=logging.DEBUG)
    root_logger.setLevel(log_level)

    # Avoid duplicated handlers on reload.
    for h in list(root_logger.handlers):
        root_logger.removeHandler(h)

    log_dir = str(settings.paths.log_dir)
    os.makedirs(log_dir, exist_ok=True)

    formatter = logging.Formatter(
        "[%(asctime)s] [req:%(request_id)s] [%(filename)s|%(funcName)s] [line:%(lineno)d] %(levelname)-8s: %(message)s",
        datefmt="%Y-%m-%d %H:%M",
    )

    request_id_filter = RequestIdFilter()

    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    console_handler.addFilter(request_id_filter)
    root_logger.addHandler(console_handler)

    # Skip file logging during tests (faster, avoids FS churn).
    if "pytest" not in sys.modules and not os.getenv("DISABLE_FILE_LOGS"):
        file_handler = HourlyFileHandler(log_dir)
        file_handler.setLevel(log_level)
        file_handler.addFilter(request_id_filter)
        root_logger.addHandler(file_handler)

    # Reduce noisy third-party logging.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("neo4j").setLevel(logging.WARNING)

    _setup_done = True


def get_logger(name: str | None = None) -> logging.Logger:
    """Get a configured logger (ensures global setup runs once)."""

    setup_global_logging()
    return logging.getLogger(name)


class LogManager:
    """
    Legacy compatibility wrapper.

    Prefer: `logger = get_logger(__name__)`.
    """

    def __init__(self, log_directory: str | None = None):
        setup_global_logging()
        self.logger = logging.getLogger("LogManager")
        self.logger.setLevel(logging.DEBUG)

    def debug(self, message: str) -> None:
        self.logger.debug(message, stacklevel=2)

    def info(self, message: str) -> None:
        self.logger.info(message, stacklevel=2)

    def warning(self, message: str) -> None:
        self.logger.warning(message, stacklevel=2)

    def error(self, message: str) -> None:
        self.logger.error(message, stacklevel=2)


if __name__ == "__main__":
    log = get_logger("TestModule")
    log.info("Testing get_logger")

    legacy = LogManager()
    legacy.info("Testing legacy LogManager")
