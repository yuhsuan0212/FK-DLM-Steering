"""
Shared logging utilities for SCDD.

Provides a rank-aware logger and a Slurm-compatible progress tracker
that outputs plain-text lines instead of carriage-return progress bars.
"""

import logging
import sys
import time
from pathlib import Path
from typing import Iterable, Iterator, TypeVar

T = TypeVar("T")

_DEFAULT_FMT = "%(levelname)s | %(asctime)s | [rank%(rank)s] %(message)s"
_RANK0_FMT = "%(levelname)s | %(asctime)s | %(message)s"
_DEFAULT_DATEFMT = "%Y-%m-%d %H:%M:%S"


def get_logger(
    name: str = "scdd",
    rank: int = 0,
    log_file: str | Path | None = None,
    level: int = logging.INFO,
    fmt: str = _DEFAULT_FMT,
    datefmt: str = _DEFAULT_DATEFMT,
    all_ranks: bool = False,
) -> logging.Logger:
    """Create a rank-aware logger.

    By default only rank 0 produces output (other ranks are silenced).
    Set ``all_ranks=True`` to let every rank log with a ``[rankN]`` prefix.

    Args:
        name: Logger name (used by Python's logging hierarchy).
        rank: Distributed rank.
        log_file: Optional path.  When set, a ``FileHandler`` is attached.
        level: Logging level (default ``INFO``).
        fmt: Log line format string.
        datefmt: Date/time format string.
        all_ranks: If True, all ranks produce output (with rank prefix).
    """
    logger = logging.getLogger(name)

    # Prevent duplicate handlers when called multiple times.
    if logger.hasHandlers():
        logger.handlers.clear()

    # Silence non-rank-0 loggers unless all_ranks is set.
    if rank != 0 and not all_ranks:
        logger.addHandler(logging.NullHandler())
        logger.setLevel(logging.CRITICAL + 1)
        logger.propagate = False
        return logger

    logger.setLevel(level)
    logger.propagate = False  # prevent double-printing via parent loggers

    # Rank 0 keeps the clean format; other ranks get a rank-prefixed format.
    if rank == 0:
        effective_fmt = _RANK0_FMT if fmt == _DEFAULT_FMT else fmt
    else:
        effective_fmt = fmt

    class _RankFilter(logging.Filter):
        def filter(self, record):
            record.rank = rank
            return True

    formatter = logging.Formatter(effective_fmt, datefmt)

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(formatter)
    console.addFilter(_RankFilter())
    logger.addHandler(console)

    if log_file is not None:
        fh = logging.FileHandler(str(log_file), mode="a")
        fh.setLevel(level)
        fh.setFormatter(formatter)
        fh.addFilter(_RankFilter())
        logger.addHandler(fh)

    return logger


class ProgressLogger:
    """Slurm-friendly progress tracker.

    Wraps an iterable and periodically emits a plain-text log line via
    ``logger.info``.  Unlike ``tqdm``, output does **not** rely on
    carriage returns so it works correctly in non-TTY environments
    (e.g. Slurm ``--output`` log files).

    Example::

        logger = get_logger("eval", rank=rank)
        for batch in ProgressLogger(dataloader, logger, desc="Generating"):
            ...

    Each log line looks like::

        INFO | 2025-02-14 17:30:00 | [Generating]  128/3108 ( 4.1%)  elapsed=00:02:15  ETA=00:52:30
    """

    def __init__(
        self,
        iterable: Iterable[T],
        logger: logging.Logger,
        *,
        total: int | None = None,
        desc: str = "Progress",
        log_every_n: int | None = None,
        log_every_secs: float = 30.0,
        disable: bool = False,
    ):
        self._iterable = iterable
        self._logger = logger
        self._desc = desc
        self._log_every_n = log_every_n
        self._log_every_secs = log_every_secs
        self._disable = disable

        if total is not None:
            self._total: int | None = total
        elif hasattr(iterable, "__len__"):
            self._total = len(iterable)  # type: ignore[arg-type]
        else:
            self._total = None

    def __iter__(self) -> Iterator[T]:
        if self._disable:
            yield from self._iterable
            return

        start = time.monotonic()
        last_log_time = start
        step = 0

        for item in self._iterable:
            yield item
            step += 1

            now = time.monotonic()
            should_log = False

            if self._log_every_n is not None and step % self._log_every_n == 0:
                should_log = True
            if now - last_log_time >= self._log_every_secs:
                should_log = True
            # Always log the last step
            if self._total is not None and step == self._total:
                should_log = True

            if should_log:
                elapsed = now - start
                elapsed_str = _format_duration(elapsed)
                if self._total is not None and step > 0:
                    pct = step / self._total * 100
                    eta = elapsed / step * (self._total - step)
                    eta_str = _format_duration(eta)
                    self._logger.info(
                        "[%s]  %d/%d (%5.1f%%)  elapsed=%s  ETA=%s",
                        self._desc,
                        step,
                        self._total,
                        pct,
                        elapsed_str,
                        eta_str,
                    )
                else:
                    self._logger.info(
                        "[%s]  %d steps  elapsed=%s",
                        self._desc,
                        step,
                        elapsed_str,
                    )
                last_log_time = now

    def __len__(self) -> int:
        if self._total is not None:
            return self._total
        raise TypeError(f"object of type '{type(self).__name__}' has no len()")


def _format_duration(seconds: float) -> str:
    """Format seconds into HH:MM:SS."""
    s = int(seconds)
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"
