"""Filesystem-based coordination helpers for long-running distributed jobs."""

from __future__ import annotations

import json
import logging
import os
import time
import traceback
from pathlib import Path

from utils.distributed import is_main_process


class PeerRankFailedError(RuntimeError):
    """Raised when another distributed worker has already reported failure."""


def write_json_atomic(path: Path, payload: object) -> None:
    """Write JSON atomically via a temporary file in the same directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp.{os.getpid()}.{time.time_ns()}")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    os.replace(tmp_path, path)


def failure_file(directory: Path, rank: int) -> Path:
    return directory / f".rank_{rank}_failed.json"


def write_rank_failure(directory: Path, rank: int, exc: BaseException) -> None:
    payload = {
        "rank": rank,
        "error_type": type(exc).__name__,
        "error": str(exc),
        "traceback": traceback.format_exc(),
    }
    write_json_atomic(failure_file(directory, rank), payload)


def write_rank_failure_best_effort(
    directory: Path,
    rank: int,
    exc: BaseException,
    logger: logging.Logger | None = None,
) -> None:
    failure_path = failure_file(directory, rank)
    if failure_path.exists():
        return
    try:
        write_rank_failure(directory, rank, exc)
    except BaseException as write_exc:
        if logger is not None:
            logger.warning(
                "Failed to write failure sentinel for rank %d to %s: %s",
                rank,
                failure_path,
                write_exc,
            )


def raise_if_any_rank_failed(
    directory: Path,
    dist_info: dict[str, int | bool],
) -> None:
    rank = int(dist_info["rank"])
    world_size = int(dist_info["world_size"])
    for other_rank in range(world_size):
        if other_rank == rank:
            continue
        failure_path = failure_file(directory, other_rank)
        if not failure_path.exists():
            continue
        with open(failure_path, encoding="utf-8") as f:
            failure = json.load(f)
        raise PeerRankFailedError(
            f"Rank {other_rank} failed: "
            f"{failure.get('error_type', 'UnknownError')}: "
            f"{failure.get('error', 'No error message provided.')}"
        )


def wait_for_paths(
    paths: list[Path],
    *,
    directory: Path,
    dist_info: dict[str, int | bool],
    poll_secs: float = 5.0,
) -> None:
    while not all(path.exists() for path in paths):
        raise_if_any_rank_failed(directory, dist_info)
        time.sleep(poll_secs)


def cleanup_gather_artifacts(directory: Path) -> None:
    for pattern in (
        ".rank_*_records.json",
        ".rank_*_oracle_collection.json",
        ".rank_*_records_ready",
        ".rank_*_failed.json",
    ):
        for path in directory.glob(pattern):
            path.unlink(missing_ok=True)


_file_barrier_counter: dict[str, int] = {}


def file_barrier(name: str, dist_info: dict[str, int | bool], directory: Path) -> None:
    """File-based barrier that never times out (unlike NCCL)."""
    world_size = int(dist_info["world_size"])
    if world_size <= 1:
        return
    rank = int(dist_info["rank"])
    directory.mkdir(parents=True, exist_ok=True)

    seq = _file_barrier_counter.get(name, 0)
    _file_barrier_counter[name] = seq + 1
    tag = f".barrier_{name}_{seq}"

    arrive = directory / f"{tag}_arrive_rank{rank}"
    arrive.touch()
    all_arrivals = [directory / f"{tag}_arrive_rank{r}" for r in range(world_size)]
    wait_for_paths(
        all_arrivals,
        directory=directory,
        dist_info=dist_info,
    )

    ack = directory / f"{tag}_ack_rank{rank}"
    ack.touch()
    all_acks = [directory / f"{tag}_ack_rank{r}" for r in range(world_size)]
    wait_for_paths(
        all_acks,
        directory=directory,
        dist_info=dist_info,
    )

    done = directory / f"{tag}_done_rank{rank}"
    done.touch()

    if not is_main_process(dist_info):
        return
    all_done = [directory / f"{tag}_done_rank{r}" for r in range(world_size)]
    wait_for_paths(
        all_done,
        directory=directory,
        dist_info=dist_info,
    )
    for path in [*all_arrivals, *all_acks, *all_done]:
        path.unlink(missing_ok=True)
