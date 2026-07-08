"""Shared progress-bar helpers for pipeline steps."""

from __future__ import annotations

import os
from collections.abc import Iterable, Iterator
from typing import TypeVar

T = TypeVar("T")

_FALSE_VALUES = {"0", "false", "no", "n", "off"}


def progress_enabled() -> bool:
    """Return whether progress bars should be displayed."""
    value = os.environ.get("PIPELINE_PROGRESS", "1").strip().lower()
    return value not in _FALSE_VALUES


def progress_bar(
    iterable: Iterable[T],
    *,
    total: int | None,
    desc: str,
    unit: str,
) -> Iterator[T]:
    """Wrap an iterable in tqdm when available, otherwise return it unchanged."""
    if not progress_enabled():
        yield from iterable
        return

    try:
        from tqdm.auto import tqdm
    except Exception:
        yield from iterable
        return

    with tqdm(
        iterable,
        total=total,
        desc=desc,
        unit=unit,
        dynamic_ncols=True,
        mininterval=1.0,
        leave=True,
    ) as bar:
        yield from bar
