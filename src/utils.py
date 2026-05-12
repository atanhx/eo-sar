"""Shared utilities for logging and timestamping."""

import sys
import time
from pathlib import Path


class Tee:
    """Get ``stdout`` to both the terminal and a log file.

    All ``print()`` calls while the Tee is active are written to both
    destinations.  Supports use as a context manager.

    Example::

        with Tee(Path("logs/run.log")):
            print("this goes to console and file")

    Args:
        log_path: Path to the log file.  Parent directories are created
            automatically if they do not exist.
    """

    def __init__(self, log_path: Path) -> None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(log_path, "a", encoding="utf-8")
        self._stdout = sys.stdout
        sys.stdout = self

    def write(self, data: str) -> None:
        """Write *data* to both the terminal and the log file."""
        self._stdout.write(data)
        self._file.write(data)
        self._file.flush()

    def flush(self) -> None:
        """Flush both output streams."""
        self._stdout.flush()
        self._file.flush()

    def close(self) -> None:
        """Restore the original ``stdout`` and close the log file."""
        sys.stdout = self._stdout
        self._file.close()

    def __enter__(self) -> "Tee":
        return self

    def __exit__(self, *_) -> None:
        self.close()


def run_timestamp() -> str:
    """Return a compact timestamp string for log file naming.

    Returns:
        String in ``YYYYMMDD_HHMMSS`` format.
    """
    return time.strftime("%Y%m%d_%H%M%S")
