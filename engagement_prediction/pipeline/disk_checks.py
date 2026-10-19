"""Free-space checks for pipeline output storage."""

from math import ceil
from pathlib import Path
import shutil
from typing import Any


def check_disk_space(
    *,
    output_dir: Path,
    min_output_free_gib: float,
) -> dict[str, Any]:
    """Report available space without reserving storage or deleting artifacts."""

    path = Path(output_dir).resolve()
    required_free_bytes = ceil(min_output_free_gib * 1024**3)
    check = {
        "name": "disk_output",
        "filesystem_role": "output",
        "path": str(path),
        "required_free_bytes": required_free_bytes,
    }
    try:
        free_bytes = shutil.disk_usage(path).free
    except OSError as exc:
        check.update(
            status="fail",
            free_bytes=None,
            message=f"Cannot check output disk space at {path}: {exc}",
        )
    else:
        check.update(
            status="pass" if free_bytes >= required_free_bytes else "fail",
            free_bytes=free_bytes,
            message=(
                f"Output disk space at {path}: {free_bytes / 1024**3:.2f} GiB "
                f"free; requires at least {min_output_free_gib:.2f} GiB"
            ),
        )

    return {
        "status": check["status"],
        "checks": [check],
    }
