#!/usr/bin/env python3
import argparse
import os
import sys
import pickle
import math
import datetime as dt
from pathlib import Path
from typing import Any, Iterable, Optional, Tuple


def human_size(num_bytes: int) -> str:
    if num_bytes is None:
        return "unknown"
    if num_bytes < 0:
        return str(num_bytes)
    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    if num_bytes == 0:
        return "0 B"
    power = int(math.floor(math.log(num_bytes, 1024)))
    power = min(power, len(units) - 1)
    size = num_bytes / (1024 ** power)
    return f"{size:.2f} {units[power]}"


def try_import_pandas():
    try:
        import pandas as pd  # type: ignore
        return pd
    except Exception:
        return None


def try_import_numpy():
    try:
        import numpy as np  # type: ignore
        return np
    except Exception:
        return None


def is_dataframe(obj: Any) -> bool:
    pd = try_import_pandas()
    return pd is not None and isinstance(obj, pd.DataFrame)


def is_series(obj: Any) -> bool:
    pd = try_import_pandas()
    return pd is not None and isinstance(obj, pd.Series)


def is_numpy_array(obj: Any) -> bool:
    np = try_import_numpy()
    return np is not None and isinstance(obj, np.ndarray)


def safe_len(obj: Any) -> Optional[int]:
    try:
        return len(obj)  # type: ignore[arg-type]
    except Exception:
        return None


def summarize_dataframe(df, head_rows: int = 5, max_cols_print: int = 12) -> str:
    pd = try_import_pandas()
    if pd is None:
        return "pandas not available"

    lines = []
    mem_bytes = None
    try:
        mem_bytes = int(df.memory_usage(deep=True).sum())
    except Exception:
        mem_bytes = None
    lines.append(f"DataFrame: shape={df.shape}, memory={human_size(mem_bytes)}")
    try:
        dtypes = df.dtypes.astype(str)
        nunique = None
        try:
            nunique = df.nunique(dropna=False)
        except Exception:
            nunique = None
        nulls = None
        try:
            nulls = df.isna().sum()
        except Exception:
            nulls = None
        header = ["column", "dtype", "nunique", "nulls"]
        rows = []
        for col in df.columns[:max_cols_print]:
            dtype = dtypes.get(col, "?")
            nu = int(nunique[col]) if nunique is not None and col in nunique else "?"
            na = int(nulls[col]) if nulls is not None and col in nulls else "?"
            rows.append([str(col), str(dtype), str(nu), str(na)])
        if len(df.columns) > max_cols_print:
            rows.append([f"… (+{len(df.columns) - max_cols_print} more)", "", "", ""]) 
        # simple table formatting
        col_widths = [max(len(r[i]) for r in ([header] + rows)) for i in range(len(header))]
        fmt = "  ".join(["{:<" + str(w) + "}" for w in col_widths])
        lines.append(fmt.format(*header))
        lines.append(fmt.format(*["-" * w for w in col_widths]))
        for r in rows:
            lines.append(fmt.format(*r))
    except Exception as e:
        lines.append(f"Failed to summarize columns: {e}")

    try:
        # print head with limited columns
        with pd.option_context(
            "display.max_rows", min(head_rows, 10),
            "display.max_columns", max_cols_print,
            "display.width", 200,
        ):
            lines.append("\nHead:")
            lines.append(str(df.head(head_rows)))
    except Exception:
        pass
    return "\n".join(lines)


def summarize_series(s, head_rows: int = 10) -> str:
    lines = []
    lines.append(f"Series: len={len(s)}, dtype={getattr(s, 'dtype', '?')}")
    try:
        lines.append("Head:")
        lines.append(str(s.head(head_rows)))
    except Exception:
        pass
    try:
        vc = s.value_counts(dropna=False).head(20)
        lines.append("\nValue counts (top 20):")
        lines.append(str(vc))
    except Exception:
        pass
    return "\n".join(lines)


def summarize_numpy(arr) -> str:
    np = try_import_numpy()
    lines = [f"ndarray: shape={arr.shape}, dtype={arr.dtype}"]
    if np is not None:
        try:
            if np.issubdtype(arr.dtype, np.number) and arr.size > 0:
                lines.append(
                    f"min={np.nanmin(arr):.4g}, max={np.nanmax(arr):.4g}, mean={np.nanmean(arr):.4g}, std={np.nanstd(arr):.4g}"
                )
        except Exception:
            pass
    return "\n".join(lines)


def iter_sample(iterable: Iterable[Any], max_items: int) -> Iterable[Any]:
    count = 0
    for item in iterable:
        if count >= max_items:
            break
        yield item
        count += 1


def summarize_object(
    obj: Any,
    name: Optional[str] = None,
    indent: int = 0,
    max_depth: int = 2,
    max_items: int = 10,
    head_rows: int = 5,
) -> str:
    prefix = " " * indent
    title = f"{prefix}{name + ': ' if name else ''}{type(obj).__name__}"

    try:
        if is_dataframe(obj):
            return f"{title}\n{prefix}" + summarize_dataframe(obj, head_rows=head_rows).replace("\n", f"\n{prefix}")
        if is_series(obj):
            return f"{title}\n{prefix}" + summarize_series(obj, head_rows=10).replace("\n", f"\n{prefix}")
        if is_numpy_array(obj):
            return f"{title}\n{prefix}" + summarize_numpy(obj).replace("\n", f"\n{prefix}")
    except Exception:
        pass

    # Collections
    if isinstance(obj, dict):
        lines = [f"{title} (len={len(obj)})"]
        if max_depth <= 0:
            sample_keys = list(iter_sample(obj.keys(), max_items))
            lines.append(f"{prefix}keys sample: {sample_keys}{' …' if len(obj) > len(sample_keys) else ''}")
            return "\n".join(lines)
        shown = 0
        for k in obj:
            if shown >= max_items:
                lines.append(f"{prefix}… (+{len(obj) - shown} more)")
                break
            try:
                v = obj[k]
            except Exception as e:
                lines.append(f"{prefix}{k}: <error reading value: {e}>")
                shown += 1
                continue
            lines.append(summarize_object(v, name=str(k), indent=indent + 2, max_depth=max_depth - 1, max_items=max_items, head_rows=head_rows))
            shown += 1
        return "\n".join(lines)

    if isinstance(obj, (list, tuple)):
        length = safe_len(obj)
        lines = [f"{title} (len={length})"]
        if max_depth <= 0:
            return "\n".join(lines)
        for idx, item in enumerate(iter_sample(obj, max_items)):
            lines.append(summarize_object(item, name=f"[{idx}]", indent=indent + 2, max_depth=max_depth - 1, max_items=max_items, head_rows=head_rows))
        if length is not None and length > max_items:
            lines.append(f"{prefix}… (+{length - max_items} more)")
        return "\n".join(lines)

    if isinstance(obj, set):
        length = len(obj)
        sample_vals = list(iter_sample(iter(obj), max_items))
        more = f" … (+{length - len(sample_vals)} more)" if length > len(sample_vals) else ""
        return f"{title} (len={length}) sample={sample_vals}{more}"

    # Fallback
    length = safe_len(obj)
    extra = f", len={length}" if length is not None else ""
    return f"{title}{extra}"


def find_latest_pickle(preprocess_dir: Path) -> Optional[Path]:
    candidates = sorted(preprocess_dir.glob("processed_data_*.pkl"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def find_latest_log(preprocess_dir: Path) -> Optional[Path]:
    candidates = sorted(preprocess_dir.glob("preprocessing_log_*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def load_pickle(path: Path) -> Any:
    with path.open("rb") as f:
        return pickle.load(f)


def tail_file(path: Path, lines: int = 80) -> str:
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            data = f.readlines()
        return "".join(data[-lines:])
    except Exception as e:
        return f"<failed to read log: {e}>"


def fmt_path_info(path: Path) -> str:
    try:
        stat = path.stat()
        size = human_size(stat.st_size)
        mtime = dt.datetime.fromtimestamp(stat.st_mtime).isoformat(sep=" ", timespec="seconds")
        return f"{path}  (size={size}, modified={mtime})"
    except Exception:
        return str(path)


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect structure of training preprocess data (.pkl) and logs.")
    parser.add_argument("path", nargs="?", default=str(Path.cwd()), help="Path to a preprocess directory or a specific processed_data_*.pkl file")
    parser.add_argument("--max-depth", type=int, default=2, help="Max recursion depth when summarizing nested objects")
    parser.add_argument("--max-items", type=int, default=10, help="Max items to show per container")
    parser.add_argument("--head", type=int, default=5, help="Number of rows to show for DataFrame head")
    parser.add_argument("--log-lines", type=int, default=80, help="Number of lines to tail from preprocessing log")
    parser.add_argument("--no-log", action="store_true", help="Do not print log tail")
    args = parser.parse_args(list(argv) if argv is not None else None)

    target = Path(args.path)
    if target.is_dir():
        preprocess_dir = target
        pkl_path = find_latest_pickle(preprocess_dir)
        log_path = find_latest_log(preprocess_dir)
    else:
        pkl_path = target if target.suffix == ".pkl" else None
        preprocess_dir = target.parent if target.exists() else Path.cwd()
        log_path = find_latest_log(preprocess_dir)

    print("\n=== Preprocess directory ===")
    print(preprocess_dir)
    print("\n=== Pickle file ===")
    if pkl_path is None:
        print("No processed_data_*.pkl found")
    else:
        print(fmt_path_info(pkl_path))
        try:
            obj = load_pickle(pkl_path)
        except Exception as e:
            print(f"\nFailed to load pickle: {e}")
            obj = None

        if obj is not None:
            print("\n=== Top-level summary ===")
            print(summarize_object(
                obj,
                name="processed_data",
                indent=0,
                max_depth=args.max_depth,
                max_items=args.max_items,
                head_rows=args.head,
            ))

    if not args.no_log:
        print("\n=== Preprocessing log (tail) ===")
        if log_path is None:
            print("No preprocessing_log_*.log found")
        else:
            print(fmt_path_info(log_path))
            print("\n" + tail_file(log_path, lines=args.log_lines))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


