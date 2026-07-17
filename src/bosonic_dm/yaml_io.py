# Copyright (C) 2025 Francesco Borra
#

"""Portable, human-readable YAML serialization and migration."""

from __future__ import annotations

import argparse
import logging
import math
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
from ruamel.yaml import YAML

logger = logging.getLogger(__name__)


def to_builtin(value: object) -> object:
    """Recursively convert values to portable YAML-compatible built-ins."""
    if isinstance(value, np.generic):
        return to_builtin(value.item())
    if isinstance(value, np.ndarray):
        return [to_builtin(item) for item in value.tolist()]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {to_builtin(key): to_builtin(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [to_builtin(item) for item in value]
    return value


def read_yaml(path: str | Path) -> object:
    """Safely read a portable YAML file."""
    yaml = YAML(typ="safe")
    with Path(path).open("r", encoding="utf-8") as stream:
        return yaml.load(stream)


def write_yaml(data: object, path: Path | str) -> None:
    """Atomically write data using only portable, human-readable YAML types."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    clean_data = to_builtin(data)

    yaml = YAML()
    yaml.default_flow_style = False
    yaml.indent(mapping=2, sequence=4, offset=2)

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            yaml.dump(clean_data, stream)
        temporary_path.replace(output_path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def migrate_yaml_file(
    source: str | Path,
    destination: str | Path | None = None,
    *,
    overwrite: bool = False,
) -> Path:
    """Rewrite a legacy dbetto YAML file using portable built-in scalar types.

    When *destination* is omitted, the source is replaced atomically in place.
    """
    from dbetto import Props  # noqa: PLC0415

    source_path = Path(source)
    destination_path = Path(destination) if destination else source_path
    if destination_path.exists() and destination_path != source_path and not overwrite:
        msg = f"Destination already exists: {destination_path}"
        raise FileExistsError(msg)

    legacy_data = Props.read_from(str(source_path))
    expected = to_builtin(legacy_data)
    write_yaml(expected, destination_path)
    migrated = read_yaml(destination_path)
    if migrated != expected:
        msg = f"YAML migration verification failed for {source_path}"
        raise ValueError(msg)
    return destination_path


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for migrating legacy YAML files."""
    parser = argparse.ArgumentParser(
        description="Rewrite legacy YAML files without NumPy/Python tags."
    )
    parser.add_argument("inputs", nargs="+", type=Path)
    destination = parser.add_mutually_exclusive_group(required=True)
    destination.add_argument(
        "--in-place",
        action="store_true",
        help="Atomically replace each source file.",
    )
    destination.add_argument(
        "--output-dir",
        type=Path,
        help="Write migrated files to this directory.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite files that already exist in --output-dir.",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    for source in args.inputs:
        if args.in_place:
            output_path = source
        else:
            output_dir: Path = args.output_dir
            output_path = output_dir / source.name
        migrated_path = migrate_yaml_file(
            source,
            output_path,
            overwrite=args.overwrite,
        )
        logger.info("%s", migrated_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
