#!/usr/bin/env python3
import re
import shlex
import sys
from pathlib import Path

KEY_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _parse_env_line(raw_line: str) -> tuple[str, str] | None:
    line = raw_line.strip()
    if not line or line.startswith("#") or "=" not in line:
        return None

    key, value = line.split("=", 1)
    key = key.strip()
    if key.startswith("export "):
        key = key.removeprefix("export ").strip()
    if not KEY_PATTERN.match(key):
        return None

    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        value = value[1:-1]
    return key, value


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: export_env_keys.py <env-file> [KEY ...]", file=sys.stderr)
        return 2

    env_path = Path(sys.argv[1])
    wanted = set(sys.argv[2:])
    for raw_line in env_path.read_text().splitlines():
        parsed = _parse_env_line(raw_line)
        if parsed is None:
            continue

        key, value = parsed
        if wanted and key not in wanted:
            continue
        print(f"export {key}={shlex.quote(value)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
