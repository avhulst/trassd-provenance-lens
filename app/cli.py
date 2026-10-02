"""Command line: python -m app.cli image1.jpg image2.png [--json]

Exit codes: 0 = no AI label, 2 = at least one image with an AI label,
1 = at least one file was not readable (and no AI label found).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .detectors import LEVEL_AI, analyze
from .report import DEFAULT_LANGUAGE, LANGUAGES, to_text


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m app.cli",
        description="Checks images for machine-readable AI labels.",
    )
    parser.add_argument("images", nargs="+", type=Path, help="image files to check")
    parser.add_argument("--json", action="store_true", help="output JSON instead of plain text")
    parser.add_argument("--lang", choices=LANGUAGES, default=DEFAULT_LANGUAGE, help="language of the plain-text report")
    args = parser.parse_args(argv)

    results = []
    unreadable = False
    for path in args.images:
        try:
            data = path.read_bytes()
        except OSError as exc:
            prefix = "Datei nicht lesbar" if args.lang == "de" else "File not readable"
            print(f"{prefix}: {path}: {exc.strerror or exc}", file=sys.stderr)
            unreadable = True
            continue
        results.append(analyze(data, str(path)))

    if args.json:
        print(json.dumps([r.to_dict() for r in results], ensure_ascii=False, indent=2))
    else:
        print(("\n" + "#" * 52 + "\n\n").join(to_text(r, args.lang) for r in results), end="")

    if any(r.verdict_level == LEVEL_AI for r in results):
        return 2
    return 1 if unreadable else 0


if __name__ == "__main__":
    sys.exit(main())
