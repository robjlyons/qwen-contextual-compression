"""Audit the exact installed FreeToken sources needed by the low-VRAM adapter."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import _bootstrap


NEEDLES = (
    "_load_maybe_quantized",
    "Qwen3_5Model",
    "Qwen3_5DecoderLayer",
    "Qwen3_5DenseMLP",
    "_run_scheduler",
    "load_state_dict",
    "finalize",
)


def main():
    parser = argparse.ArgumentParser(description="Inspect an installed FreeToken tree without modifying or importing it")
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.package_root.resolve()
    if not (root / "__init__.py").is_file():
        raise FileNotFoundError(f"Not a FreeToken package root: {root}")
    matches = {needle: [] for needle in NEEDLES}
    files = []
    for path in sorted(root.rglob("*.py")):
        data = path.read_bytes()
        text = data.decode("utf-8", errors="replace")
        relative = str(path.relative_to(root))
        files.append({"path": relative, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)})
        for needle in NEEDLES:
            for line_number, line in enumerate(text.splitlines(), 1):
                if needle in line:
                    matches[needle].append({"path": relative, "line": line_number, "text": line.strip()})
    report = {"package_root": str(root), "files": files, "matches": matches}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"files": len(files), "output": str(args.output), "match_counts": {k: len(v) for k, v in matches.items()}}, indent=2))


if __name__ == "__main__":
    main()
