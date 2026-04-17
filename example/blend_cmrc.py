"""CMRC 实验薄封装：默认走 blend_runner 的 prefill_qaw 模式。"""

from __future__ import annotations

import os
import subprocess
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main() -> int:
    cmd = [
        sys.executable,
        os.path.join(ROOT_DIR, "example", "blend_runner.py"),
        "--dataset",
        "cmrc",
        "--count",
        "50",
        "--qaw-ratio",
        "0.7",
        "--suffix-len",
        "32",
        "--methods",
        "prefill_qaw",
        *sys.argv[1:],
    ]
    return subprocess.call(cmd, cwd=ROOT_DIR)


if __name__ == "__main__":
    raise SystemExit(main())
