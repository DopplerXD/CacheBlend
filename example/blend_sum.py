"""汇总执行脚本：顺序运行 musique/wikimqa/samsum 三个实验脚本。"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import time
from typing import Dict, Optional

# 允许从项目根目录导入模块（保持 `python example/blend_sum.py` 可直接运行）。
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from utils.experiment_output import ExperimentOutputWriter, utc8_now_str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="汇总运行 blend_* 实验脚本")
    parser.add_argument("--count", type=int, default=100, help="每个数据集最多评测样本数")
    parser.add_argument("--qaw-ratio", type=float, default=0.7, help="query-aware 重算比例")
    return parser.parse_args()


def _latest_output_file(output_dir: str, run_tag: str) -> Optional[str]:
    pattern = os.path.join(output_dir, f"*_{run_tag}.output")
    files = glob.glob(pattern)
    if not files:
        return None
    files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return files[0]


def _load_run_summary(output_file: Optional[str]) -> Optional[Dict]:
    if not output_file or not os.path.exists(output_file):
        return None
    last_summary = None
    with open(output_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if payload.get("event") == "run_summary":
                last_summary = payload
    return last_summary


def main() -> None:
    args = parse_args()

    sum_dir = os.path.join(ROOT_DIR, "outputs", "sum")
    writer = ExperimentOutputWriter.create(sum_dir, run_tag="sum")

    jobs = [
        {
            "name": "musique",
            "script": os.path.join(ROOT_DIR, "example", "blend_musique.py"),
            "run_tag": "musique",
        },
        {
            "name": "wikimqa",
            "script": os.path.join(ROOT_DIR, "example", "blend_wikimqa.py"),
            "run_tag": "wikimqa",
        },
        {
            "name": "samsum",
            "script": os.path.join(ROOT_DIR, "example", "blend_samsum.py"),
            "run_tag": "samsum",
        },
    ]

    writer.append_json({
        "event": "run_start",
        "script": "example/blend_sum.py",
        "started_at": utc8_now_str(),
        "count": args.count,
        "qaw_ratio": args.qaw_ratio,
        "output_dir": sum_dir,
        "jobs": [j["name"] for j in jobs],
    })

    success_count = 0
    for job in jobs:
        before_latest = _latest_output_file(sum_dir, job["run_tag"])
        cmd = [
            sys.executable,
            job["script"],
            "--count",
            str(args.count),
            "--qaw-ratio",
            str(args.qaw_ratio),
            "--output-dir",
            sum_dir,
        ]

        t0 = time.perf_counter()
        proc = subprocess.run(
            cmd,
            cwd=ROOT_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        duration_s = time.perf_counter() - t0
        after_latest = _latest_output_file(sum_dir, job["run_tag"])
        summary = _load_run_summary(after_latest)

        if proc.returncode == 0:
            success_count += 1

        writer.append_json({
            "event": "job_result",
            "job": job["name"],
            "command": cmd,
            "returncode": proc.returncode,
            "duration_s": duration_s,
            "before_latest_output": before_latest,
            "after_latest_output": after_latest,
            "run_summary": summary,
            "stderr_tail": proc.stderr[-2000:] if proc.stderr else "",
        })

    writer.append_json({
        "event": "run_summary",
        "job_count": len(jobs),
        "success_count": success_count,
        "failed_count": len(jobs) - success_count,
        "ended_at": utc8_now_str(),
    })


if __name__ == "__main__":
    main()
