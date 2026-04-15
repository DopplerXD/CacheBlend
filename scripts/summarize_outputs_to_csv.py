"""将 outputs/*.output 中的结构化事件聚合为 CSV。"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from typing import Dict, List


PREFERRED_COLUMNS = [
    "source_file",
    "source_path",
    "line_no",
    "event",
    "script",
    "dataset_name",
    "dataset",
    "model",
    "recomp_ratio",
    "sample_count",
    "started_at",
    "ended_at",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="聚合 .output 为 CSV")
    parser.add_argument(
        "--inputs",
        nargs="*",
        default=["outputs/*.output"],
        help="输入文件或 glob 模式（默认: outputs/*.output）",
    )
    parser.add_argument("--event",
                        default="run_summary",
                        help="筛选事件名（默认: run_summary）")
    parser.add_argument(
        "--output",
        default="outputs/run_summary_aggregate.csv",
        help="输出 CSV 路径",
    )
    return parser.parse_args()


def expand_input_files(patterns: List[str]) -> List[str]:
    files: List[str] = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if matches:
            files.extend(matches)
        elif os.path.isfile(pattern):
            files.append(pattern)
    # 去重并保持顺序
    deduped = []
    seen = set()
    for path in files:
        abs_path = os.path.abspath(path)
        if abs_path in seen:
            continue
        seen.add(abs_path)
        deduped.append(abs_path)
    return deduped


def normalize_value(value):
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return value


def collect_rows(files: List[str], target_event: str) -> List[Dict]:
    rows: List[Dict] = []
    for file_path in files:
        run_meta = {}
        with open(file_path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(payload, dict):
                    continue
                if payload.get("event") == "run_start":
                    for key in ("script", "dataset", "dataset_name", "model",
                                "started_at"):
                        if key in payload:
                            run_meta[key] = payload[key]
                if payload.get("event") != target_event:
                    continue
                row = {
                    "source_file": os.path.basename(file_path),
                    "source_path": file_path,
                    "line_no": line_no,
                }
                row.update(run_meta)
                row.update(payload)
                rows.append(row)
    return rows


def build_fieldnames(rows: List[Dict]) -> List[str]:
    keys = set()
    for row in rows:
        keys.update(row.keys())
    ordered = [k for k in PREFERRED_COLUMNS if k in keys]
    tail = sorted([k for k in keys if k not in ordered])
    return ordered + tail


def write_csv(rows: List[Dict], output_path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fieldnames = build_fieldnames(rows)
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            normalized = {k: normalize_value(v) for k, v in row.items()}
            writer.writerow(normalized)


def main() -> None:
    args = parse_args()
    files = expand_input_files(args.inputs)
    if not files:
        raise FileNotFoundError("未找到可用 .output 文件")

    rows = collect_rows(files, args.event)
    if not rows:
        raise ValueError(f"未找到 event={args.event} 的记录")

    write_csv(rows, args.output)
    print(f"聚合完成: {len(rows)} 行 -> {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()
