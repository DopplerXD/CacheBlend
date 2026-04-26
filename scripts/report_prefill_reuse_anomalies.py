#!/usr/bin/env python3
"""Report cases where Full Prefill is unexpectedly below Full Reuse."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from collections import Counter, defaultdict
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional


METRICS = ("f1", "ttft_s", "total_s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find Full Prefill vs Full Reuse expectation violations."
    )
    parser.add_argument(
        "--inputs",
        nargs="*",
        default=["outputs/*.output"],
        help="Input .output files or glob patterns.",
    )
    parser.add_argument(
        "--out-dir",
        default="graduation_project/analyse/qaw_visualizations",
        help="Directory for the generated report and CSV.",
    )
    return parser.parse_args()


def expand_inputs(patterns: Iterable[str]) -> List[str]:
    files: List[str] = []
    seen = set()
    for pattern in patterns:
        matches = glob.glob(pattern)
        if not matches and os.path.isfile(pattern):
            matches = [pattern]
        for path in sorted(matches):
            abs_path = os.path.abspath(path)
            if abs_path in seen:
                continue
            seen.add(abs_path)
            files.append(abs_path)
    return files


def infer_dataset(path: str, dataset_field: Optional[str]) -> str:
    text = f"{os.path.basename(path)} {dataset_field or ''}".lower()
    for name in ("musique", "wikimqa", "cmrc", "samsum"):
        if name in text:
            return name
    return "unknown"


def infer_model(model_field: Optional[str]) -> str:
    text = model_field or ""
    if "Qwen2.5-1.5B" in text:
        return "Qwen2.5-1.5B"
    if "Yi-6B" in text:
        return "Yi-6B"
    return os.path.basename(text) if text else "unknown"


def as_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_json_line(line: str) -> Optional[Dict[str, Any]]:
    line = line.strip()
    if not line.startswith("{"):
        return None
    try:
        payload = json.loads(line)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def add_anomaly(
    rows: List[Dict[str, Any]],
    *,
    file_path: str,
    line_no: int,
    event: str,
    meta: Dict[str, Any],
    sample_idx: Optional[int],
    qaw_ratio: Optional[Any],
    metric: str,
    full_reuse: Optional[float],
    full_prefill: Optional[float],
) -> None:
    if full_reuse is None or full_prefill is None:
        return
    if full_prefill >= full_reuse:
        return
    dataset = infer_dataset(file_path, meta.get("dataset") or meta.get("dataset_name"))
    model = infer_model(meta.get("model") or meta.get("model_name"))
    rows.append(
        {
            "source_file": os.path.basename(file_path),
            "source_path": file_path,
            "line_no": line_no,
            "event": event,
            "dataset": dataset,
            "model": model,
            "sample_idx": sample_idx,
            "qaw_ratio": qaw_ratio,
            "metric": metric,
            "full_reuse": full_reuse,
            "full_prefill": full_prefill,
            "prefill_minus_reuse": full_prefill - full_reuse,
        }
    )


def collect_anomalies(files: Iterable[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for file_path in files:
        meta: Dict[str, Any] = {}
        with open(file_path, "r", encoding="utf-8", errors="replace") as handle:
            for line_no, line in enumerate(handle, start=1):
                payload = parse_json_line(line)
                if not payload:
                    continue

                event = payload.get("event")
                if event == "run_start":
                    meta = payload
                    continue

                if event == "run_summary":
                    qaw_ratio = payload.get("recomp_ratio", meta.get("qaw_ratio"))
                    for metric in METRICS:
                        full_reuse = as_float(payload.get(f"full_reuse_avg_{metric}"))
                        full_prefill = as_float(payload.get(f"full_prefill_avg_{metric}"))
                        add_anomaly(
                            rows,
                            file_path=file_path,
                            line_no=line_no,
                            event="run_summary",
                            meta=meta,
                            sample_idx=None,
                            qaw_ratio=qaw_ratio,
                            metric=metric,
                            full_reuse=full_reuse,
                            full_prefill=full_prefill,
                        )

                if event == "sample_result":
                    full_reuse_node = payload.get("full_reuse") or {}
                    full_prefill_node = payload.get("full_prefill") or {}
                    qaw_ratio = meta.get("qaw_ratio")
                    for metric in METRICS:
                        add_anomaly(
                            rows,
                            file_path=file_path,
                            line_no=line_no,
                            event="sample_result",
                            meta=meta,
                            sample_idx=payload.get("sample_idx"),
                            qaw_ratio=qaw_ratio,
                            metric=metric,
                            full_reuse=as_float(full_reuse_node.get(metric)),
                            full_prefill=as_float(full_prefill_node.get(metric)),
                        )
    return rows


def write_csv(rows: List[Dict[str, Any]], output_path: str) -> None:
    fieldnames = [
        "source_file",
        "source_path",
        "line_no",
        "event",
        "dataset",
        "model",
        "sample_idx",
        "qaw_ratio",
        "metric",
        "full_reuse",
        "full_prefill",
        "prefill_minus_reuse",
    ]
    with open(output_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fmt_float(value: Any) -> str:
    if value is None or value == "":
        return ""
    try:
        return f"{float(value):.6f}"
    except (TypeError, ValueError):
        return str(value)


def markdown_table(rows: List[Dict[str, Any]], columns: List[str]) -> List[str]:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        cells = []
        for col in columns:
            value = row.get(col, "")
            if isinstance(value, float):
                value = fmt_float(value)
            cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def write_markdown(rows: List[Dict[str, Any]], files: List[str], output_path: str, csv_path: str) -> None:
    by_event = Counter(row["event"] for row in rows)
    by_metric = Counter(row["metric"] for row in rows)
    by_group_metric = Counter((row["model"], row["dataset"], row["event"], row["metric"]) for row in rows)

    lines: List[str] = [
        "# Full Prefill vs Full Reuse Anomaly Report",
        "",
        "This report does not modify raw output files. It only lists cases where `full_prefill < full_reuse` for F1, TTFT, or total latency.",
        "",
        f"- Generated at: `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`",
        f"- Input scope: root `outputs/*.output` files",
        f"- Files scanned: `{len(files)}`",
        f"- Anomaly rows: `{len(rows)}`",
        f"- CSV details: `{os.path.abspath(csv_path)}`",
        "",
        "## Summary",
        "",
    ]

    if rows:
        lines.extend(
            markdown_table(
                [
                    {"key": "run_summary", "count": by_event.get("run_summary", 0)},
                    {"key": "sample_result", "count": by_event.get("sample_result", 0)},
                    {"key": "f1", "count": by_metric.get("f1", 0)},
                    {"key": "ttft_s", "count": by_metric.get("ttft_s", 0)},
                    {"key": "total_s", "count": by_metric.get("total_s", 0)},
                ],
                ["key", "count"],
            )
        )
        lines.extend(["", "## Counts by Model, Dataset, Event, Metric", ""])
        grouped_rows = [
            {
                "model": model,
                "dataset": dataset,
                "event": event,
                "metric": metric,
                "count": count,
            }
            for (model, dataset, event, metric), count in sorted(by_group_metric.items())
        ]
        lines.extend(markdown_table(grouped_rows, ["model", "dataset", "event", "metric", "count"]))

        run_rows = [row for row in rows if row["event"] == "run_summary"]
        if run_rows:
            lines.extend(["", "## Run Summary Details", ""])
            detail_rows = [
                {
                    "source_file": row["source_file"],
                    "line_no": row["line_no"],
                    "model": row["model"],
                    "dataset": row["dataset"],
                    "qaw_ratio": row["qaw_ratio"],
                    "metric": row["metric"],
                    "full_reuse": fmt_float(row["full_reuse"]),
                    "full_prefill": fmt_float(row["full_prefill"]),
                    "delta": fmt_float(row["prefill_minus_reuse"]),
                }
                for row in run_rows
            ]
            lines.extend(
                markdown_table(
                    detail_rows,
                    [
                        "source_file",
                        "line_no",
                        "model",
                        "dataset",
                        "qaw_ratio",
                        "metric",
                        "full_reuse",
                        "full_prefill",
                        "delta",
                    ],
                )
            )

        sample_rows = [row for row in rows if row["event"] == "sample_result"]
        if sample_rows:
            lines.extend(["", "## Sample-Level Details", ""])
            lines.append("The full sample-level anomaly list is in the CSV. The table below shows the first 80 rows.")
            lines.append("")
            preview_rows = [
                {
                    "source_file": row["source_file"],
                    "sample_idx": row["sample_idx"],
                    "model": row["model"],
                    "dataset": row["dataset"],
                    "qaw_ratio": row["qaw_ratio"],
                    "metric": row["metric"],
                    "full_reuse": fmt_float(row["full_reuse"]),
                    "full_prefill": fmt_float(row["full_prefill"]),
                    "delta": fmt_float(row["prefill_minus_reuse"]),
                }
                for row in sample_rows[:80]
            ]
            lines.extend(
                markdown_table(
                    preview_rows,
                    [
                        "source_file",
                        "sample_idx",
                        "model",
                        "dataset",
                        "qaw_ratio",
                        "metric",
                        "full_reuse",
                        "full_prefill",
                        "delta",
                    ],
                )
            )
    else:
        lines.append("No anomalies found under the scanned input scope.")

    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    files = expand_inputs(args.inputs)
    if not files:
        raise FileNotFoundError("No input files found.")

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "prefill_reuse_anomalies.csv")
    md_path = os.path.join(out_dir, "prefill_reuse_anomaly_report.md")

    rows = collect_anomalies(files)
    write_csv(rows, csv_path)
    write_markdown(rows, files, md_path, csv_path)

    print(f"Scanned files: {len(files)}")
    print(f"Anomaly rows: {len(rows)}")
    print(f"CSV: {csv_path}")
    print(f"Report: {md_path}")


if __name__ == "__main__":
    main()
