"""MusiQue 质量退化案例分析：筛选 full_prefill 正确但 qaw 错误样本。"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
from typing import Dict, List, Optional, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="筛选 full_prefill 正确但 qaw 错误的样本，并输出人工归因模板")
    parser.add_argument(
        "--inputs",
        nargs="*",
        default=["outputs/*musique*.output"],
        help="输入日志文件或 glob（默认: outputs/*musique*.output）",
    )
    parser.add_argument(
        "--single-log",
        default="",
        help="仅分析单个日志文件（优先级高于 --inputs）",
    )
    parser.add_argument(
        "--dataset",
        default="inputs/musique_s.json",
        help="MusiQue 数据集路径，用于补充问题上下文",
    )
    parser.add_argument(
        "--qaw-key",
        default="qaw_default",
        choices=["qaw_default", "qaw_no_suffix", "qaw_random_topk", "query_aware"],
        help="选择要分析的 qaw 结果来源",
    )
    parser.add_argument("--top-k", type=int, default=20, help="输出案例数量")
    parser.add_argument(
        "--full-correct-f1",
        type=float,
        default=1.0,
        help="认为 full_prefill 正确的 F1 下限（默认 1.0）",
    )
    parser.add_argument(
        "--qaw-wrong-f1",
        type=float,
        default=0.999999,
        help="认为 qaw 错误的 F1 上限（默认 < 1.0）",
    )
    parser.add_argument(
        "--out-csv",
        default="outputs/musique_degradation_cases.csv",
        help="输出 CSV 路径",
    )
    parser.add_argument(
        "--out-jsonl",
        default="outputs/musique_degradation_cases.jsonl",
        help="输出 JSONL 路径",
    )
    parser.add_argument(
        "--out-appendix-csv",
        default="outputs/musique_degradation_appendix.csv",
        help="论文附录格式 CSV 路径",
    )
    return parser.parse_args()


def expand_files(patterns: List[str]) -> List[str]:
    files: List[str] = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if matches:
            files.extend(matches)
        elif os.path.isfile(pattern):
            files.append(pattern)
    dedup = []
    seen = set()
    for path in files:
        abs_path = os.path.abspath(path)
        if abs_path in seen:
            continue
        seen.add(abs_path)
        dedup.append(abs_path)
    return dedup


def load_musique_dataset(path: str) -> Dict[int, Dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # sample_idx 在实验日志里通常从 1 开始。
    return {idx: item for idx, item in enumerate(data, start=1)}


def extract_qaw_block(sample_result: Dict, qaw_key: str) -> Optional[Dict]:
    # runner 格式
    if "qaw_variants" in sample_result:
        return sample_result["qaw_variants"].get(qaw_key)
    # 旧格式
    if qaw_key == "query_aware" and "query_aware" in sample_result:
        return sample_result["query_aware"]
    # 回退：如果目标是 qaw_default，且只有旧字段，则使用 query_aware
    if qaw_key == "qaw_default" and "query_aware" in sample_result:
        return sample_result["query_aware"]
    return None


def iter_sample_results(files: List[str]) -> List[Tuple[str, int, Dict]]:
    rows: List[Tuple[str, int, Dict]] = []
    for file_path in files:
        with open(file_path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line or not line.startswith("{"):
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if payload.get("event") == "sample_result":
                    rows.append((file_path, line_no, payload))
    return rows


def tokenize_simple(text: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9]+", (text or "").lower())


def suggest_reason(answers: List[str], qaw_text: str) -> str:
    gold = " | ".join(answers or [])
    pred = qaw_text or ""
    gold_tokens = tokenize_simple(gold)
    pred_tokens = tokenize_simple(pred)

    if not pred_tokens:
        return "信息缺失"

    gold_digits = re.findall(r"\d+(?:\.\d+)?", gold)
    pred_digits = re.findall(r"\d+(?:\.\d+)?", pred)
    if gold_digits and gold_digits != pred_digits:
        return "数值错误"

    yesno_gold = {"yes", "no"} & set(gold_tokens)
    yesno_pred = {"yes", "no"} & set(pred_tokens)
    if yesno_gold and yesno_pred and yesno_gold != yesno_pred:
        return "极性错误"

    overlap = len(set(gold_tokens) & set(pred_tokens))
    denom = max(1, len(set(gold_tokens)))
    overlap_ratio = overlap / denom
    if overlap_ratio < 0.2:
        return "实体混淆"

    if len(pred_tokens) < max(1, len(gold_tokens) // 2):
        return "信息缺失"

    return "其他（待人工判断）"


def build_case_records(sample_rows: List[Tuple[str, int, Dict]], dataset_map: Dict[int, Dict],
                       qaw_key: str, full_correct_f1: float,
                       qaw_wrong_f1: float) -> List[Dict]:
    cases: List[Dict] = []
    for source_file, line_no, item in sample_rows:
        full_block = item.get("full_prefill") or {}
        qaw_block = extract_qaw_block(item, qaw_key) or {}
        if not full_block or not qaw_block:
            continue

        full_f1 = full_block.get("f1")
        qaw_f1 = qaw_block.get("f1")
        if full_f1 is None or qaw_f1 is None:
            continue
        if full_f1 < full_correct_f1:
            continue
        if qaw_f1 > qaw_wrong_f1:
            continue

        sample_idx = int(item.get("sample_idx", -1))
        data_item = dataset_map.get(sample_idx, {})
        answers = item.get("answers") or data_item.get("answers") or []
        question = item.get("question") or data_item.get("question") or ""
        ctxs = data_item.get("ctxs") or []
        ctx_titles = [ctx.get("title", "") for ctx in ctxs[:5]]

        case = {
            "source_file": os.path.basename(source_file),
            "source_path": source_file,
            "line_no": line_no,
            "sample_idx": sample_idx,
            "question": question,
            "answers": answers,
            "ctx_titles_top5": ctx_titles,
            "full_prefill_text": full_block.get("generated_text", ""),
            "full_prefill_f1": full_f1,
            "qaw_key": qaw_key,
            "qaw_text": qaw_block.get("generated_text", ""),
            "qaw_f1": qaw_f1,
            "degradation_gap": round(float(full_f1) - float(qaw_f1), 6),
        }
        case["auto_reason"] = suggest_reason(answers, case["qaw_text"])
        case["manual_reason"] = ""
        case["manual_notes"] = ""
        cases.append(case)
    return cases


def write_jsonl(path: str, rows: List[Dict]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")


def write_csv(path: str, rows: List[Dict]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    columns = [
        "source_file",
        "line_no",
        "sample_idx",
        "question",
        "answers",
        "ctx_titles_top5",
        "full_prefill_text",
        "full_prefill_f1",
        "qaw_key",
        "qaw_text",
        "qaw_f1",
        "degradation_gap",
        "auto_reason",
        "manual_reason",
        "manual_notes",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            out = dict(row)
            out["answers"] = json.dumps(out.get("answers", []), ensure_ascii=False)
            out["ctx_titles_top5"] = json.dumps(out.get("ctx_titles_top5", []),
                                                ensure_ascii=False)
            writer.writerow(out)


def write_appendix_csv(path: str, rows: List[Dict]) -> None:
    """导出更适合论文附录的格式：含答案对照与标签枚举。"""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    columns = [
        "case_id",
        "sample_idx",
        "question",
        "gold_answers",
        "full_prefill_answer",
        "full_prefill_f1",
        "qaw_variant",
        "qaw_answer",
        "qaw_f1",
        "degradation_gap",
        "label_enum",
        "auto_label",
        "manual_label",
        "manual_note",
        "source_file",
        "line_no",
    ]
    label_enum = "信息缺失|实体混淆|数值错误|极性错误|其他"
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for i, row in enumerate(rows, start=1):
            writer.writerow({
                "case_id": i,
                "sample_idx": row.get("sample_idx"),
                "question": row.get("question", ""),
                "gold_answers": json.dumps(row.get("answers", []), ensure_ascii=False),
                "full_prefill_answer": row.get("full_prefill_text", ""),
                "full_prefill_f1": row.get("full_prefill_f1"),
                "qaw_variant": row.get("qaw_key", ""),
                "qaw_answer": row.get("qaw_text", ""),
                "qaw_f1": row.get("qaw_f1"),
                "degradation_gap": row.get("degradation_gap"),
                "label_enum": label_enum,
                "auto_label": row.get("auto_reason", ""),
                "manual_label": row.get("manual_reason", ""),
                "manual_note": row.get("manual_notes", ""),
                "source_file": row.get("source_file", ""),
                "line_no": row.get("line_no", ""),
            })


def main() -> None:
    args = parse_args()
    input_patterns = [args.single_log] if args.single_log else args.inputs
    files = expand_files(input_patterns)
    if not files:
        raise FileNotFoundError("未找到可用日志文件，请检查 --inputs")

    dataset_map = load_musique_dataset(args.dataset)
    sample_rows = iter_sample_results(files)
    if not sample_rows:
        raise ValueError("未在日志中找到 sample_result 记录")

    candidates = build_case_records(
        sample_rows=sample_rows,
        dataset_map=dataset_map,
        qaw_key=args.qaw_key,
        full_correct_f1=args.full_correct_f1,
        qaw_wrong_f1=args.qaw_wrong_f1,
    )
    if not candidates:
        raise ValueError("未筛到符合条件的退化样本，请放宽 F1 阈值")

    # 优先选“退化幅度最大”的样本。
    candidates.sort(key=lambda x: (x["degradation_gap"], x["full_prefill_f1"]),
                    reverse=True)
    selected = candidates[:args.top_k]

    write_jsonl(args.out_jsonl, selected)
    write_csv(args.out_csv, selected)
    write_appendix_csv(args.out_appendix_csv, selected)

    reason_count: Dict[str, int] = {}
    for row in selected:
        reason = row["auto_reason"]
        reason_count[reason] = reason_count.get(reason, 0) + 1

    print(f"输入日志文件: {len(files)}")
    print(f"sample_result 总数: {len(sample_rows)}")
    print(f"符合条件候选数: {len(candidates)}")
    print(f"输出案例数: {len(selected)}")
    print(f"CSV: {os.path.abspath(args.out_csv)}")
    print(f"JSONL: {os.path.abspath(args.out_jsonl)}")
    print(f"Appendix CSV: {os.path.abspath(args.out_appendix_csv)}")
    print(f"自动归因分布: {json.dumps(reason_count, ensure_ascii=False)}")


if __name__ == "__main__":
    main()
