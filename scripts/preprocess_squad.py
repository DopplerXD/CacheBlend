"""SQuAD 预处理脚本：抽取前 N 条并转换为 CacheBlend 示例输入格式。"""

from __future__ import annotations

import argparse
import json
import os
import urllib.request
from typing import Dict, Iterable, List


DEFAULT_SQUAD_V11_TRAIN_URL = "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json"


def _download_squad_json(download_url: str, cache_path: str) -> str:
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    if not os.path.exists(cache_path):
        urllib.request.urlretrieve(download_url, cache_path)
    return cache_path


def _unique_keep_order(items: Iterable[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def convert_squad_to_cacheblend_format(squad_obj: Dict,
                                       max_samples: int) -> List[Dict]:
    """将 SQuAD 原始结构转换为 [ctxs, question, answers] 列表。"""
    converted: List[Dict] = []
    data = squad_obj.get("data", [])

    for article in data:
        title = article.get("title", "")
        for paragraph in article.get("paragraphs", []):
            context = paragraph.get("context", "")
            if not context.strip():
                continue
            for qa in paragraph.get("qas", []):
                question = qa.get("question", "").strip()
                if not question:
                    continue
                answers_raw = qa.get("answers", [])
                answers = _unique_keep_order([
                    (a.get("text", "") or "").strip() for a in answers_raw
                    if (a.get("text", "") or "").strip()
                ])
                if not answers:
                    continue

                converted.append({
                    "ctxs": [{
                        "title": title,
                        "text": context,
                    }],
                    "question": question,
                    "answers": answers,
                })
                if len(converted) >= max_samples:
                    return converted
    return converted


def main() -> None:
    parser = argparse.ArgumentParser(
        description="预处理 SQuAD 为 CacheBlend 输入格式")
    parser.add_argument(
        "--input-json",
        type=str,
        default="",
        help="SQuAD 原始 JSON 文件路径；为空时自动下载 train-v1.1.json",
    )
    parser.add_argument(
        "--download-url",
        type=str,
        default=DEFAULT_SQUAD_V11_TRAIN_URL,
        help="自动下载时使用的 SQuAD URL",
    )
    parser.add_argument(
        "--download-cache-path",
        type=str,
        default="inputs/_cache/squad_train_v11_dev.json",
        help="自动下载文件的本地缓存路径",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="inputs/squad_s.json",
        help="输出文件路径",
    )
    parser.add_argument("--max-samples", type=int, default=1000)
    args = parser.parse_args()

    input_path = args.input_json
    if not input_path:
        input_path = _download_squad_json(args.download_url,
                                          args.download_cache_path)

    with open(input_path, "r", encoding="utf-8") as f:
        squad_obj = json.load(f)

    converted = convert_squad_to_cacheblend_format(
        squad_obj=squad_obj,
        max_samples=args.max_samples,
    )

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(converted, f, ensure_ascii=False, indent=2)

    print(
        f"Done. input={input_path}, output={args.output_json}, samples={len(converted)}"
    )


if __name__ == "__main__":
    main()
