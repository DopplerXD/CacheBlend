"""实验输出落盘工具。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import json
import os
from typing import Any, Dict


UTC_PLUS_8 = timezone(timedelta(hours=8))


def utc8_now() -> datetime:
    """返回 UTC+8 的当前时间。"""
    return datetime.now(timezone.utc).astimezone(UTC_PLUS_8)


def utc8_now_str() -> str:
    """返回 UTC+8 的标准时间字符串。"""
    return utc8_now().strftime("%Y-%m-%d %H:%M:%S")


@dataclass
class ExperimentOutputWriter:
    """将实验结果追加写入 outputs/YYYYMMDDHHMM_<tag>。"""

    output_dir: str
    file_path: str

    @classmethod
    def create(cls, output_dir: str, run_tag: str) -> "ExperimentOutputWriter":
        os.makedirs(output_dir, exist_ok=True)
        safe_tag = (run_tag or "run").strip().replace(" ", "_")
        filename = f"{utc8_now().strftime('%Y%m%d%H%M')}_{safe_tag}"
        return cls(output_dir=output_dir, file_path=os.path.join(output_dir, filename))

    def append_line(self, text: str) -> None:
        with open(self.file_path, "a", encoding="utf-8") as f:
            f.write(text.rstrip("\n"))
            f.write("\n")

    def append_json(self, payload: Dict[str, Any]) -> None:
        self.append_line(json.dumps(payload, ensure_ascii=False))
