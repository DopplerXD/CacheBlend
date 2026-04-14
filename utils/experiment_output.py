"""实验输出落盘工具。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
import os
from typing import Any, Dict


@dataclass
class ExperimentOutputWriter:
    """将实验结果追加写入 outputs/YYYYMMDDHHMM.output。"""

    output_dir: str
    file_path: str

    @classmethod
    def create(cls, output_dir: str) -> "ExperimentOutputWriter":
        os.makedirs(output_dir, exist_ok=True)
        filename = f"{datetime.now().strftime('%Y%m%d%H%M')}.output"
        return cls(output_dir=output_dir, file_path=os.path.join(output_dir, filename))

    def append_line(self, text: str) -> None:
        with open(self.file_path, "a", encoding="utf-8") as f:
            f.write(text.rstrip("\n"))
            f.write("\n")

    def append_json(self, payload: Dict[str, Any]) -> None:
        self.append_line(json.dumps(payload, ensure_ascii=False))
