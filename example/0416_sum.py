"""0416 调度脚本：顺序运行 5 个实验任务，并在任务间清理显存。"""

from __future__ import annotations

import glob
import json
import os
import subprocess
import sys
import time
from typing import Dict, List, Optional

# 允许从项目根目录导入模块（保持 `python example/0416_sum.py` 可直接运行）。
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from utils.experiment_output import ExperimentOutputWriter, utc8_now, utc8_now_str


OUTPUT_DIR = os.path.join(ROOT_DIR, "outputs", "0416_sum")


def _latest_output_file(output_dir: str, run_tag: str) -> Optional[str]:
    pattern = os.path.join(output_dir, f"*_{run_tag}.output")
    files = glob.glob(pattern)
    if not files:
        return None
    files.sort(key=lambda path: os.path.getmtime(path), reverse=True)
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


def _gpu_cleanup_command() -> List[str]:
    cleanup_code = (
        "import gc\n"
        "gc.collect()\n"
        "try:\n"
        "    import torch\n"
        "    if torch.cuda.is_available():\n"
        "        torch.cuda.empty_cache()\n"
        "        if hasattr(torch.cuda, 'ipc_collect'):\n"
        "            torch.cuda.ipc_collect()\n"
        "except Exception:\n"
        "    pass\n"
    )
    return [sys.executable, "-c", cleanup_code]


def _run_cleanup() -> Dict[str, object]:
    t0 = time.perf_counter()
    proc = subprocess.run(
        _gpu_cleanup_command(),
        cwd=ROOT_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return {
        "returncode": proc.returncode,
        "duration_s": time.perf_counter() - t0,
        "stderr_tail": proc.stderr[-1000:] if proc.stderr else "",
    }


def _job_spec(job: Dict[str, object]) -> str:
    return (f"{job['script_name']}-{job['count']}-{job['ratio_spec']}-"
            f"{job['suffix_spec']}")


def _build_fallback_job_output_path(run_tag: str) -> str:
    timestamp = utc8_now().strftime("%Y%m%d%H%M%S")
    return os.path.join(OUTPUT_DIR, f"{timestamp}_{run_tag}.output")


def _resolve_job_output_file(run_tag: str, before_latest: Optional[str],
                             after_latest: Optional[str]) -> str:
    if after_latest is not None and after_latest != before_latest:
        return after_latest
    return _build_fallback_job_output_path(run_tag)


def _append_job_exception(job: Dict[str, object], output_file: str, returncode: int,
                          duration_s: float, stdout_text: str, stderr_text: str,
                          exception_message: str = "",
                          exception_type: str = "") -> None:
    job_writer = ExperimentOutputWriter(output_dir=OUTPUT_DIR, file_path=output_file)
    job_writer.append_json({
        "event": "job_exception",
        "job": job["name"],
        "job_spec": _job_spec(job),
        "script": job["script"],
        "command": job["command"],
        "returncode": returncode,
        "duration_s": duration_s,
        "exception_type": exception_type,
        "exception_message": exception_message,
        "stdout_tail": stdout_text[-2000:] if stdout_text else "",
        "stderr_tail": stderr_text[-4000:] if stderr_text else "",
        "logged_at": utc8_now_str(),
    })


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    writer = ExperimentOutputWriter.create(OUTPUT_DIR, run_tag="0416_sum")

    jobs: List[Dict[str, object]] = [
        {
            "name": "musique_curve_025_100",
            "script_name": "blend_musique_curve.py",
            "script": os.path.join(ROOT_DIR, "example", "blend_musique_curve.py"),
            "run_tag": "0416_musique_curve_025_100",
            "count": 50,
            "ratio_spec": "0.25~1.0(step=0.05)",
            "suffix_spec": "32",
            "command": [
                sys.executable,
                os.path.join(ROOT_DIR, "example", "blend_musique_curve.py"),
                "--count",
                "50",
                "--qaw-ratio-min",
                "0.25",
                "--qaw-ratio-max",
                "1.0",
                "--qaw-ratio-step",
                "0.05",
                "--suffix-len",
                "32",
                "--run-tag",
                "0416_musique_curve_025_100",
                "--output-dir",
                OUTPUT_DIR,
            ],
        },
        {
            "name": "wikimqa_curve_025_100",
            "script_name": "blend_wikimqa_curve.py",
            "script": os.path.join(ROOT_DIR, "example", "blend_wikimqa_curve.py"),
            "run_tag": "0416_wikimqa_curve_025_100",
            "count": 50,
            "ratio_spec": "0.25~1.0(step=0.05)",
            "suffix_spec": "32",
            "command": [
                sys.executable,
                os.path.join(ROOT_DIR, "example", "blend_wikimqa_curve.py"),
                "--count",
                "50",
                "--qaw-ratio-min",
                "0.25",
                "--qaw-ratio-max",
                "1.0",
                "--qaw-ratio-step",
                "0.05",
                "--suffix-len",
                "32",
                "--run-tag",
                "0416_wikimqa_curve_025_100",
                "--output-dir",
                OUTPUT_DIR,
            ],
        },
        {
            "name": "blend_curve_default",
            "script_name": "blend_curve.py",
            "script": os.path.join(ROOT_DIR, "example", "blend_curve.py"),
            "run_tag": "0416_blend_curve_default",
            "count": 50,
            "ratio_spec": "0~1.0(step=0.05)",
            "suffix_spec": "32",
            "command": [
                sys.executable,
                os.path.join(ROOT_DIR, "example", "blend_curve.py"),
                "--count",
                "50",
                "--qaw-ratio-min",
                "0.0",
                "--qaw-ratio-max",
                "1.0",
                "--qaw-ratio-step",
                "0.05",
                "--run-tag",
                "0416_blend_curve_default",
                "--output-dir",
                OUTPUT_DIR,
            ],
        },
        {
            "name": "suffix_curve_musique_a",
            "script_name": "blend_suffix_curve.py",
            "script": os.path.join(ROOT_DIR, "example", "blend_suffix_curve.py"),
            "run_tag": "0416_suffix_curve_musique_a",
            "count": 50,
            "ratio_spec": "0.7",
            "suffix_spec": "0~32(step=8)",
            "command": [
                sys.executable,
                os.path.join(ROOT_DIR, "example", "blend_suffix_curve.py"),
                "--dataset",
                "musique",
                "--count",
                "50",
                "--qaw-ratio",
                "0.7",
                "--suffix-len-min",
                "0",
                "--suffix-len-max",
                "32",
                "--suffix-len-step",
                "8",
                "--run-tag",
                "0416_suffix_curve_musique_a",
                "--output-dir",
                OUTPUT_DIR,
            ],
        },
        {
            "name": "suffix_curve_musique_b",
            "script_name": "blend_suffix_curve.py",
            "script": os.path.join(ROOT_DIR, "example", "blend_suffix_curve.py"),
            "run_tag": "0416_suffix_curve_musique_b",
            "count": 50,
            "ratio_spec": "0.7",
            "suffix_spec": "0~32(step=8)",
            "command": [
                sys.executable,
                os.path.join(ROOT_DIR, "example", "blend_suffix_curve.py"),
                "--dataset",
                "musique",
                "--count",
                "50",
                "--qaw-ratio",
                "0.7",
                "--suffix-len-min",
                "0",
                "--suffix-len-max",
                "32",
                "--suffix-len-step",
                "8",
                "--run-tag",
                "0416_suffix_curve_musique_b",
                "--output-dir",
                OUTPUT_DIR,
            ],
        },
    ]

    writer.append_json({
        "event": "run_start",
        "script": "example/0416_sum.py",
        "started_at": utc8_now_str(),
        "output_dir": OUTPUT_DIR,
        "job_count": len(jobs),
        "jobs": [_job_spec(job) for job in jobs],
        "note": (
            "按用户提供列表执行；suffix_len_curve musique 任务在原始需求中重复两次，"
            "因此这里保留两次。"
        ),
    })

    success_count = 0
    cleanup_success_count = 0
    for index, job in enumerate(jobs, start=1):
        before_latest = _latest_output_file(OUTPUT_DIR, str(job["run_tag"]))
        proc: Optional[subprocess.CompletedProcess[str]] = None
        duration_s = 0.0
        stdout_text = ""
        stderr_text = ""
        run_exception_message = ""
        run_exception_type = ""
        t0 = time.perf_counter()
        try:
            proc = subprocess.run(
                job["command"],
                cwd=ROOT_DIR,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            duration_s = time.perf_counter() - t0
            stdout_text = proc.stdout or ""
            stderr_text = proc.stderr or ""
        except Exception as exc:  # pragma: no cover - 仅用于任务调度兜底
            duration_s = time.perf_counter() - t0
            run_exception_message = str(exc)
            run_exception_type = type(exc).__name__

        after_latest = _latest_output_file(OUTPUT_DIR, str(job["run_tag"]))
        if run_exception_type or (proc is not None and proc.returncode != 0):
            exception_output_file = _resolve_job_output_file(
                str(job["run_tag"]), before_latest, after_latest)
            _append_job_exception(
                job=job,
                output_file=exception_output_file,
                returncode=proc.returncode if proc is not None else -1,
                duration_s=duration_s,
                stdout_text=stdout_text,
                stderr_text=stderr_text,
                exception_message=run_exception_message,
                exception_type=run_exception_type,
            )
            after_latest = exception_output_file

        summary = _load_run_summary(after_latest)
        cleanup_result = _run_cleanup()

        if proc is not None and proc.returncode == 0 and not run_exception_type:
            success_count += 1
        if cleanup_result["returncode"] == 0:
            cleanup_success_count += 1

        writer.append_json({
            "event": "job_result",
            "job_index": index,
            "job": job["name"],
            "job_spec": _job_spec(job),
            "command": job["command"],
            "returncode": proc.returncode if proc is not None else -1,
            "duration_s": duration_s,
            "before_latest_output": before_latest,
            "after_latest_output": after_latest,
            "run_summary": summary,
            "exception_type": run_exception_type,
            "exception_message": run_exception_message,
            "stdout_tail": stdout_text[-2000:] if stdout_text else "",
            "stderr_tail": stderr_text[-2000:] if stderr_text else "",
            "cleanup": cleanup_result,
        })

    writer.append_json({
        "event": "run_summary",
        "job_count": len(jobs),
        "success_count": success_count,
        "failed_count": len(jobs) - success_count,
        "cleanup_success_count": cleanup_success_count,
        "cleanup_failed_count": len(jobs) - cleanup_success_count,
        "ended_at": utc8_now_str(),
    })


if __name__ == "__main__":
    main()
