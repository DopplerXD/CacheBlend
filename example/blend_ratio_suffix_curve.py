"""QAW ratio x suffix_len 网格实验：支持 MusiQue / WikiMQA。"""

from __future__ import annotations

import argparse
import os
import sys

# 允许从项目根目录导入模块（保持 `python example/blend_ratio_suffix_curve.py` 可直接运行）。
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
EXAMPLE_DIR = os.path.dirname(os.path.abspath(__file__))
if EXAMPLE_DIR not in sys.path:
    sys.path.insert(0, EXAMPLE_DIR)

from config import RuntimeConfig
from model.yi_model import YiModelRunner
from utils.experiment_output import ExperimentOutputWriter, utc8_now_str
from utils.logging_utils import setup_logger

from blend_curve_common import DATASET_SPECS, build_run_summary_payload, evaluate_qaw_grid, float_grid, int_grid, load_dataset, run_fixed_baselines


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="QAW ratio x suffix_len grid curve")
    parser.add_argument("--dataset",
                        choices=sorted(DATASET_SPECS.keys()),
                        default="musique")
    parser.add_argument("--model-name",
                        type=str,
                        default="",
                        help="显式指定模型路径/名称；未传时回退 MODEL_NAME")
    parser.add_argument("--count", type=int, default=50, help="最多评测样本数")
    parser.add_argument("--qaw-ratio-min", type=float, default=0.45)
    parser.add_argument("--qaw-ratio-max", type=float, default=0.80)
    parser.add_argument("--qaw-ratio-step", type=float, default=0.05)
    parser.add_argument("--suffix-len-min", type=int, default=0)
    parser.add_argument("--suffix-len-max", type=int, default=32)
    parser.add_argument("--suffix-len-step", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--output-dir",
                        type=str,
                        default=os.path.join(ROOT_DIR, "outputs"),
                        help="实验日志输出目录")
    parser.add_argument("--run-tag",
                        type=str,
                        default="",
                        help="输出文件 run_tag，默认按 dataset 生成")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    spec, eval_dataset = load_dataset(args.dataset)

    cfg = RuntimeConfig()
    if args.model_name.strip():
        cfg.model_name = args.model_name.strip()
    cfg.max_new_tokens = args.max_new_tokens

    logger = setup_logger(f"blend_ratio_suffix_curve_{args.dataset}", cfg.log_level)
    logger.disabled = True

    model_runner = YiModelRunner(cfg.model_name, cfg.device, cfg.model_dtype,
                                 logger)
    run_tag = args.run_tag or spec["grid_curve_tag"]
    output_writer = ExperimentOutputWriter.create(
        args.output_dir, run_tag=run_tag)

    ratios = float_grid(args.qaw_ratio_min, args.qaw_ratio_max,
                        args.qaw_ratio_step)
    suffix_values = int_grid(args.suffix_len_min, args.suffix_len_max,
                             args.suffix_len_step)
    output_writer.append_json({
        "event": "run_start",
        "script": "example/blend_ratio_suffix_curve.py",
        "dataset": spec["dataset_path"],
        "dataset_name": args.dataset,
        "started_at": utc8_now_str(),
        "model": cfg.model_name,
        "max_new_tokens": cfg.max_new_tokens,
        "temperature": cfg.temperature,
        "top_p": cfg.top_p,
        "count": args.count,
        "qaw_ratio_min": args.qaw_ratio_min,
        "qaw_ratio_max": args.qaw_ratio_max,
        "qaw_ratio_step": args.qaw_ratio_step,
        "suffix_len_min": args.suffix_len_min,
        "suffix_len_max": args.suffix_len_max,
        "suffix_len_step": args.suffix_len_step,
        "ratio_group_count": len(ratios),
        "suffix_group_count": len(suffix_values),
        "grid_group_count": len(ratios) * len(suffix_values),
        "curve_mode": "ratio_suffix_grid",
        "kvd_enabled": False,
    })

    sample_limit = min(len(eval_dataset), args.count)
    baseline_stats = run_fixed_baselines(
        eval_dataset=eval_dataset,
        spec=spec,
        cfg=cfg,
        model_runner=model_runner,
        logger=logger,
        sample_limit=sample_limit,
    )

    processed_samples, metrics = evaluate_qaw_grid(
        eval_dataset=eval_dataset,
        spec=spec,
        cfg=cfg,
        model_runner=model_runner,
        logger=logger,
        sample_limit=sample_limit,
        ratio_values=ratios,
        suffix_values=suffix_values,
    )

    total_groups = len(ratios) * len(suffix_values)
    group_idx = 0
    for recomp_ratio in ratios:
        for suffix_len in suffix_values:
            group_idx += 1
            payload = build_run_summary_payload(
                processed_samples=processed_samples,
                baseline_stats=baseline_stats,
                grid_bucket=metrics[(recomp_ratio, suffix_len)],
                recomp_ratio=recomp_ratio,
                suffix_len=suffix_len,
            )
            payload["ended_at"] = utc8_now_str()
            output_writer.append_json(payload)
            print(
                f"[curve] 组完成 {group_idx}/{total_groups}, "
                f"run_summary 已写入, recomp_ratio={recomp_ratio:.2f}, "
                f"suffix_len={suffix_len}",
                flush=True,
            )


if __name__ == "__main__":
    main()
