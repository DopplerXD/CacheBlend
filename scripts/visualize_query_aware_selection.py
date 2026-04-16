#!/usr/bin/env python3
"""为 query-aware 选择性重算生成论文可用可视化。

支持四类图：
1. Query-Token similarity heatmap
2. Token score + selected mask
3. Token-level annotated text view
4. Optional attention heatmap (requires extra memory)

说明：
1. 本脚本复用当前仓库的 prompt 构造方式和 query-aware 选点逻辑。
2. 默认假设候选区间与当前 prompt 前缀完全重叠，即 old cache 的 overlap_len=new_len。
   这样更适合展示“选择机制本身”，而不是复原某一次具体运行日志。
3. attention 热力图是补充图。当前方法并不依赖 attention 做选择，因此它默认关闭。
4. 若环境中未安装 matplotlib，可先安装后再运行：
     pip install matplotlib
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import torch

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import colors as mcolors
except ImportError as exc:  # pragma: no cover - runtime dependency hint
    raise ImportError(
        "本脚本需要 matplotlib，请先安装：pip install matplotlib"
    ) from exc


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
EXAMPLE_DIR = os.path.join(ROOT_DIR, "example")
if EXAMPLE_DIR not in sys.path:
    sys.path.insert(0, EXAMPLE_DIR)

from config import RuntimeConfig
from model.yi_model import YiModelRunner
from utils.logging_utils import setup_logger
from example.utils import build_qa_prompt, normalize_question


DATASET_SPECS: Dict[str, Dict[str, str]] = {
    "musique": {
        "dataset_path": os.path.join(ROOT_DIR, "inputs", "musique_s.json"),
        "prefix_prompt": (
            "You will be asked a question after reading several passages. "
            "Please directly answer the question based on the given passages. "
            "Do NOT repeat the question. The answer should be within 5 words.\nPassages:\n"
        ),
        "query_prompt": (
            "\n\nAnswer the question directly based on the given passages. "
            "Do NOT repeat the question. The answer should be within 5 words. \nQuestion:"
        ),
    },
    "wikimqa": {
        "dataset_path": os.path.join(ROOT_DIR, "inputs", "wikimqa_s.json"),
        "prefix_prompt": (
            "Answer the question based on the given passages. "
            "Only give me the answer and do not output any other words.\n\n"
            "The following are given passages.\n"
        ),
        "query_prompt": (
            "\n\nAnswer the question based on the given passages. "
            "Answer the question within 5 words. Do NOT repeat the question or output any other words. "
            "Question: "
        ),
    },
    "squad": {
        "dataset_path": os.path.join(ROOT_DIR, "inputs", "squad_s.json"),
        "prefix_prompt": (
            "Answer the question based on the given passage. "
            "Only output the short answer phrase.\n\n"
            "Passage:\n"
        ),
        "query_prompt": (
            "\n\nAnswer the question based on the given passage. "
            "Only output the short answer phrase. "
            "Question: "
        ),
    },
}


@dataclass
class VisualizationBundle:
    dataset: str
    sample_idx: int
    question: str
    answers: List[str]
    prompt_token_ids: List[int]
    query_token_ids: List[int]
    query_prompt_positions: List[int]
    scores: torch.Tensor
    selected_indices: torch.Tensor


def find_subsequence_positions(full_ids: Sequence[int], sub_ids: Sequence[int]) -> List[int]:
    """返回 sub_ids 在 full_ids 中最后一次连续出现的位置。"""
    if not full_ids or not sub_ids or len(sub_ids) > len(full_ids):
        return []
    last_start = -1
    limit = len(full_ids) - len(sub_ids) + 1
    for start in range(limit):
        if list(full_ids[start:start + len(sub_ids)]) == list(sub_ids):
            last_start = start
    if last_start < 0:
        return []
    return list(range(last_start, last_start + len(sub_ids)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate query-aware visualizations")
    parser.add_argument("--dataset", choices=sorted(DATASET_SPECS.keys()), required=True)
    parser.add_argument("--sample-idx", type=int, default=1, help="1-based sample index")
    parser.add_argument("--recomp-ratio", type=float, default=0.7)
    parser.add_argument("--suffix-len", type=int, default=32)
    parser.add_argument("--query-text", default="", help="override query text")
    parser.add_argument(
        "--output-dir",
        default=os.path.join(ROOT_DIR, "graduation_project", "analyse", "selection_visuals"),
    )
    parser.add_argument(
        "--heatmap-topk",
        type=int,
        default=48,
        help="number of candidate tokens to show in similarity heatmap",
    )
    parser.add_argument(
        "--score-topk-labels",
        type=int,
        default=12,
        help="number of top selected tokens to annotate in score plot",
    )
    parser.add_argument(
        "--text-window",
        type=int,
        default=96,
        help="number of top-scoring candidate tokens to render in annotated text view",
    )
    parser.add_argument(
        "--query-topk",
        type=int,
        default=12,
        help="number of query tokens to show on heatmap x-axis",
    )
    parser.add_argument(
        "--with-attention",
        action="store_true",
        help="also export last-layer attention heatmap for query tail -> selected tokens",
    )
    parser.add_argument(
        "--attention-layer",
        type=int,
        default=-1,
        help="attention layer index, default last layer",
    )
    return parser.parse_args()


def load_dataset_example(dataset_name: str, sample_idx: int) -> Dict:
    spec = DATASET_SPECS[dataset_name]
    with open(spec["dataset_path"], "r", encoding="utf-8") as f:
        data = json.load(f)
    if sample_idx < 1 or sample_idx > len(data):
        raise IndexError(f"sample_idx={sample_idx} 超出数据集范围 1..{len(data)}")
    return data[sample_idx - 1]


def build_final_prompt(dataset_name: str, example: Dict) -> Tuple[str, str, List[str]]:
    spec = DATASET_SPECS[dataset_name]
    doc_prompts, q_prompt = build_qa_prompt(example, spec["query_prompt"])
    final_prompt = spec["prefix_prompt"] + "".join(doc_prompts) + q_prompt
    question = normalize_question(example.get("question", ""))
    answers = list(example.get("answers", []))
    return final_prompt, question, answers


def decode_token_labels(tokenizer, token_ids: Sequence[int]) -> List[str]:
    labels = []
    for tid in token_ids:
        token = tokenizer.convert_ids_to_tokens(int(tid))
        if token is None:
            token = str(tid)
        token = token.replace("▁", " ")
        token = token.replace("Ġ", " ")
        token = token.replace("\n", "\\n")
        labels.append(token)
    return labels


def compute_query_aware_scores(
    model_runner: YiModelRunner,
    prompt_token_ids: List[int],
    query_token_ids: List[int],
    overlap_len: int,
) -> torch.Tensor:
    if overlap_len <= 0:
        return torch.zeros(0, device=model_runner.device)
    if len(query_token_ids) == 0:
        return torch.zeros(overlap_len, device=model_runner.device)

    cand_emb = model_runner.lookup_token_embeddings(prompt_token_ids[:overlap_len])
    query_emb = model_runner.lookup_token_embeddings(query_token_ids)
    cand_emb = torch.nn.functional.normalize(cand_emb, dim=-1)
    query_emb = torch.nn.functional.normalize(query_emb, dim=-1)
    sim = cand_emb @ query_emb.transpose(0, 1)
    return sim.max(dim=1).values


def select_token_indices(
    scores: torch.Tensor,
    new_len: int,
    overlap_len: int,
    recomp_ratio: float,
    suffix_len: int,
) -> torch.Tensor:
    force_start = max(0, new_len - max(suffix_len, 0))
    added = set(range(overlap_len, new_len))
    forced = set(range(force_start, new_len))
    candidate_end = min(overlap_len, force_start)

    topk = set()
    if candidate_end > 0 and recomp_ratio > 0:
        topk_num = max(1, int(candidate_end * recomp_ratio))
        topk_num = min(topk_num, candidate_end)
        top_indices = torch.topk(scores[:candidate_end], k=topk_num).indices
        topk = {int(i) for i in top_indices.tolist()}

    selected = sorted(added | forced | topk)
    if not selected and new_len > 0:
        selected = [new_len - 1]
    return torch.tensor(selected, dtype=torch.long, device=scores.device)


def prepare_visualization_bundle(
    model_runner: YiModelRunner,
    dataset_name: str,
    sample_idx: int,
    recomp_ratio: float,
    suffix_len: int,
    query_text_override: str,
) -> VisualizationBundle:
    example = load_dataset_example(dataset_name, sample_idx)
    final_prompt, question, answers = build_final_prompt(dataset_name, example)
    prompt_token_ids = model_runner.encode(final_prompt)

    query_text = (query_text_override or question).strip()
    if query_text:
        query_token_ids = model_runner.encode_no_special(query_text)
    else:
        tail_start = max(0, len(prompt_token_ids) - max(suffix_len, 1))
        query_token_ids = prompt_token_ids[tail_start:]

    overlap_len = len(prompt_token_ids)
    scores = compute_query_aware_scores(
        model_runner=model_runner,
        prompt_token_ids=prompt_token_ids,
        query_token_ids=query_token_ids,
        overlap_len=overlap_len,
    )
    selected_indices = select_token_indices(
        scores=scores,
        new_len=len(prompt_token_ids),
        overlap_len=overlap_len,
        recomp_ratio=recomp_ratio,
        suffix_len=suffix_len,
    )

    query_prompt_positions = find_subsequence_positions(prompt_token_ids, query_token_ids)
    if not query_prompt_positions:
        query_prompt_positions = list(
            range(max(0, len(prompt_token_ids) - min(len(query_token_ids), 12)),
                  len(prompt_token_ids))
        )

    return VisualizationBundle(
        dataset=dataset_name,
        sample_idx=sample_idx,
        question=question,
        answers=answers,
        prompt_token_ids=prompt_token_ids,
        query_token_ids=query_token_ids,
        query_prompt_positions=query_prompt_positions,
        scores=scores.detach().cpu(),
        selected_indices=selected_indices.detach().cpu(),
    )


def make_output_prefix(bundle: VisualizationBundle, output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    return os.path.join(
        output_dir,
        f"{bundle.dataset}_sample{bundle.sample_idx:03d}_qaw"
    )


def plot_similarity_heatmap(
    model_runner: YiModelRunner,
    bundle: VisualizationBundle,
    output_prefix: str,
    heatmap_topk: int,
    query_topk: int,
) -> str:
    overlap_len = len(bundle.scores)
    query_ids = bundle.query_token_ids
    top_queryk = min(len(query_ids), max(1, query_topk))
    query_ids_show = query_ids[-top_queryk:]

    top_candidates = torch.topk(bundle.scores[:overlap_len], k=min(heatmap_topk, overlap_len)).indices
    top_candidates, _ = torch.sort(top_candidates)

    cand_emb = model_runner.lookup_token_embeddings(
        [bundle.prompt_token_ids[int(i)] for i in top_candidates.tolist()]
    )
    query_emb = model_runner.lookup_token_embeddings(query_ids_show)
    cand_emb = torch.nn.functional.normalize(cand_emb, dim=-1)
    query_emb = torch.nn.functional.normalize(query_emb, dim=-1)
    sim = (cand_emb @ query_emb.transpose(0, 1)).detach().cpu().numpy()

    fig_h = max(6, 0.24 * sim.shape[0])
    fig, ax = plt.subplots(figsize=(8.5, fig_h))
    im = ax.imshow(sim, aspect="auto", cmap="YlGnBu", interpolation="nearest")

    cand_labels = decode_token_labels(
        model_runner.tokenizer,
        [bundle.prompt_token_ids[int(i)] for i in top_candidates.tolist()],
    )
    cand_labels = [f"{int(pos)}:{tok}" for pos, tok in zip(top_candidates.tolist(), cand_labels)]
    query_labels = decode_token_labels(model_runner.tokenizer, query_ids_show)

    ax.set_xticks(range(len(query_labels)))
    ax.set_xticklabels(query_labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(cand_labels)))
    ax.set_yticklabels(cand_labels, fontsize=8)
    ax.set_title("Query-Token Similarity Heatmap")
    ax.set_xlabel("Query Tokens")
    ax.set_ylabel("Candidate Prompt Tokens (Top Score)")
    fig.colorbar(im, ax=ax, shrink=0.85, label="Cosine Similarity")
    fig.tight_layout()

    output_path = f"{output_prefix}_similarity_heatmap.png"
    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_score_and_mask(
    model_runner: YiModelRunner,
    bundle: VisualizationBundle,
    output_prefix: str,
    score_topk_labels: int,
    suffix_len: int,
) -> str:
    positions = list(range(len(bundle.prompt_token_ids)))
    scores = bundle.scores.numpy()
    selected = set(int(i) for i in bundle.selected_indices.tolist())
    suffix_start = max(0, len(bundle.prompt_token_ids) - max(suffix_len, 0))

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 6.5), sharex=True,
        gridspec_kw={"height_ratios": [3, 1]}
    )
    ax1.plot(positions[: len(scores)], scores, color="#1b9e77", linewidth=1.2)
    ax1.axvspan(suffix_start, len(bundle.prompt_token_ids), color="#fddbc7", alpha=0.4, label="Forced suffix")

    selected_non_suffix = sorted([i for i in selected if i < len(scores)])
    if selected_non_suffix:
        ax1.scatter(selected_non_suffix, scores[selected_non_suffix], color="#d95f02", s=16, label="Selected")

    label_count = min(score_topk_labels, len(scores))
    if label_count > 0:
        top_idx = torch.topk(bundle.scores, k=label_count).indices.tolist()
        token_labels = decode_token_labels(
            model_runner.tokenizer,
            [bundle.prompt_token_ids[int(i)] for i in top_idx],
        )
        for pos, tok in zip(top_idx, token_labels):
            ax1.annotate(
                tok.strip() or "<ws>",
                xy=(pos, scores[pos]),
                xytext=(0, 8),
                textcoords="offset points",
                fontsize=8,
                rotation=25,
                ha="left",
            )

    mask_values = [1 if i in selected else 0 for i in positions]
    ax2.bar(positions, mask_values, color=["#d95f02" if v else "#bdbdbd" for v in mask_values], width=1.0)
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(["reuse", "selected"])
    ax2.set_xlabel("Prompt Token Position")
    ax1.set_ylabel("Query-aware Score")
    ax2.set_ylabel("Mask")
    ax1.set_title("Token Score Distribution and Selected Mask")
    ax1.legend(loc="upper left")
    fig.tight_layout()

    output_path = f"{output_prefix}_score_mask.png"
    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _normalize_for_color(values: Sequence[float]) -> List[Tuple[float, float, float, float]]:
    if not values:
        return []
    vmin = min(values)
    vmax = max(values)
    if math.isclose(vmin, vmax):
        vmax = vmin + 1e-6
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap("YlOrRd")
    return [cmap(norm(v)) for v in values]


def plot_annotated_text(
    model_runner: YiModelRunner,
    bundle: VisualizationBundle,
    output_prefix: str,
    text_window: int,
) -> str:
    window_count = min(text_window, len(bundle.scores))
    top_idx = torch.topk(bundle.scores, k=window_count).indices.tolist()
    top_idx = sorted(top_idx)
    selected = set(int(i) for i in bundle.selected_indices.tolist())

    token_ids = [bundle.prompt_token_ids[i] for i in top_idx]
    token_texts = decode_token_labels(model_runner.tokenizer, token_ids)
    token_scores = [float(bundle.scores[i]) for i in top_idx]
    token_colors = _normalize_for_color(token_scores)

    per_line = 8
    line_count = math.ceil(len(top_idx) / per_line)
    fig_h = max(4.5, 0.8 * line_count)
    fig, ax = plt.subplots(figsize=(16, fig_h))
    ax.axis("off")

    x_step = 0.12
    y = 1.0
    for start in range(0, len(top_idx), per_line):
        chunk_idx = top_idx[start:start + per_line]
        chunk_text = token_texts[start:start + per_line]
        chunk_scores = token_scores[start:start + per_line]
        chunk_colors = token_colors[start:start + per_line]
        x = 0.01
        for pos, tok, score, color in zip(chunk_idx, chunk_text, chunk_scores, chunk_colors):
            tok_show = tok.strip() or "<ws>"
            edge = "#1b1b1b" if pos in selected else "#777777"
            lw = 1.6 if pos in selected else 0.6
            label = f"{pos}:{tok_show}\n{score:.3f}"
            ax.text(
                x,
                y,
                label,
                transform=ax.transAxes,
                fontsize=9,
                va="top",
                ha="left",
                bbox={
                    "boxstyle": "round,pad=0.25",
                    "facecolor": color,
                    "edgecolor": edge,
                    "linewidth": lw,
                },
            )
            x += x_step
        y -= 0.13

    ax.set_title(
        "Annotated Prompt Tokens (Top-Scoring Window)\n"
        "Bordered boxes indicate selected tokens; fill color indicates query-aware score.",
        fontsize=12,
        pad=12,
    )
    fig.tight_layout()

    output_path = f"{output_prefix}_annotated_text.png"
    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_attention_heatmap(
    model_runner: YiModelRunner,
    bundle: VisualizationBundle,
    output_prefix: str,
    attention_layer: int,
    query_topk: int,
    heatmap_topk: int,
) -> str:
    input_ids = torch.tensor([bundle.prompt_token_ids], dtype=torch.long, device=model_runner.device)
    attn_mask = torch.ones((1, len(bundle.prompt_token_ids)), dtype=torch.long, device=model_runner.device)
    outputs = model_runner.model(
        input_ids=input_ids,
        attention_mask=attn_mask,
        use_cache=False,
        output_attentions=True,
        return_dict=True,
    )

    layer_idx = attention_layer
    attn_tensor = outputs.attentions[layer_idx][0]  # [heads, seq, seq]
    attn_mean = attn_tensor.mean(dim=0).detach().cpu()  # [seq, seq]

    query_positions = bundle.query_prompt_positions[-max(1, query_topk):]
    selected_positions = [int(i) for i in bundle.selected_indices.tolist() if i < len(bundle.prompt_token_ids)]
    selected_positions = selected_positions[:heatmap_topk]
    if not selected_positions:
        selected_positions = list(range(max(0, len(bundle.prompt_token_ids) - heatmap_topk), len(bundle.prompt_token_ids)))

    mat = attn_mean[query_positions][:, selected_positions].numpy()
    fig_h = max(4.2, 0.35 * len(query_positions))
    fig, ax = plt.subplots(figsize=(9.5, fig_h))
    im = ax.imshow(mat, aspect="auto", cmap="magma", interpolation="nearest")

    query_labels = decode_token_labels(
        model_runner.tokenizer,
        [bundle.prompt_token_ids[i] for i in query_positions],
    )
    cand_labels = decode_token_labels(
        model_runner.tokenizer,
        [bundle.prompt_token_ids[i] for i in selected_positions],
    )
    cand_labels = [f"{pos}:{tok}" for pos, tok in zip(selected_positions, cand_labels)]

    ax.set_yticks(range(len(query_labels)))
    ax.set_yticklabels(query_labels, fontsize=9)
    ax.set_xticks(range(len(cand_labels)))
    ax.set_xticklabels(cand_labels, rotation=45, ha="right", fontsize=8)
    ax.set_title("Last-Layer Attention: Query Tail -> Selected Tokens")
    ax.set_xlabel("Selected / Visualized Prompt Tokens")
    ax.set_ylabel("Query Tail Tokens")
    fig.colorbar(im, ax=ax, shrink=0.85, label="Attention Weight")
    fig.tight_layout()

    output_path = f"{output_prefix}_attention_heatmap.png"
    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return output_path


def write_manifest(bundle: VisualizationBundle, output_prefix: str, figure_paths: List[str]) -> str:
    manifest_path = f"{output_prefix}_manifest.json"
    payload = {
        "dataset": bundle.dataset,
        "sample_idx": bundle.sample_idx,
        "question": bundle.question,
        "answers": bundle.answers,
        "prompt_token_count": len(bundle.prompt_token_ids),
        "query_token_count": len(bundle.query_token_ids),
        "selected_token_count": int(bundle.selected_indices.numel()),
        "selected_indices": [int(i) for i in bundle.selected_indices.tolist()],
        "figure_paths": figure_paths,
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return manifest_path


def main() -> None:
    args = parse_args()
    cfg = RuntimeConfig()

    logger = setup_logger("qaw_visualize", cfg.log_level)
    logger.disabled = True
    model_runner = YiModelRunner(cfg.model_name, cfg.device, cfg.model_dtype, logger)

    bundle = prepare_visualization_bundle(
        model_runner=model_runner,
        dataset_name=args.dataset,
        sample_idx=args.sample_idx,
        recomp_ratio=args.recomp_ratio,
        suffix_len=args.suffix_len,
        query_text_override=args.query_text,
    )

    output_prefix = make_output_prefix(bundle, args.output_dir)
    figure_paths = [
        plot_similarity_heatmap(
            model_runner=model_runner,
            bundle=bundle,
            output_prefix=output_prefix,
            heatmap_topk=args.heatmap_topk,
            query_topk=args.query_topk,
        ),
        plot_score_and_mask(
            model_runner=model_runner,
            bundle=bundle,
            output_prefix=output_prefix,
            score_topk_labels=args.score_topk_labels,
            suffix_len=args.suffix_len,
        ),
        plot_annotated_text(
            model_runner=model_runner,
            bundle=bundle,
            output_prefix=output_prefix,
            text_window=args.text_window,
        ),
    ]

    if args.with_attention:
        figure_paths.append(
            plot_attention_heatmap(
                model_runner=model_runner,
                bundle=bundle,
                output_prefix=output_prefix,
                attention_layer=args.attention_layer,
                query_topk=args.query_topk,
                heatmap_topk=args.heatmap_topk,
            )
        )

    manifest_path = write_manifest(bundle, output_prefix, figure_paths)
    print(f"saved manifest: {manifest_path}")
    for path in figure_paths:
        print(f"saved figure: {path}")


if __name__ == "__main__":
    main()
