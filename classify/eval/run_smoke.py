"""Smoke-scale end-to-end pipeline run (P0-P2+P5 wiring check).

Intentionally tiny (few fragments/epochs) so it finishes in under a minute
on a laptop GPU and exists purely to prove the pipeline is wired correctly
end to end. Numbers from this script are NOT methodology-grade results —
see classify/eval/ablation_runner.py docstring and RESEARCH_LOG.md.
"""

import torch

from classify.data.custom_corpus import build_dataset
from classify.eval.ablation_runner import run_ablation, to_markdown_table


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    corpus = build_dataset(seed=0, n_per_size=15)
    summary = run_ablation(
        corpus=corpus,
        fragment_size=512,
        seeds=[0, 1, 2],
        model_kwargs=dict(d_model=64, n_heads=4, n_layers=2, d_ff=128),
        epochs=2,
        batch_size=32,
        lr=1e-3,
        device=device,
        out_dir="experiments/smoke",
    )
    print(to_markdown_table(summary))


if __name__ == "__main__":
    main()
