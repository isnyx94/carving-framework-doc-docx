"""Analytic FLOPs estimate for the transformer encoder (Sec 10, "Cost").

Not profiler-measured (no torch.profiler/fvcore dependency added) — this is
the standard closed-form estimate for a transformer encoder layer
(attention QKV+output projections + score/weighted-sum + FFN), multiplied
by n_layers. Treat as an order-of-magnitude figure, not an exact count.
"""


def estimate_transformer_flops(d_model: int, n_layers: int, d_ff: int, seq_len: int) -> int:
    qkv_and_out_proj = 4 * 2 * seq_len * d_model * d_model
    attention_scores_and_weighted_sum = 2 * 2 * seq_len * seq_len * d_model
    ffn = 2 * 2 * seq_len * d_model * d_ff
    per_layer = qkv_and_out_proj + attention_scores_and_weighted_sum + ffn
    return per_layer * n_layers
