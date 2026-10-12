"""Pluggable compound-container adapters — the *framework* interface.

The thesis title promises a carving **framework**, and THESIS_SPINE_v2 Sec 5
draws it as a format-agnostic core plus per-container adapters. This module is
that interface: one abstraction (`CompoundAdapter`) with two instantiations
covering the two structurally unrelated compound technologies named in the
title.

    OPCAdapter   .docx  OPC/ZIP    recovery = partial DEFLATE decompression
    CFBFAdapter  .doc   OLE2/CFBF  recovery = structural decode (FAT/dir/header)

Both satisfy the same three-method contract, so the model in
multi_adapter_model.py is written against the interface and never against a
specific container format. Adding ODF or a further OOXML variant means adding
one adapter, not touching the model — which is the generality claim RQ2 makes
(THESIS_SPINE_v2 Sec 3).

The two adapters differ in a way that matters for the thesis argument: the
OPC adapter's recovery has a *precondition* (the fragment must begin at a
DEFLATE stream start) which uniform sampling rarely satisfies, while the CFBF
adapter's recovery has none, because CFBF stores its structure uncompressed.
That asymmetry is the mechanistic reason to expect different outcomes from the
two adapters, and it is the hypothesis the ablation in cfbf_ablation.py tests.
"""

from classify.modules.c1_compound import cfbf_adapter
from classify.modules.c1_compound.detector import (
    HIGH_ENTROPY_THRESHOLD, deflate_likely, heuristic_decision,
)
from classify.modules.c1_compound.partial_decompress import attempt_partial_decompress


class CompoundAdapter:
    """Contract: detect structure, score how much recovery should be trusted,
    then produce a length-preserving recovered view.

    `recover` MUST return exactly `target_len` bytes and MUST fall back to the
    raw fragment when recovery is impossible — the residual gate can learn to
    ignore a useless view, but it cannot recover from a length mismatch or
    from garbage presented as if it were signal.
    """

    name = "abstract"

    def signals(self, data: bytes) -> dict:
        raise NotImplementedError

    def confidence(self, data: bytes, signals: dict = None) -> float:
        raise NotImplementedError

    def recover(self, data: bytes, target_len: int, signals: dict = None) -> bytes:
        raise NotImplementedError


class OPCAdapter(CompoundAdapter):
    """OPC / ZIP (.docx, .xlsx, .pptx). Wraps the existing C1 front-end so the
    published Chapter 4/5 behaviour is reproduced exactly — this is a
    re-presentation of detector.py + partial_decompress.py behind the adapter
    interface, not a reimplementation."""

    name = "opc"

    def signals(self, data: bytes) -> dict:
        s = deflate_likely(data)
        s["heuristic"] = heuristic_decision(data)
        return s

    def confidence(self, data: bytes, signals: dict = None) -> float:
        s = signals if signals is not None else self.signals(data)
        if s["heuristic"] == "compressed":
            return 1.0
        if s["heuristic"] == "not_compressed":
            return 0.0
        # Ambiguous. Without the CNN refiner (which needs a GPU forward pass and
        # is owned by the model, not the adapter) fall back to a graded entropy
        # score so the adapter stays usable standalone in analysis scripts.
        return float(min(1.0, max(0.0, (s["entropy"] - 4.0) / (HIGH_ENTROPY_THRESHOLD - 4.0))))

    def recover(self, data: bytes, target_len: int, signals: dict = None) -> bytes:
        result = attempt_partial_decompress(data)
        recovered = result["output"]
        if not result["success"] or len(recovered) == 0:
            recovered = data
        if len(recovered) < target_len:
            recovered = recovered + data[len(recovered):target_len]
        return recovered[:target_len]


class CFBFAdapter(CompoundAdapter):
    """OLE2 / CFBF (.doc, .xls, .ppt). Recovery decodes stored structure —
    no decompression precondition, so it applies to any fragment overlapping a
    FAT sector, directory sector, or the container header."""

    name = "cfbf"

    def signals(self, data: bytes) -> dict:
        s = cfbf_adapter.cfbf_signals(data)
        s["role"] = cfbf_adapter.structural_role(data, s)
        return s

    def confidence(self, data: bytes, signals: dict = None) -> float:
        s = signals if signals is not None else self.signals(data)
        return cfbf_adapter.recovery_confidence(data, s)

    def recover(self, data: bytes, target_len: int, signals: dict = None) -> bytes:
        return cfbf_adapter.cfbf_recovered_view(data, target_len)


ADAPTER_REGISTRY = {"opc": OPCAdapter, "cfbf": CFBFAdapter}


def build_adapters(names) -> list:
    """names: iterable of registry keys, e.g. ("opc", "cfbf"). Order is the
    order the model concatenates their views in, so keep it stable across
    runs that are compared to each other."""
    unknown = [n for n in names if n not in ADAPTER_REGISTRY]
    if unknown:
        raise ValueError(f"unknown adapter(s) {unknown}; known: {sorted(ADAPTER_REGISTRY)}")
    return [ADAPTER_REGISTRY[n]() for n in names]
