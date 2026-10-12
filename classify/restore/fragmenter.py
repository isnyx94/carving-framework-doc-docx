"""Fragmented-image generation for the restoration experiments.

Deliberately separate from generate_three_fragmented_images.py at the repo
root: that script writes a byte-per-row ground-truth CSV, which produced the
13-30 GB CSVs sitting next to the .bin images and makes an experiment
effectively unrepeatable. Ground truth here is block-level and held in memory
(one row per block, not per byte), which is all any block-based reassembly
evaluation can use anyway.

Fragmentation model: each source file is cut into `n_fragments` contiguous
runs, the runs of all files are shuffled together, and noise blocks are
interleaved at the requested rate. That is the same model as the original
script, so severity levels stay comparable with the earlier reassembly work.
"""

import os
import random
from dataclasses import dataclass, field

BLOCK_SIZE = 512

# Matches the severity tiers used by the existing reassembly track
# (INDEX.md "Three Fragmentation Levels") so results are comparable.
SEVERITIES = {
    "basic": {"fragments": (2, 4), "noise_rate": 0.05},
    "moderate": {"fragments": (4, 8), "noise_rate": 0.20},
    "severe": {"fragments": (10, 20), "noise_rate": 0.40},
}


@dataclass
class BlockTruth:
    """Which source file and byte offset each image block came from."""
    source: str  # file path, or "NOISE"
    file_offset: int  # byte offset within the source file; -1 for noise


@dataclass
class FragmentedImage:
    data: bytearray = field(default_factory=bytearray)
    truth: list = field(default_factory=list)  # index = block number
    block_size: int = BLOCK_SIZE

    def block(self, i: int) -> bytes:
        return bytes(self.data[i * self.block_size:(i + 1) * self.block_size])

    @property
    def n_blocks(self) -> int:
        return len(self.truth)

    def blocks_of(self, source: str) -> list:
        return [i for i, t in enumerate(self.truth) if t.source == source]


def _split_into_runs(n_blocks: int, n_fragments: int, rng: random.Random) -> list:
    """Cut range(n_blocks) into n_fragments contiguous runs at random points."""
    n_fragments = max(1, min(n_fragments, n_blocks))
    if n_fragments == 1:
        return [(0, n_blocks)]
    cuts = sorted(rng.sample(range(1, n_blocks), n_fragments - 1))
    bounds = [0] + cuts + [n_blocks]
    return [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]


def build_image(paths, severity: str, seed: int = 0,
                block_size: int = BLOCK_SIZE) -> FragmentedImage:
    """Fragment every file in `paths` at the given severity and interleave the
    runs into one image with noise.

    Returns block-level ground truth, so a restorer's output can be scored
    against the true source of every block without a per-byte table.
    """
    if severity not in SEVERITIES:
        raise ValueError(f"unknown severity {severity!r}; known: {sorted(SEVERITIES)}")
    config = SEVERITIES[severity]
    rng = random.Random(seed)

    runs = []  # (source_path, start_block, end_block, file_bytes)
    for path in paths:
        with open(path, "rb") as f:
            payload = f.read()
        n_blocks = max(1, (len(payload) + block_size - 1) // block_size)
        n_fragments = rng.randint(*config["fragments"])
        for start, end in _split_into_runs(n_blocks, n_fragments, rng):
            runs.append((path, start, end, payload))

    rng.shuffle(runs)

    image = FragmentedImage(block_size=block_size)
    n_noise = int(len(runs) * config["noise_rate"])
    noise_positions = set(rng.sample(range(len(runs) + n_noise), n_noise)) if n_noise else set()

    run_iter = iter(runs)
    for slot in range(len(runs) + n_noise):
        if slot in noise_positions:
            image.data += bytes(rng.getrandbits(8) for _ in range(block_size))
            image.truth.append(BlockTruth("NOISE", -1))
            continue
        try:
            path, start, end, payload = next(run_iter)
        except StopIteration:
            continue
        for b in range(start, end):
            chunk = payload[b * block_size:(b + 1) * block_size]
            if len(chunk) < block_size:
                chunk = chunk + bytes(block_size - len(chunk))
            image.data += chunk
            image.truth.append(BlockTruth(path, b * block_size))
    return image


# --------------------------------------------------------------------------
# realistic (locality-preserving) fragmentation
# --------------------------------------------------------------------------

# Fragments-per-file distributions, and how many files are being written at once.
#
# The shape matters more than the exact values: in real corpora most files are
# *contiguous*, and among fragmented files two fragments dominate (Garfinkel's
# bifragmentation observation — verify the citation before quoting it). An
# interleaving model with a fixed small chunk size instead splits every file into
# dozens or hundreds of runs (measured: 47-419 runs/file), which is not a harder
# version of reality but a different and unrealistic regime.
REALISTIC_SEVERITIES = {
    "basic": {"concurrent": 2, "fragments": {1: 0.90, 2: 0.10}},
    "moderate": {"concurrent": 3, "fragments": {1: 0.55, 2: 0.30, 3: 0.10, 4: 0.05}},
    "severe": {"concurrent": 5, "fragments": {1: 0.15, 2: 0.35, 3: 0.25, 4: 0.15, 6: 0.10}},
}


def _draw_fragment_count(dist: dict, rng: random.Random) -> int:
    r = rng.random()
    cumulative = 0.0
    for n, p in sorted(dist.items()):
        cumulative += p
        if r <= cumulative:
            return n
    return max(dist)


def build_image_realistic(paths, severity: str, seed: int = 0,
                          block_size: int = BLOCK_SIZE) -> FragmentedImage:
    """Fragmentation produced by *interleaved writes* rather than global shuffling.

    Why a second model exists
    -------------------------
    `build_image` shuffles every run of every file uniformly across the image.
    That is a worst case, not a realistic one, and it flatters any method that
    searches globally for continuations: a bounded search window recovers nothing
    under it (measured), so the whole B2 advantage would rest on an unrealistic
    assumption if it were the only model reported.

    Real filesystems fragment mainly because several files are written
    concurrently and the allocator interleaves their extents. That produces three
    properties this model reproduces and the shuffled model destroys:

      1. **In-order fragments** — a file's runs appear in increasing file-offset
         order on disk, because allocators append forward.
      2. **Locality** — consecutive runs of one file are separated by a bounded
         gap (the other writers' turns), not by the whole image.
      3. **Gaps hold real data** — the bytes between two runs are another
         document's content, not random noise, so a carver cannot use
         "looks like noise" as a boundary cue.

    Property 3 makes this model *harder* than the shuffled one in one specific
    way: a wrong continuation is real compressed document data, which is exactly
    the case the plausibility gate cannot detect and only the CRC can.
    """
    if severity not in REALISTIC_SEVERITIES:
        raise ValueError(f"unknown severity {severity!r}; "
                         f"known: {sorted(REALISTIC_SEVERITIES)}")
    config = REALISTIC_SEVERITIES[severity]
    rng = random.Random(seed)

    # Each writer holds one file pre-split into its (in-order) runs; a turn emits
    # one whole run. Because a file's fragment count is drawn up front, run counts
    # stay realistic, while the gap between a file's consecutive runs is whatever
    # the other active writers emit in between — bounded and always forward.
    pending = list(paths)
    rng.shuffle(pending)

    def _open_next():
        while pending:
            path = pending.pop()
            with open(path, "rb") as f:
                payload = f.read()
            n_blocks = max(1, (len(payload) + block_size - 1) // block_size)
            n_frags = min(_draw_fragment_count(config["fragments"], rng), n_blocks)
            runs = _split_into_runs(n_blocks, n_frags, rng)
            return {"path": path, "payload": payload, "runs": runs, "next_run": 0}
        return None

    writers = []
    for _ in range(config["concurrent"]):
        w = _open_next()
        if w is None:
            break
        writers.append(w)

    image = FragmentedImage(block_size=block_size)
    turn = 0
    while writers:
        # Round-robin, not random choice. Under random selection one writer can
        # finish several whole files while another waits, so the gap between one
        # file's consecutive runs grows to thousands of blocks (measured mean
        # 9085) — which destroys the locality this model exists to represent.
        # Rotating bounds the gap to roughly one turn from each other writer.
        idx = turn % len(writers)
        turn += 1
        if len(writers) > 1 and rng.random() < 0.15:
            continue  # occasional skipped turn, for gap variability
        w = writers[idx]
        start, end = w["runs"][w["next_run"]]
        for b in range(start, end):
            data = w["payload"][b * block_size:(b + 1) * block_size]
            if len(data) < block_size:
                data = data + bytes(block_size - len(data))
            image.data += data
            image.truth.append(BlockTruth(w["path"], b * block_size))
        w["next_run"] += 1

        if w["next_run"] >= len(w["runs"]):
            replacement = _open_next()
            if replacement is None:
                writers.pop(idx)
            else:
                writers[idx] = replacement
    return image


def fragmentation_stats(image: FragmentedImage) -> dict:
    """Descriptive statistics of an image's actual fragmentation, so the two
    models can be compared on what they produced rather than on their
    parameters: runs per file, and the block gap between consecutive runs."""
    per_file = {}
    for i, t in enumerate(image.truth):
        if t.source == "NOISE":
            continue
        per_file.setdefault(t.source, []).append((i, t.file_offset))

    runs_per_file = []
    gaps = []
    in_order = 0
    total = 0
    for blocks in per_file.values():
        blocks.sort(key=lambda p: p[1])  # by file offset
        runs = 1
        for (prev_i, _), (cur_i, _) in zip(blocks, blocks[1:]):
            if cur_i != prev_i + 1:
                runs += 1
                gaps.append(cur_i - prev_i)
            total += 1
            if cur_i > prev_i:
                in_order += 1
        runs_per_file.append(runs)

    def _median(xs):
        if not xs:
            return 0.0
        s = sorted(xs)
        return float(s[len(s) // 2])

    return {
        "n_files": len(per_file),
        "mean_runs_per_file": (sum(runs_per_file) / len(runs_per_file)) if runs_per_file else 0.0,
        "median_runs_per_file": _median(runs_per_file),
        "frac_fragmented": (sum(1 for r in runs_per_file if r > 1) / len(runs_per_file))
        if runs_per_file else 0.0,
        "median_gap_blocks": _median(gaps),
        "mean_gap_blocks": (sum(gaps) / len(gaps)) if gaps else 0.0,
        "frac_successive_in_order": (in_order / total) if total else 0.0,
        "n_blocks": image.n_blocks,
    }


def corpus_paths(root: str, limit: int = None, pattern: str = ".docx") -> list:
    paths = sorted(os.path.join(root, f) for f in os.listdir(root)
                   if f.lower().endswith(pattern))
    return paths[:limit] if limit else paths
