import os
import tempfile

from classify.data.fragmentation import sample_fragments, SIGNATURE_REGION


def _make_file(size: int) -> str:
    fd, path = tempfile.mkstemp(suffix=".bin")
    with os.fdopen(fd, "wb") as f:
        f.write(bytes(range(256)) * (size // 256 + 1))
        f.truncate(size)
    return path


def test_fragment_sizes_correct():
    path = _make_file(20000)
    try:
        for frag in sample_fragments(path, "docx", seed=1, sizes=(512, 4096), n_per_size=5):
            assert len(frag.data) == frag.fragment_size
            assert frag.offset + frag.fragment_size <= os.path.getsize(path)
    finally:
        os.remove(path)


def test_hardest_variant_excludes_signature_region():
    path = _make_file(20000)
    try:
        for frag in sample_fragments(path, "docx", seed=1, sizes=(512, 4096),
                                      n_per_size=20, hardest=True):
            assert frag.offset >= SIGNATURE_REGION
    finally:
        os.remove(path)


def test_deterministic_given_seed():
    path = _make_file(20000)
    try:
        a = [f.offset for f in sample_fragments(path, "docx", seed=7, n_per_size=10)]
        b = [f.offset for f in sample_fragments(path, "docx", seed=7, n_per_size=10)]
        assert a == b
    finally:
        os.remove(path)


def test_skips_files_smaller_than_fragment_size():
    path = _make_file(300)  # smaller than both 512 and 4096
    try:
        frags = list(sample_fragments(path, "docx", seed=1, sizes=(512, 4096), n_per_size=5))
        assert frags == []
    finally:
        os.remove(path)
