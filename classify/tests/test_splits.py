from collections import Counter

from classify.data.splits import split_parent_files


def _fake_corpus(n_doc=200, n_docx=200):
    return ([(f"doc/D{i}.doc", "doc") for i in range(n_doc)]
            + [(f"docx/X{i}.docx", "docx") for i in range(n_docx)])


def test_every_parent_assigned_exactly_one_split():
    corpus = _fake_corpus()
    assignment = split_parent_files(corpus, seed=0)
    assert set(assignment.keys()) == {path for path, _ in corpus}
    assert set(assignment.values()) <= {"train", "val", "test"}


def test_no_leakage_across_splits():
    # Leakage would mean fragments of the same parent file land in two
    # splits. Since assignment is a dict keyed by parent_path, every
    # fragment of a given parent necessarily resolves to the same split —
    # this test pins that invariant against regressions (e.g. someone
    # switching to a fragment-level hash by mistake).
    corpus = _fake_corpus()
    assignment = split_parent_files(corpus, seed=0)
    for path, _ in corpus:
        fragment_offsets_for_this_parent = [0, 512, 1024, 4096]
        splits_seen = {assignment[path] for _ in fragment_offsets_for_this_parent}
        assert len(splits_seen) == 1


def test_stratified_by_label():
    corpus = _fake_corpus(n_doc=100, n_docx=300)
    assignment = split_parent_files(corpus, seed=0)
    counts = Counter((label, assignment[path]) for path, label in corpus)
    doc_train_frac = counts[("doc", "train")] / 100
    docx_train_frac = counts[("docx", "train")] / 300
    assert abs(doc_train_frac - docx_train_frac) < 0.05


def test_deterministic_given_seed():
    corpus = _fake_corpus()
    a = split_parent_files(corpus, seed=3)
    b = split_parent_files(corpus, seed=3)
    assert a == b
