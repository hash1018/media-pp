"""Scores what `cut_detect` printed for a film against its hand-marked cuts.

    python score.py tears-of-steel.txt tos.out [sintel.txt sintel.out ...]

Each marked line is `picture label`: C a cut, found within one picture of
it; F the start of a fade back from black, found within 24; B a cut into
black, which a detector need not find and is not held against it for
finding. Any other cut found is a false one.
"""
import re
import sys


def marked(path):
    return [(int(n), label) for n, label in (line.split() for line in open(path) if line.strip())]


def found(path):
    return [int(m.group(1)) for m in re.finditer(r"cut at picture (\d+)", open(path).read())]


def score(cuts, truth):
    used, hits, false = set(), 0, 0
    for cut in cuts:
        match = next(
            (
                i
                for i, (n, label) in enumerate(truth)
                if i not in used and abs(cut - n) <= {"C": 1, "F": 24, "B": 2}[label]
            ),
            None,
        )
        if match is None:
            false += 1
        else:
            used.add(match)
            hits += truth[match][1] != "B"
    missed = sum(1 for i, (_, label) in enumerate(truth) if label != "B" and i not in used)
    return hits, false, missed


totals = [0, 0, 0]
for truth_path, out_path in zip(sys.argv[1::2], sys.argv[2::2]):
    hits, false, missed = score(found(out_path), marked(truth_path))
    totals = [a + b for a, b in zip(totals, (hits, false, missed))]
    print(f"{truth_path}: found {hits}, false {false}, missed {missed}")
hits, false, missed = totals
print(
    f"precision {hits / max(hits + false, 1):.3f}, recall {hits / max(hits + missed, 1):.3f}"
)
