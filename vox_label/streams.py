"""Turn a sequence of annotations into the Bernoulli streams the intervals are built on.

Section 2 of the report: precision, relative recall and F1 are all a single Bernoulli
rate p estimated from an i.i.d. stream of binary observations. So there is exactly one
statistical problem, solved once in `exact_grid`, and this module's only job is to
extract the right stream for a given rule.

For a rule V, from the annotated candidates in annotation order:

    precision stream : candidates with V(z) = 1,          recording x       (eq. 4)
    recall stream    : candidates with x = 1,             recording V(z)    (eq. 5)
    jaccard stream   : candidates with V(z) = 1 or x = 1, recording both

Precision is P(x = 1 | V(z) = 1) and relative recall is P(V(z) = 1 | x = 1), so each is
an ordinary sample proportion of its own stream. F1 = 2r/(1+r) for the Jaccard rate r
of eq. 7, which is why the third stream exists.

Two consequences worth keeping in mind. The streams are shorter than the annotation
budget, and by different amounts -- a selective rule buys its precision estimate from
only a slice of the budget, which is the dominant term in what a target costs. And
recall here is *relative* to the candidate pool: a genuine call that no detector fired
on produces no candidate at all, so eq. 5 answers "of the calls at least one detector
found, what fraction does V keep?" and not "of the calls that occurred".
"""


def precision_stream(annotations, rule_fn):
    """[x for annotated candidates the rule accepts] (eq. 4)."""
    return [a["label"] for a in annotations if rule_fn(a["z"])]


def recall_stream(annotations, rule_fn):
    """[V(z) for annotated candidates that are genuine] (eq. 5)."""
    return [int(rule_fn(a["z"])) for a in annotations if a["label"] == 1]


def jaccard_stream(annotations, rule_fn):
    """[accepted and real, over candidates accepted or real] -- the r_V of eq. 7."""
    out = []
    for a in annotations:
        v, x = int(rule_fn(a["z"])), a["label"]
        if v or x:
            out.append(int(v and x))
    return out


STREAM_KINDS = {
    "precision": precision_stream,
    "recall": recall_stream,
    "f1": jaccard_stream,
}


def counts_at(stream, grid):
    """Cumulative successes at each committed checkpoint the stream has reached.

    Returns (counts, n_crossed). Only the first `n_crossed` checkpoints are usable:
    the interval may be reported at checkpoint j only once the stream is at least
    grid[j] long. This is the enforcement the report asks for, and it lives here so
    that no caller can accidentally read a rate at an uncommitted length.
    """
    counts, crossed = [], 0
    for n_j in grid:
        if len(stream) < n_j:
            break
        counts.append(sum(stream[:n_j]))
        crossed += 1
    return counts, crossed
