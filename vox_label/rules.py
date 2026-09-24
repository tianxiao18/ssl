"""Decision rules V(z) -- the ways a detection pattern can be turned into an accept.

A benchmark is only interesting relative to some way of *using* the detectors, so
every reported quantity is attached to a rule V, where V(z) = 1 means "accept a
candidate whose detection pattern is z". Isolating detector k is the rule 1_k
(eq. 1); ensembles and disagreements are just other functions of the same z.

The point of routing everything through named rules is that a single uniform
annotation pass serves all of them at once: each annotated candidate contributes to
the precision stream of every rule that accepts it and, if genuine, to the recall
stream of every rule. Nothing has to be decided in advance about which rules will be
reported -- adding one later costs no annotations, only arithmetic.
"""


def fired(detector, detectors):
    """1_k of eq. 1: accept iff this detector fired, whatever the others did."""
    i = detectors.index(detector)
    return lambda z: bool(z[i])


def at_least(k):
    """Accept iff at least k detectors fired."""
    return lambda z: sum(z) >= k


def unanimous(detectors):
    return lambda z: all(z)


def disagreement(a, b, detectors):
    """V(z) = 1_a(z) * (1 - 1_b(z)): candidates a flagged and b did not.

    The precision of this rule answers "of the candidates a caught but b missed, what
    fraction are real?" -- which is the question you actually want when deciding
    whether adding a detector to an ensemble buys anything.
    """
    ia, ib = detectors.index(a), detectors.index(b)
    return lambda z: bool(z[ia]) and not bool(z[ib])


def resolve(name, detectors):
    """Build a rule from its campaign-spec name. Kept string-addressable so a frozen
    campaign names its rules and the report reproduces them exactly."""
    if name in detectors:
        return fired(name, detectors)
    if name == "unanimous":
        return unanimous(detectors)
    if name.startswith("atleast_"):
        return at_least(int(name.split("_", 1)[1]))
    if "_not_" in name:
        a, b = name.split("_not_", 1)
        if a in detectors and b in detectors:
            return disagreement(a, b, detectors)
    raise ValueError(f"unknown rule {name!r}")


def default_rules(detectors):
    """Each detector alone, plus the ensembles worth reporting for K detectors."""
    rules = list(detectors)
    rules += [f"atleast_{k}" for k in range(2, len(detectors))]
    rules.append("unanimous")
    return rules


def accept_fraction(rule_fn, rows):
    """f_V: the fraction of the pool this rule accepts.

    Computable exactly before any annotation exists, because V depends only on z. This
    is what lets a precision stream's checkpoint grid be committed up front (section 5:
    f is "the fraction of annotations reaching the stream in question").
    """
    if not rows:
        return 0.0
    return sum(1 for r in rows if rule_fn(r["z"])) / len(rows)
