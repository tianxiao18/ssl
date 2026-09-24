"""The frozen campaign spec: what was committed to, before any label was collected.

The exact-grid guarantee (eq. 9) holds only if three things were fixed in advance:

1. **The candidate pool.** Candidate formation decides what a candidate *is*, and so
   what precision and recall mean. Pinned here by the pool file's sha256.
2. **The annotation order.** The statistics assume annotated candidates are drawn
   uniformly at random from the pool -- "the one design requirement the statistics
   rest on". Rather than trusting the annotator to sample uniformly, the spec stores
   a seeded permutation and the GUI serves candidates strictly in that order.
3. **The checkpoint grids.** J enters the calibration, so checkpoints cannot be
   appended once annotation is under way. A campaign that reaches n_max without
   converging reports the interval as it stands.

Grids are committed **per stream, in that stream's own observation counts**, which is
exactly the construction of section 4 (n_j counts observations of the stream). The two
kinds are committed differently because we know different things about them:

* A **precision** stream's length is n * f_V, and f_V -- the fraction of the pool the
  rule accepts -- depends only on z, so it is known exactly before any label exists.
  Its grid can be committed tightly.
* A **recall** stream's length is n * P(x = 1), which nobody knows until labels exist.
  Its grid is committed generously. Reaching only the first few checkpoints is fine:
  eq. 10 defines C_j for any j <= J, and calibrating for twelve looks while using six
  is slightly conservative, never invalid. Table 2 is the argument -- going from twelve
  checkpoints to twenty-four costs 9% more annotations, while an invalidated guarantee
  costs everything.
"""
import json
import random
from pathlib import Path

from vox_label.candidates import load_pool, sha256_file
from vox_label.exact_grid import even_grid
from vox_label.rules import accept_fraction, default_rules, resolve


def create(out_dir, pool_csv, detectors, rules=None, alpha=0.05, n_max=6000, J=12,
           seed=20260911, recall_stream_max=None, alpha_per_rule=False, dataset=None):
    """Write campaign.json once. Refuses to overwrite an existing spec."""
    out_dir = Path(out_dir)
    spec_path = out_dir / "campaign.json"
    if spec_path.exists():
        raise FileExistsError(
            f"{spec_path} already exists. A campaign spec is frozen once written -- "
            f"editing it after annotation begins invalidates the coverage guarantee. "
            f"Start a new campaign directory instead.")

    rows = load_pool(pool_csv, detectors)
    rules = rules or default_rules(detectors)

    order = [r["cand_id"] for r in rows]
    random.Random(seed).shuffle(order)

    # A simultaneous claim across rules needs level alpha/J_rules (section 6); read
    # individually, each interval is valid at alpha as it stands.
    eff_alpha = alpha / len(rules) if alpha_per_rule else alpha
    recall_max = recall_stream_max or n_max

    grids, f_v = {}, {}
    for name in rules:
        fn = resolve(name, detectors)
        f = accept_fraction(fn, rows)
        f_v[name] = f
        grids[f"{name}|precision"] = even_grid(max(J, int(f * n_max)), J)
        grids[f"{name}|recall"] = even_grid(recall_max, J)
        # The Jaccard stream is at least as long as the precision stream (it also
        # takes genuine candidates the rule rejected), so commit it on the same scale.
        grids[f"{name}|f1"] = even_grid(max(J, int(f * n_max)), J)

    spec = {
        "name": out_dir.name,
        # Which corpus the clip paths refer to, so the GUI can find the spectrograms
        # without being told again. Specs frozen before this field fall back to a guess
        # (vox_label/discover.py:guess_dataset).
        "dataset": dataset,
        "created": None,  # set by the caller if wanted; kept out of the hash-relevant fields
        "pool_csv": str(Path(pool_csv).as_posix()),
        "pool_sha256": sha256_file(pool_csv),
        "n_candidates": len(rows),
        "detectors": list(detectors),
        "rules": list(rules),
        "alpha": alpha,
        "alpha_effective": eff_alpha,
        "alpha_per_rule": bool(alpha_per_rule),
        "n_max": n_max,
        "J": J,
        "seed": seed,
        "recall_stream_max": recall_max,
        "f_v": f_v,
        "grids": grids,
        "order": order,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    spec_path.write_text(json.dumps(spec, indent=1))
    return spec_path, spec


class Campaign:
    """A loaded, validated campaign. Read-only."""

    def __init__(self, spec_dir):
        self.dir = Path(spec_dir)
        self.spec = json.loads((self.dir / "campaign.json").read_text())
        self.detectors = self.spec["detectors"]
        self.dataset = self.spec.get("dataset")
        self.order = self.spec["order"]
        self.rules = self.spec["rules"]
        self.alpha = self.spec["alpha_effective"]
        self.labels_path = self.dir / "labels.jsonl"
        rows = load_pool(self.spec["pool_csv"], self.detectors)
        self.pool = {r["cand_id"]: r for r in rows}
        self._rule_fns = {n: resolve(n, self.detectors) for n in self.rules}

    def verify_pool(self):
        """Confirm the pool file on disk is still the one the campaign was built on.

        A pool edited mid-campaign would silently change every candidate's meaning, so
        this is checked at startup rather than trusted.
        """
        actual = sha256_file(self.spec["pool_csv"])
        if actual != self.spec["pool_sha256"]:
            raise RuntimeError(
                f"pool file {self.spec['pool_csv']} has changed since the campaign was "
                f"frozen (sha256 {actual[:12]} != {self.spec['pool_sha256'][:12]}). "
                f"The campaign's estimands are no longer defined.")

    # View settings live beside the spec, never inside it: how wide a crop is drawn
    # changes nothing the coverage guarantee depends on, and campaign.json is frozen.
    view_path = property(lambda self: self.dir / "view.json")

    def load_view(self):
        try:
            return json.loads(self.view_path.read_text())
        except (OSError, json.JSONDecodeError):
            return {}

    def save_view(self, view):
        self.view_path.write_text(json.dumps(view, indent=1))

    def rule_fn(self, name):
        return self._rule_fns[name]

    def grid(self, rule, kind):
        return self.spec["grids"][f"{rule}|{kind}"]

    def load_labels(self):
        """Replay the append-only log into {cand_id: record}, last write winning.

        Undo is recorded as a retraction rather than by rewriting history, so the log
        stays append-only and the campaign is auditable after the fact.
        """
        out = {}
        if not self.labels_path.exists():
            return out
        for line in self.labels_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("retracted"):
                out.pop(rec["cand_id"], None)
            else:
                out[rec["cand_id"]] = rec
        return out

    def annotations(self, labels=None):
        """Annotated candidates **in committed order**, as [{cand_id, z, label, clip}].

        Order matters: the streams are prefixes of this sequence, and a checkpoint at
        n_j means the first n_j observations. Serving or replaying them in any other
        order would break the correspondence with the committed grid.
        """
        labels = self.load_labels() if labels is None else labels
        out = []
        for cid in self.order:
            rec = labels.get(cid)
            if rec is None:
                continue
            row = self.pool[cid]
            out.append({"cand_id": cid, "z": row["z"], "label": int(rec["label"]),
                        "clip": row["clip"]})
        return out

    def next_unlabeled(self, labels=None):
        labels = self.load_labels() if labels is None else labels
        for cid in self.order:
            if cid not in labels:
                return self.pool[cid]
        return None
