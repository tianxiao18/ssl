"""Canonical outputs/ layout: outputs/<stage>/<dataset>/... for spectrograms and
segmentation, and outputs/eval/<dataset>/<method>/... for evaluation (dataset-first).
"""
from pathlib import Path


def _dataset_and_rel_parts(spec_dir):
    """(dataset, rel_parts) from a spec_dir shaped .../spectrograms/<dataset>/<rel...>.

    rel_parts is everything after <dataset> -- one segment for a flat-layout
    dataset (e.g. dryad_gerbil's <recording>), two for experiment_*/idx_*
    (gerbil_ssl, gerbil_family). Not assumed to be any fixed depth.
    """
    parts = Path(spec_dir).resolve().parts
    if "spectrograms" in parts:
        idx = parts.index("spectrograms")
        if idx + 1 < len(parts):
            return parts[idx + 1], parts[idx + 2:]
    raise ValueError(
        f"could not infer dataset from spec_dir={spec_dir} "
        f"(expected a path like .../spectrograms/<dataset>/...)"
    )


def dataset_from_spec_dir(spec_dir):
    """Extract the dataset segment from a spec_dir shaped .../spectrograms/<dataset>/...

    Used to replace hardcoded "gerbil_ssl" recording_dir fallbacks, which were
    silently wrong for any other dataset.
    """
    return _dataset_and_rel_parts(spec_dir)[0]


def recording_dir_from_spec_dir(spec_dir, data_root="data"):
    """Mirror a spec_dir (outputs/spectrograms/<dataset>/<rel...>) into
    <data_root>/<dataset>/<rel...> -- the raw-audio counterpart of spec_dir.

    Mirrors the FULL relative path after <dataset>, not just the last one or two
    segments: a fixed depth (e.g. always "parent/name") is wrong for flat-layout
    datasets like dryad_gerbil, where spec_dir is only one level below <dataset>
    and spec_dir.parent.name would be the dataset itself, not a real subfolder.
    """
    dataset, rel_parts = _dataset_and_rel_parts(spec_dir)
    return Path(data_root) / dataset / Path(*rel_parts)
