"""Let SAM3's image model run without Triton (Windows has no Triton build)."""
import sys
import types


def allow_sam3_without_triton():
    """No-op when Triton imports. Otherwise stub SAM3's video-only Triton distance transform and fill mask
    holes with SAM3's CPU connected components instead of its Triton kernel. Call before importing sam3."""
    try:
        import triton  # noqa: F401
        return
    except ImportError:
        pass
    if "sam3.model.edt" not in sys.modules:  # imported at load time, only called by video tracking
        edt = types.ModuleType("sam3.model.edt")

        def edt_triton(*args, **kwargs):
            raise RuntimeError("SAM3 video tracking needs Triton")
        edt.edt_triton = edt_triton
        sys.modules["sam3.model.edt"] = edt
    import sam3.perflib.connected_components as cc
    if cc.HAS_CC_TORCH or getattr(cc.connected_components, "cpu_fallback", False):
        return
    orig = cc.connected_components

    def connected_components(t):
        if not t.is_cuda:
            return orig(t)
        labels, counts = orig(t.cpu())
        return labels.to(t.device), counts.to(t.device)
    connected_components.cpu_fallback = True
    cc.connected_components = connected_components
