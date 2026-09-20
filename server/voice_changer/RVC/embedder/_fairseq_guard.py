"""Lazy fairseq import helper (local-custom).

fairseq is only needed by the Fairseq* embedders (torch path). The ONNX
contentvec path works without it, so importing it eagerly at module level
breaks slot loading when fairseq is absent (Win/torch-2.0 venv).
"""

try:
    from fairseq import checkpoint_utils  # noqa: F401
    FAIRSEQ_AVAILABLE = True
except Exception:  # noqa: BLE001
    checkpoint_utils = None
    FAIRSEQ_AVAILABLE = False
