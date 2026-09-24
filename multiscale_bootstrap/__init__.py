"""Multiscale bootstrap and scaleboot integration for LiNGAM models.

Provides bootstrap probabilities for edge, arc, and graph hypotheses, with
optional AU/SI p-values from R's scaleboot package.
"""

from ._bootstrap import (
    MultiScaleBootstrap,
    MultiScaleBootstrapResult,
    SingleScaleBootstrapResult,
)
from ._hypotheses import (
    EdgeHypothesis,
    GraphHypothesis,
    GraphHypothesisResult,
    HypothesisResult,
    compute_graph_hash,
    encode_edge_states,
)
from ._scaleboot import (
    RScalebootClient,
    ScalebootInput,
    ScalebootResult,
    ScalebootRow,
    build_scaleboot_input,
)

__all__ = [
    "MultiScaleBootstrap",
    "HypothesisResult",
    "GraphHypothesisResult",
    "EdgeHypothesis",
    "GraphHypothesis",
    "compute_graph_hash",
    "encode_edge_states",
    "MultiScaleBootstrapResult",
    "SingleScaleBootstrapResult",
    "RScalebootClient",
    "ScalebootInput",
    "ScalebootResult",
    "ScalebootRow",
    "build_scaleboot_input",
]
