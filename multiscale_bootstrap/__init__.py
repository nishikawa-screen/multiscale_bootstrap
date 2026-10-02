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
    encode_edge_states,
)
from ._scaleboot import (
    RScalebootClient,
    build_scaleboot_input,
)

__all__ = [
    "MultiScaleBootstrap",
    "MultiScaleBootstrapResult",
    "SingleScaleBootstrapResult",
    "EdgeHypothesis",
    "GraphHypothesis",
    "encode_edge_states",
    "RScalebootClient",
    "build_scaleboot_input",
]
