"""
mppp.error -- the reconstruction-error model: predicts the 3-D precision of a
multi-station rover reconstruction from the station geometry (``core``), reads
the parameters it needs (epsilon, the cross-station match gate, the
decorrelation angle, the view graph) back out of the pipeline's
``error_input/`` exports (``colmap``, ``alignment``), and compares the
reconstruction regimes (``cases``, ``metrics``, ``network``, ``viewgraph``,
``plotting``).  It is under development and not part of the advertised MPPP
workflow: nothing in ``mppp.sfm`` or ``mppp.pipeline`` imports it, the
studies in ``studies/error/`` and ``notebooks/03_error_analysis.ipynb`` do.
The public API is the names below (``from mppp.error import ...``); the
physics checks are ``tests/test_error_model.py``.
"""
from .core import (choose_anchor, correlation_kernel, Surface, FlatPlane, OcclusionMask,
                   DEFAULT_ROVER_MASK, Instrument, MASTCAM_Z_34, NAVCAM,
                   Station, PoseModel, ModelConfig, Grid, PrecisionField,
                   solve_precision_field)
from .metrics import compute_metrics, metric_guide, worst_direction, summarize
from .viewgraph import ViewGraph, build_view_graph, graph_report
from .network import (NetworkPose, pose_covariance_from_network, network_report,
                      rigidity_score)
from .cases import (CASE_PARAMS, run_four_cases, lbs_field, ideal_field,
                    case_table)

__all__ = ["Surface", "FlatPlane", "OcclusionMask", "DEFAULT_ROVER_MASK",
           "Instrument", "MASTCAM_Z_34", "NAVCAM", "Station", "PoseModel", "ModelConfig",
           "Grid", "PrecisionField", "solve_precision_field", "compute_metrics",
           "metric_guide", "worst_direction", "summarize",
           "ViewGraph", "build_view_graph", "graph_report",
           "choose_anchor", "correlation_kernel", "NetworkPose",
           "pose_covariance_from_network", "network_report", "rigidity_score",
           "CASE_PARAMS", "run_four_cases", "lbs_field", "ideal_field", "case_table"]
