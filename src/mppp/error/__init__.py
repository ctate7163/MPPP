"""
mppp.error -- Mars Photogrammetric Precision Prediction: the error model.

Merged into MPPP v0p2 from the stand-alone ``mppp_error`` v0p15 (186 self-tests,
``python -m mppp.error selftest``).  Code unchanged except: imports made
package-relative, and a duplicate definition of ``waypoints._last_per_sol``
renamed (the second shadowed the first, so ``build_stations`` raised TypeError).
Full lab notebook: docs/error/mppp_error_README_v0p15.md.  Open review findings:
docs/error/mppp_error_review_v0p2.md.

Status (v1): under development; not part of the advertised MPPP workflow.
The waypoint snapshot is shared with the rest of MPPP (``mppp/data``; v0p13),
and so are the COLMAP camera projection (``mppp.colmap``) and the waypoint
loader (``mppp.waypoints``).  The research scripts ``study_navcam`` and
``study_cross_station`` moved to ``studies/error/`` (not installed).
"""
from .core import (choose_anchor, correlation_kernel, Surface, FlatPlane, GriddedSurface, OcclusionMask,
                   DEFAULT_ROVER_MASK, Instrument, MASTCAM_Z_34, NAVCAM,
                   Station, PoseModel, ModelConfig, Grid, PrecisionField,
                   solve_precision_field)
from .metrics import compute_metrics, metric_guide, worst_direction, summarize
from .viewgraph import ViewGraph, build_view_graph, graph_report
from .network import (NetworkPose, pose_covariance_from_network, network_report,
                      rigidity_score)
from .cases import (CASE_PARAMS, run_four_cases, lbs_field, ideal_field,
                    case_table)
__version__ = "2.0.0"      # internal model generation, kept for provenance
ERROR_MODEL_VERSION = "v0p15"
__all__ = ["Surface","FlatPlane","GriddedSurface","OcclusionMask","DEFAULT_ROVER_MASK",
           "Instrument","MASTCAM_Z_34","NAVCAM","Station","PoseModel","ModelConfig",
           "Grid","PrecisionField","solve_precision_field","compute_metrics",
           "metric_guide","worst_direction","summarize",
           "ViewGraph","build_view_graph","graph_report",
           "choose_anchor","correlation_kernel","NetworkPose",
           "pose_covariance_from_network","network_report","rigidity_score",
           "CASE_PARAMS","run_four_cases","lbs_field","ideal_field","case_table"]
