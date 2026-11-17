"""
rpp_studies.msl - Mars Science Laboratory (Curiosity) blocks, a study before the pipeline takes MSL in (0.99.4).

Christian, 16 Nov 2026: "v0p99 will also attempt to interpret msl data in D:\\code\\MPPP\\data\\msl and align its first
block", and 16:02: "Prepare a test msl block".  :mod:`rpp_studies.msl.first_block` fetches a site's Navcam RDRs from
the PDS Imaging Node, prepares them (the MIPL rover masks, the label CAHVOR as COLMAP cameras, the label poses in the
site frame) and aligns them with COLMAP, then reports the block against the labels (registered share, reprojection
error, the scale and the stereo baseline).  The M2020 pipeline is not touched: what this study learns goes into
``rpp/missions`` (v0p99 plan, section 3.3) afterwards.
"""
__all__ = ["first_block"]
