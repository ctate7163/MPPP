"""
The names that keep the old MPPP spelling in 0.99 (the rename MPPP -> RPP).

These are not branding but the data format and the state of the PC: names RPP writes into WORK folders, roots,
records and released files, and reads back.  Renaming them would make every existing block look unprocessed
(a manifest "missing" means a full reprocessing), lose the freeze and pipeline state, or break the downloads.
They change in a later version, after freeze-3, with a one-time ``rpp migrate`` that keeps the old names readable.

``tests/test_rename.py`` checks that the old spelling appears only in these names, in the compatibility code
(``rpp.env``, the ``mppp`` / ``mppp_studies`` alias packages, ``scripts/windows/mppp_env.bat``) and in history
(``CHANGELOG.md``, ``docs/history/``, ``paper/``, the packaged camera models ``data/cmods``).
"""
from __future__ import annotations

# ---- files and folders in a WORK folder ------------------------------------------------------------------
MANIFEST_PREFIX = "mppp_manifest_"          # processed/mppp_manifest_<v0pNN>.json: per-image metadata
CONFIG_PREFIX = "mppp_config_"              # processed/mppp_config_<v0pNN>.json: the configuration snapshot
SETTINGS_FILE = "mppp_settings.json"        # <WORK>/mppp_settings.json: the block's own settings
STATUS_FILE = "mppp_status.json"            # <WORK>/mppp_status.json: the run's status and heartbeat
DENSE_FILE = "mppp_dense.json"              # the dense step's record
DENSE_RUN_FILE = "mppp_dense_run.json"
SFM_LOG_FILE = "mppp_sfm_log.json"
STEREO_FILE = "mppp_stereo.json"

# ---- files and folders in a root (D:\rover_blocks) --------------------------------------------------------
STOP_MARKER = "mppp_stopped.txt"            # written by stop_runs; _run_sites.bat then skips RPP_THEN
BATCH_FILE = "mppp_batch.json"
BATCH_DIR = "mppp_batches"
PIPELINE_DIR = "mppp_pipeline"              # the freeze's and the pipeline's state and logs

# ---- keys inside records -----------------------------------------------------------------------------------
VERSION_KEY = "mppp"                        # run_done.json, the run settings, the status, Metashape/nerfstudio jobs, panos.json
VERSION_FIELD = "mppp_version"              # manifests, health.json, the training-set marker
VERSION_TAG_FIELD = "mppp_version_tag"      # the configuration snapshot

# ---- the released mask models ------------------------------------------------------------------------------
MASK_CARD_KEY = "mppp_card"                 # the model card inside the .safetensors metadata
MASK_FORMAT = "mppp-mask-safetensors-1"     # the "format" entry of that metadata (the files' SHA-256 depend on it)
MASK_OLD_NAMES = ("mppp_mask_v1", "mppp_mask_v2", "mppp_mask_v3", "mppp_mask_v4")   # aliases of rpp_mask_v1..v4
MASK_OLD_FILE = "mppp_mask_convnext_tiny_s4_{version}.safetensors"                 # the files on Hugging Face
MASK_OLD_HF_REPO = "ctate7163/mppp-mask"    # Hugging Face forwards it to ctate7163/rpp-mask once renamed

# ---- the mask training set ---------------------------------------------------------------------------------
REGEN_MARKER = "_mppp_regen.json"
REGEN_LOG = "_mppp_regen_log.csv"

# ---- the PC ------------------------------------------------------------------------------------------------
CACHE_DIR = "mppp"                          # %LOCALAPPDATA%\mppp (~/.cache/mppp): the mask models, gpu_python.txt,
                                            # the waypoint cache and gpu.lock - shared with v0p98, so the GPU steps
                                            # of both versions take turns
LEGACY_LOCAL_FILES = ("mppp_python.txt", "mppp_scapes.txt", "mppp_pds.txt", "mppp_cmods.txt")   # scripts/windows/
CONDA_ENVS = ("mppp_gpu", "mppp_gpu_lg")    # the PC's CUDA environments (new ones are rpp_gpu)
PC_FOLDERS = (r"D:\code\MPPP", r"D:\code\MPPP_freeze")   # keep their names through freeze-3 (new work: D:\code\RPP)

# ---- published things --------------------------------------------------------------------------------------
EXAMPLE_ASSET = "mppp_example_m2020_rockytop"   # the GitHub release asset of the example data (example-data-v1)
XMP_OLD_NS = ("https://github.com/ctate7163/MPPP/ns/1.0/", "https://github.com/ctate7163/MPPP/ns/0.1/")
XMP_OLD_PREFIX = "mppp"                     # stereo panoramas and processed images written before 0.99
