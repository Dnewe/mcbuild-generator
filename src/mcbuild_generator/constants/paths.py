# ------
# DATA
# ------

BUILDS_DIR = "data/01_raw/schem"
ALL_BLOCKS_JSON = "data/09_external/all_blocks.json"

# -----------
# GENERATED
# -----------

### BUILDS FILEPATHS
RAW_BUILDS_FP_JSON = "data/02_intermediate/raw_builds_fp.json"
CLEAN_BUILDS_FP_JSON = "data/02_intermediate/clean_builds_fp.json"

### METADATA / ANALYSIS
BUILDS_METADATA_CSV = "data/02_intermediate/builds_metadata.csv"  # independant of run (applied to all files)
BLOCKS_COUNT_CSV = "data/02_intermediate/used_blocks.csv"

### INDEXES
BLOCK_TO_IDX_JSON = "data/03_processed/block_to_idx.json"
IDX_TO_BLOCK_JSON = "data/03_processed/idx_to_block.json"

### PROCESSED BUILDS
PROCESSED_BUILDS_DIR = "data/03_processed/builds"

### TRAINING
MODEL_FP = "data/05_models/model.pth"
LOSSES_PLOT_FP = "data/07_reporting/losses.jpg"

### MODEL OUTPUTS
GENERATED_SCHEM_DIR = "/home/ewen/Documents/curseforge/minecraft/Instances/mcbuild-generator/schematics"  # "data/06_outputs/"
