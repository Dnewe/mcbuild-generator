from mcbuild_generator.utils.args import get_config

# RETRIEVE RUN ID
try:
    run_name = get_config()["run_name"]
except Exception as e:
    print(f"Failed loading 'run_name': {e}")
    run_name = "default"

# ------
# DATA
# ------

BUILDS_DIR = "data/01_raw/schem"
ALL_BLOCKS_JSON = "data/09_external/all_blocks.json"

# -----------
# GENERATED
# -----------

### BUILDS FILEPATHS
RAW_BUILDS_FP_JSON = f"data/01_raw/raw_builds_fp_{run_name}.json"
CLEAN_BUILDS_FP_JSON = f"data/01_raw/clean_builds_fp_{run_name}.json"

### METADATA / ANALYSIS
BUILDS_METADATA_CSV = "data/02_intermediate/builds_metadata.csv"  # independant of run (applied to all files)
BLOCKS_COUNT_CSV = f"data/02_intermediate/used_blocks_{run_name}.csv"

### INDEXES
BLOCK_TO_IDX_JSON = f"data/03_processed/block_to_idx_{run_name}.json"
IDX_TO_BLOCK_JSON = f"data/03_processed/idx_to_block_{run_name}.json"

### PROCESSED BUILDS
PROCESSED_BUILDS_DIR = f"data/03_processed/builds_{run_name}"

### TRAINING
MODEL_FP = f"data/05_models/model_{run_name}.pth"
LOSSES_PLOT_FP = f"data/07_reporting/losses_{run_name}.jpg"

### MODEL OUTPUTS
GENERATED_SCHEM_DIR = f"data/06_outputs/"
