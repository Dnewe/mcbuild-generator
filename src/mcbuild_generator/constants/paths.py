# RUN ID
RUN_NAME = "full"

# ------
# DATA
# ------

BUILDS_DIR = "data/01_raw/schem"
ALL_BLOCKS_JSON = "data/index/all_blocks.json"

# -----------
# GENERATED
# -----------

### BUILDS FILEPATHS
RAW_BUILDS_FP_JSON = f"data/01_raw/raw_builds_fp_{RUN_NAME}.json"
CLEAN_BUILDS_FP_JSON = f"data/01_raw/clean_builds_fp_{RUN_NAME}.json"

### METADATA / ANALYSIS
BUILDS_METADATA_CSV = f"data/02_intermediate/builds_metadata_{RUN_NAME}.csv"
USED_BLOCKS_JSON = f"data/index/used_blocks_{RUN_NAME}.json"

### INDEXES
BLOCK_TO_IDX_JSON = f"data/index/block_to_idx_{RUN_NAME}.json"
IDX_TO_BLOCK_JSON = f"data/index/idx_to_block_{RUN_NAME}.json"
