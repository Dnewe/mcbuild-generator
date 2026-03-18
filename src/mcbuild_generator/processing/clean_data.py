import os
import pandas as pd
from tqdm import tqdm
from typing import List, Dict
from multiprocessing import Pool, cpu_count

from mcbuild_generator.processing.schem import Schem
from mcbuild_generator.utils.fs_io import write_csv, read_json, write_json
from mcbuild_generator.constants.paths import (
    RAW_BUILDS_FP_JSON,
    BUILDS_METADATA_CSV,
    CLEAN_BUILDS_FP_JSON,
)


def process_build(build: Dict[str, str]):
    id_ = build["id"]
    fp = build["filepath"]
    try:
        schem = Schem.load(fp)
    except:
        return None

    return {
        "id": str(id_),
        "filepath": str(fp),
        "version": int(schem.version),
        "dataversion": int(schem.dataversion),
        "height": int(schem.height),
        "length": int(schem.length),
        "width": int(schem.width),
        "volume": int(schem.height * schem.length * schem.width),
        "palettemax": int(schem.palettemax),
    }


def extract_metadata(builds: List[Dict[str, str]], multiproc=True):
    """
    Extract metadata of builds schem files.
    """
    rows = []
    processes = cpu_count() - 2 if multiproc else 1
    with Pool(processes=processes) as pool:
        for row in tqdm(pool.imap_unordered(process_build, builds), total=len(builds)):
            if row is not None:
                rows.append(row)
    write_csv(BUILDS_METADATA_CSV, rows)


def filter_outliers(df: pd.DataFrame, columns: List[str], coeffs: List[int]):
    """
    Return DF without outliers for the specified column var using Median Absolute Deviation (MAD)
    threshold = `coeff` * 1.4826 * MAD
    """
    df_filtered = df.copy()
    thresholds = {}
    for col, coeff in zip(columns, coeffs):
        vol_median = df_filtered[col].median()
        MAD = (df_filtered[col] - vol_median).abs().median()

        thresholds[col] = coeff * 1.4826 * MAD

    for col in columns:
        df_filtered = df_filtered[
            (df_filtered[col] - vol_median).abs() <= thresholds[col]
        ]
        print(f"\nOutliers {col}:")
        print(f"- threhold = {thresholds[col]:.2f}")
        print(f"- removed  = {len(df) - len(df_filtered)}")

    return df_filtered


def filter_outofbonds(df: pd.DataFrame, min_w, min_l, min_h, max_w, max_l, max_h):
    df_filtered = df.copy()
    df_filtered = df_filtered[df_filtered["width"] >= min_w]
    df_filtered = df_filtered[df_filtered["length"] >= min_l]
    df_filtered = df_filtered[df_filtered["height"] >= min_h]
    if max_w>0:
        df_filtered = df_filtered[df_filtered["width"] <= max_w]
    if max_l>0:
        df_filtered = df_filtered[df_filtered["length"] <= max_l]
    if max_h>0:
        df_filtered = df_filtered[df_filtered["height"] <= max_h]
    print(f"\nRemoved {len(df) - len(df_filtered)} out of bonds builds")
    return df_filtered


def clean_data(min_w=0, min_l=0, min_h=0, max_w=256, max_l=256, max_h=256, use_cache=True, multiproc=True):
    """
    Clean data by removing outliers
    """
    raw_builds_fp = list(read_json(RAW_BUILDS_FP_JSON))

    if not use_cache or not os.path.isfile(BUILDS_METADATA_CSV):
        extract_metadata(raw_builds_fp, multiproc)

    metadata_df = pd.read_csv(BUILDS_METADATA_CSV)

    metadata_df_filtered = filter_outliers(
        metadata_df,
        ["volume", "width", "length", "height", "palettemax"],
        coeffs=[5, 5, 5, 5, 3],
    )
    metadata_df_filtered = filter_outofbonds(metadata_df_filtered, min_w, min_l, min_h, max_w, max_l, max_h)

    start_build_count = len(metadata_df)
    end_build_count = len(metadata_df_filtered)
    print(f"Pre-cleaning build count : {start_build_count}")
    print(f"Post-cleaning build count: {end_build_count}")
    print(f"Removed: {start_build_count - end_build_count}")

    clean_builds_fp = metadata_df_filtered["filepath"].to_list()
    write_json(CLEAN_BUILDS_FP_JSON, clean_builds_fp)
