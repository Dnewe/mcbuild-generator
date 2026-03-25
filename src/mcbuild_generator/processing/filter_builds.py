import pandas as pd
from typing import List, Dict
from tqdm import tqdm

from mcbuild_generator.processing.schem import Schem


def filter_outofbonds(df: pd.DataFrame, min_w, min_l, min_h, max_w, max_l, max_h):
    df_filtered = df.copy()
    df_filtered = df_filtered[df_filtered["width"] >= min_w]
    df_filtered = df_filtered[df_filtered["length"] >= min_l]
    df_filtered = df_filtered[df_filtered["height"] >= min_h]
    if max_w > 0:
        df_filtered = df_filtered[df_filtered["width"] <= max_w]
    if max_l > 0:
        df_filtered = df_filtered[df_filtered["length"] <= max_l]
    if max_h > 0:
        df_filtered = df_filtered[df_filtered["height"] <= max_h]
    print(f"removed {len(df) - len(df_filtered)} out of bonds builds")
    return df_filtered


def filter_outliers(df: pd.DataFrame, cols_thresh: Dict[str, float]):
    """
    Return DF without outliers for the specified column var using Median Absolute Deviation (MAD)
    threshold = `coeff` * 1.4826 * MAD
    """
    df_filtered = df.copy()
    thresholds = {}
    for col, coeff in cols_thresh.items():
        vol_median = df_filtered[col].median()
        MAD = (df_filtered[col] - vol_median).abs().median()

        thresholds[col] = coeff * 1.4826 * MAD

    for col in cols_thresh.keys():
        previous_count = len(df_filtered)
        df_filtered = df_filtered[
            (df_filtered[col] - vol_median).abs() <= thresholds[col]
        ]
        print(
            f"- Outliers {col}: thresh= {thresholds[col]:.2f} ; removed= {previous_count - len(df_filtered)}"
        )

    print(f"removed {len(df) - len(df_filtered)} outlier builds")
    return df_filtered


def filter_irrelevant_builds(df: pd.DataFrame, relevant_blocks: List[str]):
    """
    Filter builds without any relevant blocks.
    (relevant block list at data/09_external/relevant_blocks.json)
    """
    mask = []

    for fp in tqdm(df["filepath"], desc="filtering irrelevant builds"):
        schem = Schem.load(fp)
        palette_base_block = [b.split("[")[0] for b in schem.palette.keys()]
        mask.append(not set(palette_base_block).isdisjoint(relevant_blocks))

    df_filtered = df[mask].copy()
    print(f"removed {len(df) - len(df_filtered)} irrelevant builds")
    return df_filtered


def filter_builds(
    metadata_df: pd.DataFrame,
    relevant_blocks: List[str],
    max_files: int,
    outlier_cols_thresh: Dict[str, float],
    min_w=0,
    min_l=0,
    min_h=0,
    max_w=256,
    max_l=256,
    max_h=256,
) -> List[str]:
    """
    Filter data by removing outliers and out of bonds builds
    """
    metadata_df_filtered = metadata_df.copy()
    # outliers
    metadata_df_filtered = filter_outliers(
        metadata_df_filtered, cols_thresh=outlier_cols_thresh
    )

    # out of bonds
    metadata_df_filtered = filter_outofbonds(
        metadata_df_filtered, min_w, min_l, min_h, max_w, max_l, max_h
    )

    # irrelevant
    metadata_df_filtered = filter_irrelevant_builds(
        metadata_df_filtered, relevant_blocks
    )

    # max files
    if max_files > 0 and len(metadata_df_filtered) > max_files:
        print(
            f"\nremoving excess files (current= {len(metadata_df_filtered)}, max_files= {max_files})"
        )
        metadata_df_filtered = metadata_df_filtered[:max_files]

    start_build_count = len(metadata_df)
    end_build_count = len(metadata_df_filtered)
    print(f"Initial build count : {start_build_count}")
    print(f"Final build count: {end_build_count}")
    print(f"-> removed {start_build_count - end_build_count} builds")

    return metadata_df_filtered["filepath"].to_list()
