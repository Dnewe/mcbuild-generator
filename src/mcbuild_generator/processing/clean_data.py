import os
import pandas as pd
from tqdm import tqdm
from typing import List, Dict
from multiprocessing import Pool, cpu_count

from mcbuild_generator.processing.utils.schem import Schem
from mcbuild_generator.utils.fs_io import write_csv, read_json, write_json
from mcbuild_generator.constants.paths import RAW_BUILDS_FP_JSON, BUILDS_METADATA_CSV, CLEAN_BUILDS_FP_JSON


def process_build(build: Dict[str, str]):
    id_ = build['id']
    fp = build['filepath']
    try:
        schem = Schem.load(fp)
    except:
        return None
    
    return {
        'id': str(id_),
        'filepath': str(fp),
        'version': int(schem.version),
        'dataversion': int(schem.dataversion),
        'height': int(schem.height),
        'length': int(schem.length),
        'width': int(schem.width),
        'volume': int(schem.height * schem.length * schem.width),
        'palettemax': int(schem.palettemax)
    }


def extract_metadata(builds: List[Dict[str, str]], multiproc=True):
    '''
    Extract metadata of builds schem files.
    '''
    rows = []
    processes = cpu_count()-2 if multiproc else 1
    with Pool(processes=processes) as pool:
        for row in tqdm(pool.imap_unordered(process_build, builds), total=len(builds)):
            if row is not None:
                rows.append(row)
    write_csv(BUILDS_METADATA_CSV ,rows)


def remove_outliers(df:pd.DataFrame, columns:List[str], coeffs:List[int]):
    '''
    Return DF without outliers for the specified column var using Median Absolute Deviation (MAD)
    threshold = `coeff` * 1.4826 * MAD
    '''
    df_filtered = df.copy()
    thresholds = {}
    for col, coeff in zip(columns, coeffs):
        vol_median = df_filtered[col].median()
        MAD = (df_filtered[col] - vol_median).abs().median()

        thresholds[col] = coeff * 1.4826 * MAD
    
    for col in columns:
        count = len(df_filtered)
        df_filtered = df_filtered[(df_filtered[col] - vol_median).abs() <= thresholds[col]]
        print(f'Outliers {col}:')
        print(f'- threhold = {thresholds[col]:.2f}')
        print(f'- removed  = {count - len(df_filtered)}')

    return df_filtered

    
def clean_data(use_cache=True, multiproc=True):
    '''
    Clean data by removing outliers
    '''
    builds = list(read_json(RAW_BUILDS_FP_JSON))

    if not use_cache or not os.path.isfile(BUILDS_METADATA_CSV):
        extract_metadata(builds, multiproc)

    metadata_df = pd.read_csv(BUILDS_METADATA_CSV)

    metadata_df_filtered = remove_outliers(metadata_df, ['volume', 'width', 'length', 'height', 'palettemax'], coeffs=[5,5,5,5,3])

    start_build_count = len(metadata_df)
    end_build_count = len(metadata_df_filtered)
    print(f'Pre-cleaning build count : {start_build_count}')
    print(f'Post-cleaning build count: {end_build_count}')
    print(f'Removed: {start_build_count - end_build_count}')

    clean_builds_fp = metadata_df_filtered['filepath'].to_list()
    write_json(CLEAN_BUILDS_FP_JSON, clean_builds_fp)
    print(clean_builds_fp[:10])
