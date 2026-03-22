from tqdm import tqdm
from typing import Dict, List
from multiprocessing import Pool, cpu_count
import pandas as pd

from mcbuild_generator.processing.schem import Schem


def process_build(build: Dict[str, str]):
    id_ = build["id"]
    fp = build["filepath"]
    try:
        schem = Schem.load(fp)
    except Exception as e:
        print(f"Failed loading Schem file {fp}. \nerror: {e}")
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


def extract_builds_data(builds: List[Dict[str, str]], multiproc=True):
    """
    Extract metadata of builds schem files.
    """
    rows = []
    processes = cpu_count() - 2 if multiproc else 1
    with Pool(processes=processes) as pool:
        for row in tqdm(pool.imap_unordered(process_build, builds), total=len(builds)):
            if row is not None:
                rows.append(row)

    return pd.DataFrame(rows)
