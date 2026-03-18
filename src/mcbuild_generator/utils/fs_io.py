import json
import os
import shutil
import csv
from typing import Dict, List


### DIR
def create_dir(dir):
    try:
        os.mkdir(dir)
    except Exception as e:
        print(f"couldn't create dir at path {dir} \nerror: {e}")


def del_dir(dir):
    if os.path.exists(dir):
        shutil.rmtree(dir)


### CSV
def write_csv(fp: str, rows: List[Dict]):
    with open(fp, "w", newline="") as f:
        w = csv.DictWriter(f, rows[0].keys())
        w.writeheader()
        w.writerows(rows)


### SCHEM
def get_schem_filepaths(dir):
    filenames = os.listdir(dir)
    return [os.path.join(dir, fn) for fn in filenames if fn.split(".")[-1] == "schem"]


### JSON
def read_json(fp: str) -> Dict | List:
    with open(fp, "r") as f:
        d = json.load(f)
    return d


def write_json(fp: str, data) -> None:
    with open(fp, "w") as f:
        json.dump(data, f, indent=4, sort_keys=True)
