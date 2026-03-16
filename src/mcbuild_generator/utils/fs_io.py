import json
import os
import csv
from typing import Dict, List


### CSV
def write_csv(rows:List[Dict], fp:str):
    with open(fp, 'w', newline='') as f:
        w = csv.DictWriter(f, rows[0].keys())
        w.writeheader()
        w.writerows(rows)


### SCHEM
def get_schem_filepaths(dir):
    filenames = os.listdir(dir)
    return [os.path.join(dir,fn) for fn in filenames if fn.split('.')[-1] == 'schem']

### JSON

def read_json(fp: str) -> Dict|List:
    with open(fp, 'r') as f:
        d = json.load(f)
    return d

def write_json(data, fp: str) -> None:
    with open(fp, 'w') as f:
        json.dump(data, f, indent=4, sort_keys=True)
        
        
