import nbtlib
from nbtlib import File
import numpy as np
from .utils.schem_utils import get_schem_blockdata


def schem2vec(fp:str):
    '''
    Transforms schem files to 3D array
    '''
    schem = nbtlib.load(fp)
    
    palette = schem['Palette']
    palette_inv = {v: k for k, v in palette.items()}
    
    
    