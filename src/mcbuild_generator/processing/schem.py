import os
import nbtlib
import numpy as np
from typing import Dict, Any, List
from collections import defaultdict


class Schem:
    version: int
    dataversion: int
    width: int
    height: int
    length: int
    palette: Dict[str, int]
    palettemax: int
    metadata: Dict[str, Any]
    blockdata: np.ndarray
    blockentities: List[Any]

    def __init__(self, schemfile, id) -> None:
        self.id = id
        self.schemfile = schemfile
        for k, v in schemfile.items():
            setattr(self, k.lower(), v)
        self.palette_inv = {v: k for k, v in self.palette.items()}

    @classmethod
    def load(cls, fp: str):
        schemfile = nbtlib.load(fp)
        id = os.path.basename(fp).split(".")[0]
        return cls(schemfile, id)

    def get_blockdata(self) -> np.ndarray:
        """
        Returns block data as 3D array of shape (h, l, w)
        """
        version = self.version
        blockdata = self.blockdata

        w, h, l = self.width, self.height, self.length
        size = w * h * l
        if version == 1:
            blocks = np.frombuffer(bytes(blockdata), dtype=np.uint8).astype(np.int32)
        else:
            blocks = np.array(
                self._decode_varints(blockdata, size), dtype=np.int32
            ).reshape((h, l, w))
        return blocks

    def get_block_counts(self) -> Dict[str, int]:
        """
        Returns a dictionary of the use count of each block in the schem
        """
        version = self.version
        if version == 1:
            data = np.frombuffer(bytes(self.blockdata), dtype=np.uint8)
            counts = np.bincount(data)
            return {self.palette_inv[i]: v for i, v in enumerate(counts)}
        else:
            return self._decode_counts(self.blockdata)

    def _decode_varints(self, data, expected_size):
        out = [0] * expected_size

        value = 0
        shift = 0
        i = 0
        for b in data:
            b &= 0xFF
            value |= (b & 0x7F) << shift
            if b & 0x80:
                shift += 7
            else:
                out[i] = value
                i += 1
                value = 0
                shift = 0
        return out

    def _decode_counts(self, data):
        counts = defaultdict(int)
        value = 0
        shift = 0
        for b in data:
            b &= 0xFF
            value |= (b & 0x7F) << shift
            if b & 0x80:
                shift += 7
            else:
                counts[self.palette_inv[value]] += 1
                value = 0
                shift = 0
        return counts

    def to_array(self, blocks_to_idx):
        data = self.get_blockdata()

        max_idx = max(self.palette_inv.keys())
        lookup = np.zeros(max_idx + 1, dtype=int)

        for i in range(max_idx + 1):
            lookup[i] = blocks_to_idx[self.palette_inv[i]]

        return lookup[data]
