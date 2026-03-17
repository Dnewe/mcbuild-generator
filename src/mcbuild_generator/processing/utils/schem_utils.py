import nbtlib
import numpy as np


def decode_varints(data, expected_size):
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


def get_schem_blockdata(schem: nbtlib.File) -> np.ndarray:
    version = schem["Version"]
    blockdata = schem["BlockData"]

    w, h, l = schem["Width"], schem["Height"], schem["Length"]
    size = w * h * l
    if version == 1:
        blocks = np.frombuffer(bytes(blockdata), dtype=np.uint8).astype(np.int32)
    else:
        blocks = np.array(decode_varints(blockdata, size), dtype=np.int32).reshape(
            (h, l, w)
        )
    return blocks
