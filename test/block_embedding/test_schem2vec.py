from block_embedding.utils.schem_utils import get_schem_blockdata
import nbtlib

def test_get_schem_blockdata(schem_fp):
    schem = nbtlib.load(schem_fp)
    blockdata = get_schem_blockdata(schem)
    height, length, width = blockdata.shape
    # test dims
    assert height == schem['Height']
    assert length == schem['Length']
    assert width == schem['Width']
    # test block in palette
    for h in range(height):
        for l in range(length):
            for w in range(width):
                assert blockdata[h,l,w] in schem['Palette'].values()