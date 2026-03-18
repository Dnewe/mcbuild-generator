from mcbuild_generator.processing.schem import Schem


def test_get_schem_blockdata(schem_fp):
    schem = Schem.load(schem_fp)
    blockdata = schem.get_blockdata()
    height, length, width = blockdata.shape
    # test dims
    assert height == schem.height
    assert length == schem.length
    assert width == schem.width
    # test block in palette
    for h in range(height):
        for l in range(length):
            for w in range(width):
                assert blockdata[h, l, w] in schem.palette.values()
