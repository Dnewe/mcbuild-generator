import mcschematic


def create_schematic(schem_dir:str, schem_name:str, data, idx_to_block):
    schem = mcschematic.MCSchematic()

    height ,length ,width = data.shape
    for h in range(height):
        for l in range(length):
            for w in range(width):
                idx = str(data[h, l, w])
                block = idx_to_block[idx]
                #print(f"{idx}: {block}")
                schem.setBlock((l, h, w), block)

    schem.save(schem_dir, schem_name, mcschematic.Version.JE_1_20_1)