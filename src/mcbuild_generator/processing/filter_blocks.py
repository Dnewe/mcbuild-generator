import pandas as pd
import re


WOOD = [
    "oak",
    "birch",
    "jungle",
    "warped",
    "crimson",
    "acacia",
    "dark_oak",
    "spruce",
    "mangrove",
]


def _replace_rotation(match):
    """helper function to clean rotations from 16 to 8"""
    value = int(match.group(1))
    if value % 2 == 1:
        value -= 1
    return f"rotation={value}"


def _replace_block(s, prev_block, new_block="air"):
    return re.sub(rf"minecraft:{prev_block}\[.*$", f"minecraft:{new_block}", s)


def replace_block(block: str) -> str:
    """
    Replace irrelevant blocks with a different one.
    eg.: minecraft:infested_stone -> minecraft:stone
    """
    # old version blocks -> new block
    block = _replace_block(block, "flowing_water", "water[level=0]")
    block = _replace_block(block, "flowing_lava", "lava[level=0]")
    block = _replace_block(block, "dead_shrub", "dead_bush")
    block = _replace_block(block, "grass_path", "dirt_path")
    block = _replace_block(block, "short_grass", "grass")
    block = block.replace("wooden_slab", "oak_slab")
    block = block.replace(":skull", ":skeleton_skull")
    block = block.replace(":wall_skull", ":skeleton_wall_skull")
    block = block.replace(":sign", ":oak_sign")
    block = block.replace(":wall_sign", ":oak_wall_sign")
    block = block.replace(":bed[", ":red_bed[") # '[' to not include bedrock, ':' to not include other beds
    block = block.replace(":banner", ":white_banner") # ':' to not include other banners
    block = block.replace(":wall_banner", ":white_wall_banner")

    # light block -> air
    block = _replace_block(block, "light")  # light[ to ensure it is not light_cyan...

    # structure block/void / barrier -> air
    block = _replace_block(block, "structure_block")
    block = _replace_block(block, "structure_void")
    block = _replace_block(block, "barrier")
    block = _replace_block(block, "command_block")

    # redstone related blocks -> air
    block = _replace_block(block, "tripwire_hook")
    block = _replace_block(block, "repeater")
    block = _replace_block(block, "comparator")
    block = _replace_block(block, "redstone_wire")

    # suspicious sand/gravel -> sand/gravel
    block = _replace_block(block, "suspicious_gravel", "gravel")
    block = _replace_block(block, "suspicious_sand", "sand")

    # damaged/chipped anvil -> anvil
    block = block.replace("chipped_", "")
    block = block.replace("damaged_", "")

    # infested stone -> normal stone
    block = block.replace("infested_", "")

    # petrified oak -> normal oak
    block = block.replace("petrified_", "")

    # unwaxed copper -> waxed copper
    if "copper" in block and "copper_ore" not in block:
        block = block.replace(":", ":waxed_", 1)

    # all shulker -> purple shulker facing up (arbitrary purple)
    if "shulker" in block:
        block = "minecraft:purple_shulker_box[facing=up]"

    # double slabs -> normal block
    if "type=double" in block:
        if "brick" in block:  # bricks
            block = re.sub(r"brick.*$", "bricks", block)
        elif any(wood in block for wood in WOOD):  # wood
            block = re.sub(r"slab.*$", "planks", block)
        elif "quartz" in block:  # quartz
            block = re.sub(r"slab.*$", "block", block)
        else:  # others
            block = re.sub(r"_slab.*$", "", block)

    return block


def normalize_variant(block: str) -> str:
    """
    Normalizes / removes irrelevant block variant tags.
    eg.: minecraft:oak_leaves[distance=1,persistent=False,waterlogged=false] -> minecraft:oak_leaves[distance=7,persistent=true]
    """
    # remove False/false and True/true mismatch
    block = block.lower()

    # 16 rotations -> 8 rotations
    block = re.sub(r"rotation=(\d+)", _replace_rotation, block)

    # normalize tree saplings tags
    block = block.replace("stage=1", "stage=0")

    # normalize tree leaves tags
    block = re.sub(r"leaves.*$", "leaves[distance=7,persistent=true]", block)

    # normalize level tag (and honey_level)
    block = re.sub(r"level=(1[0-5]|[0-9])", "level=0", block)

    # normalize candles count
    block = re.sub(r"candles=[1-4]", "candles=3", block)

    # normalize cake bite count
    block = re.sub(r"bites=[0-5]", "bites=0", block)

    # normalize redstone power
    block = re.sub(r"power=(1[0-5]|[0-9])", "power=0", block)

    # bed occupied -> unoccupiedd
    block = block.replace("occupied=true", "occupied=false")

    # clean false waterlogged
    block = re.sub(r",?waterlogged=false,?", ",", block)
    block = re.sub(r"\[,", "[", block)
    block = re.sub(r",\]", "]", block)

    # clean empty tags
    block = block.replace("[]", "")

    return block


def normalize_block(block: str) -> str:
    block = replace_block(block)
    block = normalize_variant(block)
    return block


def merge_rare_variants(
    blocks_df: pd.DataFrame, threshold=0.1, prop_level="block"
) -> None:
    """
    Map rare variants to the most common variant
    """
    blocks_df["variant_prop_build"] = blocks_df["build_count"] / blocks_df.groupby(
        "base_block"
    )["build_count"].transform("sum")
    blocks_df["variant_prop_block"] = (
        blocks_df["block_count"]
        / blocks_df.groupby("base_block")["block_count"].transform("sum")
        # * usedblocks_df.groupby('base_block')['block'].transform('count')
    )

    most_common_map = blocks_df.loc[
        blocks_df.groupby("base_block")[f"variant_prop_{prop_level}"].idxmax()
    ].set_index("base_block")["block"]

    blocks_df["merged_variant"] = blocks_df["block"]
    mask = blocks_df[f"variant_prop_{prop_level}"] < threshold

    blocks_df.loc[mask, "merged_variant"] = blocks_df.loc[mask, "base_block"].map(
        most_common_map
    )


def remove_rare_blocks(
    blocks_df: pd.DataFrame,
    min_use_count_build: int,
    min_use_count_block: int,
    new_block: str = "minecraft:air",
):
    # block level
    block_total_counts = blocks_df.groupby("base_block")["block_count"].sum()
    block_rare_base_blocks = block_total_counts[
        block_total_counts < min_use_count_block
    ].index
    blocks_df.loc[blocks_df["base_block"].isin(block_rare_base_blocks), "new_block"] = (
        new_block
    )
    # build level
    build_total_counts = blocks_df.groupby("base_block")["build_count"].sum()
    build_rare_base_blocks = build_total_counts[
        build_total_counts < min_use_count_build
    ].index
    blocks_df.loc[blocks_df["base_block"].isin(build_rare_base_blocks), "new_block"] = (
        new_block
    )


def filter_blocks(
    blocks_df: pd.DataFrame,
    min_use_count_block=50,
    min_use_count_build=10,
    rare_variants_thresh=0.1,
    proportion_level="block",
):
    """
    Reduces block palette size by filtering rare/irrelevant blocks
    """
    block_count_start = len(blocks_df["block"].unique())

    merge_rare_variants(blocks_df, rare_variants_thresh, proportion_level)
    block_count_merged_variant = len(blocks_df["merged_variant"].unique())
    print(
        f"merged {block_count_start - block_count_merged_variant} rare blocks variants (thresh: <{rare_variants_thresh})"
    )

    blocks_df["new_block"] = blocks_df["merged_variant"].apply(normalize_block)
    block_count_normalized = len(blocks_df["new_block"].unique())
    print(
        f"merged {block_count_merged_variant - block_count_normalized} blocks by normalizing variants & base blocks"
    )

    remove_rare_blocks(blocks_df, min_use_count_build, min_use_count_block)
    block_count_end = len(blocks_df["new_block"].unique())
    print(f"removed {block_count_normalized - block_count_end} rare blocks.")

    print(f"Initial block palette size: {block_count_start}")
    print(f"Final block palette size  : {block_count_end}")
    print(f"-> merged {block_count_start - block_count_end} blocks")
