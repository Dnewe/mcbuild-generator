import pytest
import yaml


CONFIG_PATH = "conf/parameters.yaml"


@pytest.fixture(scope="module")
def config():
    config = yaml.load(open(CONFIG_PATH, "r"), Loader=yaml.SafeLoader)
    return config["processing"]


@pytest.fixture(scope="module")
def schem_fp():
    return "data/01_raw/test/build_batch_352_9133_6.schem"


@pytest.fixture(scope="module")
def metadata_csv():
    return "data/02_intermediate/builds_metadata_pytest.csv"


@pytest.fixture(scope="module")
def block_to_idx_json():
    return "data/03_processed/block_to_idx_pytest.json"


@pytest.fixture(scope="module")
def idx_to_block_json():
    return "data/03_processed/idx_to_block_pytest.json"


@pytest.fixture(scope="module")
def processed_dir():
    return "data/03_processed/builds_pytest"
