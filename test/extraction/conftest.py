import pytest
import yaml

CONFIG_PATH = "conf/parameters.yaml"
TEST_MAX_FILES = 100


@pytest.fixture(scope="module")
def max_files():
    return TEST_MAX_FILES


@pytest.fixture(scope="module")
def config():
    config = yaml.load(open(CONFIG_PATH, "r"), Loader=yaml.SafeLoader)
    config["extraction"]["max_files"] = TEST_MAX_FILES
    return config["extraction"]


@pytest.fixture(scope="module")
def metadata_csv():
    return "data/02_intermediate/builds_metadata_pytest.csv"
