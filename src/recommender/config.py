from pathlib import Path


DATA_PATH = Path("data/raw/ml-100k/u.data")


MAX_SEQUENCE_LENGTH = 5
BATCH_SIZE = 4


# for scripts/inspect_data.py
INSPECT_USER_ID = 196


# constants for tests
TEST_MAX_SEQUENCE_LENGTH = 5
TEST_TRUNCATION_LENGTH = 4
TEST_BATCH_SIZE = 4
