import torch
from torch.utils.data import DataLoader

from recommender.data.dataset import SequenceDataset
from recommender.config import (
    TEST_MAX_SEQUENCE_LENGTH, TEST_TRUNCATION_LENGTH, TEST_BATCH_SIZE
)


def test_short_sequence_is_left_padded():
    examples = [([10, 20], 30)]

    dataset = SequenceDataset(
        examples,
        max_sequence_length=TEST_MAX_SEQUENCE_LENGTH,
    )

    input_ids, target = dataset[0]

    assert torch.equal(
        input_ids,
        torch.tensor([0, 0, 0, 10, 20]),
    )
    assert target.item() == 30


def test_long_sequence_is_truncated():
    examples = [([1, 2, 3, 4, 5, 6], 7)]

    dataset = SequenceDataset(
        examples,
        max_sequence_length=TEST_TRUNCATION_LENGTH,
    )

    input_ids, target = dataset[0]

    assert torch.equal(
        input_ids,
        torch.tensor([3, 4, 5, 6]),
    )
    assert target.item() == 7


def test_dataloader_batch_shape():
    examples = [
        ([1], 2),
        ([1, 2], 3),
        ([1, 2, 3], 4),
        ([1, 2, 3, 4], 5),
    ]

    dataset = SequenceDataset(
        examples,
        max_sequence_length=TEST_MAX_SEQUENCE_LENGTH,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=TEST_BATCH_SIZE,
        shuffle=False,
    )

    input_ids, targets = next(iter(dataloader))

    assert input_ids.shape == (TEST_BATCH_SIZE, TEST_MAX_SEQUENCE_LENGTH)
    assert targets.shape == (TEST_BATCH_SIZE,)
    assert torch.equal(
        targets,
        torch.tensor([2, 3, 4, 5]),
    )
