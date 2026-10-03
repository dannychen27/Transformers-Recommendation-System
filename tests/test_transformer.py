import pytest
import torch

from recommender.models.transformer import TransformerRecommender


TEST_BATCH_SIZE = 4
TEST_SEQUENCE_LENGTH = 5
TEST_NUM_ITEMS = 20


def create_model() -> TransformerRecommender:
    return TransformerRecommender(
        num_items=TEST_NUM_ITEMS,
        max_sequence_length=TEST_SEQUENCE_LENGTH,
    )


def test_output_shape():
    model = create_model()
    input_ids = torch.tensor(
        [
            [0, 0, 1, 2, 3],
            [0, 4, 5, 6, 7],
            [8, 9, 10, 11, 12],
            [0, 0, 13, 14, 15],
        ]
    )

    logits = model(input_ids)

    assert logits.shape == (TEST_BATCH_SIZE, TEST_NUM_ITEMS)


def test_padded_input_produces_valid_output():
    model = create_model()
    input_ids = torch.tensor(
        [
            [0, 0, 0, 1, 2],
            [0, 0, 3, 4, 5],
        ]
    )

    logits = model(input_ids)

    assert torch.isfinite(logits).all()


def test_max_sequence_length():
    model = create_model()
    input_ids = torch.randint(
        1,
        TEST_NUM_ITEMS,
        (TEST_BATCH_SIZE, TEST_SEQUENCE_LENGTH),
    )

    logits = model(input_ids)

    assert logits.shape == (TEST_BATCH_SIZE, TEST_NUM_ITEMS)


def test_sequence_length_exceeds_maximum():
    model = create_model()
    input_ids = torch.randint(
        1,
        TEST_NUM_ITEMS,
        (TEST_BATCH_SIZE, TEST_SEQUENCE_LENGTH + 1),
    )

    with pytest.raises(ValueError):
        model(input_ids)
