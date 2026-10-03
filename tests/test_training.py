import torch
from torch import nn

from recommender.models.transformer import TransformerRecommender
from recommender.training.train import train_step


TEST_BATCH_SIZE = 4
TEST_SEQUENCE_LENGTH = 5
TEST_NUM_ITEMS = 20
TEST_LEARNING_RATE = 1e-3


def test_train_step_updates_model():
    torch.manual_seed(42)

    model = TransformerRecommender(
        num_items=TEST_NUM_ITEMS,
        max_sequence_length=TEST_SEQUENCE_LENGTH,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=TEST_LEARNING_RATE,
    )

    criterion = nn.CrossEntropyLoss()

    input_ids = torch.tensor(
        [
            [0, 0, 1, 2, 3],
            [0, 4, 5, 6, 7],
            [8, 9, 10, 11, 12],
            [0, 0, 13, 14, 15],
        ]
    )

    targets = torch.tensor([4, 8, 13, 16])

    parameters_before = [
        parameter.detach().clone()
        for parameter in model.parameters()
    ]

    loss = train_step(
        model,
        input_ids,
        targets,
        optimizer,
        criterion,
    )

    assert torch.isfinite(torch.tensor(loss))

    parameters_after = list(model.parameters())

    assert any(
        not torch.equal(before, after)
        for before, after in zip(parameters_before, parameters_after)
    )
