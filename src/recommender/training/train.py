import torch
from torch import nn

from recommender.config import (
    TRAIN_LOG_INTERVAL,
)


def train_step(
    model: nn.Module,
    input_ids: torch.Tensor,
    targets: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
) -> float:
    """Perform one optimization step and return the training loss."""
    model.train()

    optimizer.zero_grad()

    logits = model(input_ids)
    loss = criterion(logits, targets)

    loss.backward()
    optimizer.step()

    return loss.item()


def train_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    log_interval: int = TRAIN_LOG_INTERVAL,
) -> float:
    """Train the model for one epoch and return the average loss."""
    model.train()

    total_loss = 0.0
    total_examples = 0

    for batch_idx, (input_ids, targets) in enumerate(dataloader):
        loss = train_step(
            model=model,
            input_ids=input_ids,
            targets=targets,
            optimizer=optimizer,
            criterion=criterion,
        )

        batch_size = targets.size(0)
        total_loss += loss * batch_size
        total_examples += batch_size

        if batch_idx % log_interval == 0:
            print(
                f"Batch {batch_idx + 1}/{len(dataloader)} "
                f"| Loss: {loss:.4f}"
            )

    return total_loss / total_examples