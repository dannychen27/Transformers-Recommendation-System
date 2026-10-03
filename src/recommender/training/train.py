import torch
from torch import nn


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
