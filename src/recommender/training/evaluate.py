import torch
from torch import nn
from torch.utils.data import DataLoader


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
) -> float:
    """Evaluate the model and return average loss."""
    model.eval()

    total_loss = 0.0
    total_examples = 0

    for input_ids, targets in dataloader:
        logits = model(input_ids)
        loss = criterion(logits, targets)

        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        total_examples += batch_size

    return total_loss / total_examples
