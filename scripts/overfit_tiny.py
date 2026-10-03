import torch
from torch import nn
from torch.utils.data import DataLoader

from recommender.config import (
    DATA_PATH,
    LEARNING_RATE,
    MAX_SEQUENCE_LENGTH,
)
from recommender.data.dataset import SequenceDataset
from recommender.data.preprocessing import (
    build_user_sequences,
    load_ratings,
    make_training_examples,
    split_sequence,
)
from recommender.models.transformer import TransformerRecommender
from recommender.training.train import train_step


TINY_OVERFIT_SIZE = 16
TINY_OVERFIT_STEPS = 200


# 1. Load raw interactions
df = load_ratings(DATA_PATH)


# 2. Build chronological user histories
sequences = build_user_sequences(df)


# 3. Generate training examples from multiple users
all_training_examples = []

for sequence in sequences.values():
    train_sequence, _, _ = split_sequence(sequence)

    all_training_examples.extend(
        make_training_examples(
            train_sequence,
            max_sequence_length=MAX_SEQUENCE_LENGTH,
        )
    )


# 4. Keep a tiny fixed subset
tiny_examples = all_training_examples[:TINY_OVERFIT_SIZE]


# 5. Create Dataset and DataLoader
dataset = SequenceDataset(
    tiny_examples,
    max_sequence_length=MAX_SEQUENCE_LENGTH,
)

dataloader = DataLoader(
    dataset,
    batch_size=TINY_OVERFIT_SIZE,
    shuffle=False,
)

input_ids, targets = next(iter(dataloader))


# 6. Create model, loss, and optimizer
num_items = int(df["movie_id"].max()) + 1

# I deliberately set dropout=0.0 here.
# This experiment is about memorization, so we don't need
# regularization getting in the way.
model = TransformerRecommender(
    num_items=num_items,
    max_sequence_length=MAX_SEQUENCE_LENGTH,
    dropout=0.0,
)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
)


# 7. Repeatedly train on the same tiny batch
initial_loss = None

for step in range(TINY_OVERFIT_STEPS):
    loss = train_step(
        model=model,
        input_ids=input_ids,
        targets=targets,
        optimizer=optimizer,
        criterion=criterion,
    )

    if initial_loss is None:
        initial_loss = loss

    if step % 20 == 0 or step == TINY_OVERFIT_STEPS - 1:
        print(f"Step {step:3d} | Loss: {loss:.4f}")

print(f"\nInitial loss: {initial_loss:.4f}")
print(f"Final loss:   {loss:.4f}")

