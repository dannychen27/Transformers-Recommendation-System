import time
import torch
from torch import nn
from torch.utils.data import DataLoader

from recommender.config import (
    DATA_PATH,
    LEARNING_RATE,
    MAX_SEQUENCE_LENGTH,
    NUM_EPOCHS,
    TRAIN_BATCH_SIZE,
    TRAIN_LOG_INTERVAL,
)
from recommender.data.dataset import SequenceDataset
from recommender.data.preprocessing import (
    build_user_sequences,
    load_ratings,
    make_training_examples,
    split_sequence,
)
from recommender.models.transformer import TransformerRecommender
from recommender.training.train import train_epoch



# 1. Load raw interactions
df = load_ratings(DATA_PATH)


# 2. Build chronological user histories
sequences = build_user_sequences(df)


# 3. Build training examples from all users
training_examples = []
for sequence in sequences.values():
    train_sequence, _, _ = split_sequence(sequence)

    training_examples.extend(
        make_training_examples(
            train_sequence,
            max_sequence_length=MAX_SEQUENCE_LENGTH,
        )
    )


# 4. Create Dataset and DataLoader
training_dataset = SequenceDataset(
    training_examples,
    max_sequence_length=MAX_SEQUENCE_LENGTH,
)

training_dataloader = DataLoader(
    training_dataset,
    batch_size=TRAIN_BATCH_SIZE,
    shuffle=True,
)


# 5. Create model
num_items = int(df["movie_id"].max()) + 1

model = TransformerRecommender(
    num_items=num_items,
    max_sequence_length=MAX_SEQUENCE_LENGTH,
)


# 6. Configure training
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
)


# 7. Train
for epoch in range(NUM_EPOCHS):
    print(f"\nStarting epoch {epoch + 1}/{NUM_EPOCHS}...")

    epoch_start = time.perf_counter()

    loss = train_epoch(
        model=model,
        dataloader=training_dataloader,
        optimizer=optimizer,
        criterion=criterion,
        log_interval=TRAIN_LOG_INTERVAL,
    )

    epoch_duration = time.perf_counter() - epoch_start

    print(
        f"Epoch {epoch + 1:2d} | "
        f"Loss: {loss:.4f} | "
        f"Time: {epoch_duration:.2f}s"
    )

