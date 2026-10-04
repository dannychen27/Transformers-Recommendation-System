import time
import torch
from torch import nn
from torch.utils.data import DataLoader

from recommender.config import (
    DATA_PATH,
    DROPOUT,
    EMBEDDING_DIM,
    LEARNING_RATE,
    MAX_SEQUENCE_LENGTH,
    NUM_EPOCHS,
    NUM_HEADS,
    NUM_LAYERS,
    TRAIN_BATCH_SIZE,
    TRAIN_LOG_INTERVAL,
)
from recommender.data.dataset import SequenceDataset
from recommender.data.preprocessing import (
    build_user_sequences,
    load_ratings,
    make_test_examples,
    make_training_examples,
    make_validation_examples,
    split_sequence,
)
from recommender.models.transformer import TransformerRecommender
from recommender.training.train import train_epoch
from recommender.training.evaluate import evaluate



# 1. Load raw interactions
df = load_ratings(DATA_PATH)


# 2. Build chronological user histories
sequences = build_user_sequences(df)


# 3. Build training examples
training_examples = []
for sequence in sequences.values():
    train_sequence, _, _ = split_sequence(sequence)

    training_examples.extend(
        make_training_examples(
            train_sequence,
            max_sequence_length=MAX_SEQUENCE_LENGTH,
        )
    )

training_dataset = SequenceDataset(
    training_examples,
    max_sequence_length=MAX_SEQUENCE_LENGTH,
)

training_dataloader = DataLoader(
    training_dataset,
    batch_size=TRAIN_BATCH_SIZE,
    shuffle=True,
)


# 4. Build validation examples
validation_examples = make_validation_examples(sequences)

validation_dataset = SequenceDataset(
    validation_examples,
    max_sequence_length=MAX_SEQUENCE_LENGTH,
)

validation_dataloader = DataLoader(
    validation_dataset,
    batch_size=TRAIN_BATCH_SIZE,
    shuffle=False,
)


# 5. Create model
num_items = int(df["movie_id"].max()) + 1

model = TransformerRecommender(
    num_items=num_items,
    max_sequence_length=MAX_SEQUENCE_LENGTH,
    embedding_dim=EMBEDDING_DIM,
    num_heads=NUM_HEADS,
    num_layers=NUM_LAYERS,
    dropout=DROPOUT,
)


# 6. Configure training
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
)


# 7. Train
best_validation_loss = float("inf")
best_model_state = None

for epoch in range(NUM_EPOCHS):
    print(f"\nStarting epoch {epoch + 1}/{NUM_EPOCHS}...")

    epoch_start = time.perf_counter()

    training_loss = train_epoch(
        model=model,
        dataloader=training_dataloader,
        optimizer=optimizer,
        criterion=criterion,
        log_interval=TRAIN_LOG_INTERVAL,
    )

    validation_loss = evaluate(
        model=model,
        dataloader=validation_dataloader,
        criterion=criterion,
    )

    if validation_loss < best_validation_loss:
        best_validation_loss = validation_loss
        best_model_state = {
            key: value.detach().clone()
            for key, value in model.state_dict().items()
        }

        print(
            f"New best model! Validation loss: "
            f"{validation_loss:.4f}"
        )

    epoch_duration = time.perf_counter() - epoch_start

    print(
        f"Epoch {epoch + 1:2d} | "
        f"Training Loss: {training_loss:.4f} | "
        f"Validation Loss: {validation_loss:.4f} | "
        f"Time: {epoch_duration:.2f}s"
    )

model.load_state_dict(best_model_state)


# 8. Evaluate on test set
test_examples = make_test_examples(sequences)

test_dataset = SequenceDataset(
    test_examples,
    max_sequence_length=MAX_SEQUENCE_LENGTH,
)

test_dataloader = DataLoader(
    test_dataset,
    batch_size=TRAIN_BATCH_SIZE,
    shuffle=False,
)

test_loss = evaluate(
    model=model,
    dataloader=test_dataloader,
    criterion=criterion,
)

print(f"\nTest Loss: {test_loss:.4f}")

