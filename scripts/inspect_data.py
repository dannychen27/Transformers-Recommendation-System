import torch
from torch.utils.data import DataLoader

from recommender.config import (
    INSPECT_BATCH_SIZE,
    DATA_PATH,
    LEARNING_RATE,
    MAX_SEQUENCE_LENGTH,
    INSPECT_USER_ID,
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


# 1. Load raw interactions
df = load_ratings(DATA_PATH)


# 2. Build chronological user histories
sequences = build_user_sequences(df)


# 3. Select a user and create a temporal train/validation/test split
sequence = sequences[INSPECT_USER_ID]
train_sequence, validation_target, test_target = split_sequence(sequence)


# 4. Generate next-item training examples
training_examples = make_training_examples(train_sequence)


# 5. Create PyTorch training dataset
training_dataset = SequenceDataset(
    training_examples,
    max_sequence_length=MAX_SEQUENCE_LENGTH,
)

# 6. Batch training examples
dataloader = DataLoader(
    training_dataset,
    batch_size=INSPECT_BATCH_SIZE,
    shuffle=True,
)


# 7a. Retrieve one batch
input_ids, targets = next(iter(dataloader))


# 7b. Verify that the preprocessing script produces a
# correctly shaped batch.
print("\nBatch shapes:")
print("Input shape:", input_ids.shape)
print("Target shape:", targets.shape)


# 8. Run one batch through the Transformer
num_items = int(df["movie_id"].max()) + 1

model = TransformerRecommender(
    num_items=num_items,
    max_sequence_length=MAX_SEQUENCE_LENGTH,
)

logits = model(input_ids)

print("\nTransformer output:")
print("Logits shape:", logits.shape)


# 9. Create the Transformer
num_items = int(df["movie_id"].max()) + 1

model = TransformerRecommender(
    num_items=num_items,
    max_sequence_length=MAX_SEQUENCE_LENGTH,
)


# 10. Configure training
criterion = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
)


# 11. Perform one training step
loss = train_step(
    model=model,
    input_ids=input_ids,
    targets=targets,
    optimizer=optimizer,
    criterion=criterion,
)

print("\nTraining step:")
print("Loss:", loss)

