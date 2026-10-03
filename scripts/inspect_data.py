from torch.utils.data import DataLoader

from recommender.config import (
    BATCH_SIZE,
    DATA_PATH,
    MAX_SEQUENCE_LENGTH,
    TEST_USER_ID,
)
from recommender.data.dataset import SequenceDataset
from recommender.data.preprocessing import (
    build_user_sequences,
    load_ratings,
    make_training_examples,
    split_sequence,
)


# 1. Load raw interactions
df = load_ratings(DATA_PATH)


# 2. Build chronological user histories
sequences = build_user_sequences(df)


# 3. Select a user and create a temporal train/validation/test split
sequence = sequences[TEST_USER_ID]
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
    batch_size=BATCH_SIZE,
    shuffle=True,
)


# 7. Retrieve one batch
input_ids, targets = next(iter(dataloader))


# 8. Verify that the preprocessing script produces a
# correctly shaped batch.
print("\nBatch shapes:")
print("Input shape:", input_ids.shape)
print("Target shape:", targets.shape)
