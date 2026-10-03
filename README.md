# Transformers Recommendation System

A transformer-based sequential recommendation system that predicts
the next item a user is likely to interact with.


## Architecture

```text
User interaction history
→ item embeddings
→ transformer encoder
→ next-item prediction
```

## Project Status

See [PROJECT_STATUS.md](PROJECT_STATUS.md) for the current implementation checklist and progress.


## Dataset

This project uses the [MovieLens 100K dataset](https://grouplens.org/datasets/movielens/100k/).

Download `ml-100k.zip` from GroupLens and extract it to:

```text
data/raw/ml-100k/
```

The dataset is excluded from version control.


## Data Pipeline

```text
MovieLens u.data
    ↓
Load ratings
    ↓
Chronological user sequences
    ↓
Temporal train / validation / test split
    ↓
Prefix → next-item training examples
    ↓
Padding / truncation
    ↓
PyTorch Dataset
    ↓
DataLoader
```


## Setup


## Training


## Evaluation


## Results

