## Project Checklist

### Setup
- [x] Set up Python project structure
- [x] Configure PyTorch and project dependencies
- [x] Add editable package installation
- [x] Add Git ignore rules

### Data preprocessing
- [x] Load MovieLens 100K interaction data
- [x] Build chronological user interaction sequences
- [x] Create temporal train/validation/test splits
- [x] Generate prefix-based next-item training examples
- [x] Add padded and truncated PyTorch Dataset
- [x] Add DataLoader batching
- [x] Add unit tests for preprocessing and dataset behavior

### Transformer model
- [ ] Implement item embeddings
- [ ] Add positional embeddings
- [ ] Add padding-aware Transformer encoder
- [ ] Produce next-item prediction logits
- [ ] Add model unit tests

### Training
- [ ] Implement training loop
- [ ] Add validation
- [ ] Add checkpointing
- [ ] Track training/validation loss

### Evaluation
- [ ] Implement popularity baseline
- [ ] Implement Recall@K
- [ ] Implement NDCG@K
- [ ] Compare Transformer against baseline

### Experiments
- [ ] Tune sequence length
- [ ] Tune Transformer architecture
- [ ] Run ablation experiments
- [ ] Document final results