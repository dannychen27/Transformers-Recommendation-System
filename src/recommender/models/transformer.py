import torch
from torch import nn


class TransformerRecommender(nn.Module):
    """Transformer-based sequential recommender for next-item prediction."""

    def __init__(
        self,
        num_items: int,
        max_sequence_length: int = 50,
        embedding_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        pad_token: int = 0,
    ) -> None:
        """Initialize the Transformer recommender.

        Args:
            num_items: Number of item IDs, including the padding token.
            max_sequence_length: Maximum input sequence length.
            embedding_dim: Dimension of item and positional embeddings.
            num_heads: Number of self-attention heads.
            num_layers: Number of Transformer encoder layers.
            dropout: Dropout probability used in the Transformer.
            pad_token: Item ID used for padding.
        """
        super().__init__()

        self.max_sequence_length = max_sequence_length
        self.pad_token = pad_token

        self.item_embedding = nn.Embedding(
            num_items,
            embedding_dim,
            padding_idx=pad_token,
        )

        self.position_embedding = nn.Embedding(
            max_sequence_length,
            embedding_dim,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.output = nn.Linear(
            embedding_dim,
            num_items,
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Compute next-item prediction logits.

        Args:
            input_ids: Tensor of movie IDs with shape
                ``(batch_size, sequence_length)``.

        Returns:
            Tensor of prediction logits with shape
            ``(batch_size, num_items)``.
        """
        _, sequence_length = input_ids.shape

        if sequence_length > self.max_sequence_length:
            raise ValueError(
                "Input sequence length exceeds max_sequence_length."
            )

        padding_mask = input_ids.eq(self.pad_token)

        positions = torch.arange(
            sequence_length,
            device=input_ids.device,
        ).unsqueeze(0)

        x = self.item_embedding(input_ids)
        x = x + self.position_embedding(positions)

        x = self.encoder(
            x,
            src_key_padding_mask=padding_mask,
        )

        # Because we left-pad, the final position contains
        # the most recent real interaction.
        x = x[:, -1, :]

        return self.output(x)
