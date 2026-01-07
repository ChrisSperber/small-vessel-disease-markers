"""Model for tabular SVD autoencoder with auxiliary NIHSS head."""

import torch
from torch import nn


class TabularAutoencoderWithAuxHead(nn.Module):
    """A simple MLP autoencoder for tabular SVD markers with NIHSS auxiliary head."""

    def __init__(
        self,
        n_features: int,
        latent_dim: int = 1,
        hidden_dim: int = 16,
        dropout_p: float = 0.0,
    ):
        """Tabular autoencoder class.

        Args:
            n_features (int): Number of input features (SVD marker variables).
            latent_dim (int): Dimension of the compressed latent space.
            hidden_dim (int): Width of the hidden layers.
            dropout_p (float): Optional dropout probability.

        """
        super().__init__()
        self.n_features = n_features
        self.latent_dim = latent_dim

        if latent_dim < 1:
            raise ValueError("latent_dim must be >= 1")
        if n_features < 1:
            raise ValueError("n_features must be >= 1")

        drop = nn.Dropout(p=dropout_p) if dropout_p > 0 else nn.Identity()

        # Encoder (2 layers) -> latent
        self.encoder = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.ReLU(),
            drop,
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            drop,
        )
        self.fc_enc = nn.Linear(hidden_dim, latent_dim)
        self.latent_act = nn.Sigmoid()  # constrain latent to [0, 1]

        # Decoder (2 layers) -> reconstruction
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            drop,
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            drop,
            nn.Linear(hidden_dim, n_features),
        )

        # Auxiliary NIHSS head (binary classification logit)
        # Output is a logit; use BCEWithLogitsLoss in training.
        self.nihss_head = nn.Linear(latent_dim, 1)

    def forward(  # noqa: D102
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        z_raw = self.fc_enc(h)
        z = self.latent_act(z_raw)

        x_recon = self.decoder(z)
        nihss_logit = self.nihss_head(z).squeeze(1)  # (B,)

        return x_recon, nihss_logit, z

    def encode_latent(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent vector (after sigmoid)."""
        h = self.encoder(x)
        z_raw = self.fc_enc(h)
        z = self.latent_act(z_raw)
        return z

    def encode_latent_raw(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent vector (before sigmoid)."""
        h = self.encoder(x)
        z_raw = self.fc_enc(h)
        return z_raw
