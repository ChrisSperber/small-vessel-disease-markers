"""Config for autoencoder."""

from dataclasses import dataclass
from pathlib import Path

from svd_marker_tools.utils import Cols

SHARED_COLS_FOR_AUTOENCODER = [
    Cols.LACUNES,
    Cols.EPVS_BG,
    Cols.EPVS_CS,
    Cols.CMB_TOTAL,
    Cols.WMH_DEEP,
    Cols.WMH_PV,
]
# list all cols with counts to be log(x+1) deskewed
FEATURE_COLS_COUNTS = [
    Cols.LACUNES,
    Cols.CMB_TOTAL,
]
# list all ordinal rating cols to be standardized
FEATURE_COLS_ORDINAL = [
    Cols.EPVS_BG,
    Cols.EPVS_CS,
    Cols.WMH_DEEP,
    Cols.WMH_PV,
]


AUTOENCODER_OUTPUTS_DIR: Path = Path(__file__).parents[2] / "autoencoder" / "outputs"


@dataclass
class TrainingConfig:
    """Configuration container for training autoencoders.

    Attributes:
        device (str): Device to use for training ('cuda' or 'cpu').
        epochs (int): Maximum number of training epochs.
        lr (float): Learning rate for the optimizer.
        batch_size (int): Batch size.
        patience_early_stopping (int): Number of epochs without improvement before early stopping.
        patience_reduce_lr (int): Number of stagnant epochs before reducing learning rate.

    """

    device: str = "cuda"
    epochs: int = 250
    lr: float = 0.001
    batch_size: int = 128
    patience_early_stopping: int = 15
    patience_reduce_lr: int = 5
    debug_mode: bool = False
    weight_decay: float = 1e-4
    warmup_epochs: int = 5
    ramp_epochs: int = 10
    lambda_max: float = 1


autoencoder_config = TrainingConfig()
