"""Train tabular SVD autoencoder with auxiliary NIHSS head."""

# %%
import json
from dataclasses import asdict, dataclass
from datetime import datetime

import pandas as pd
import torch
from autoencoder_model import TabularAutoencoderWithAuxHead
from autoencoder_preproc_dataset import (
    AEPreprocessConfig,
    SVDTabularDataset,
    TabularAEPreprocessor,
)
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from svd_marker_tools.autoencoder_configs import (
    AUTOENCODER_OUTPUTS_DIR,
    FEATURE_COLS_COUNTS,
    FEATURE_COLS_ORDINAL,
    SHARED_COLS_FOR_AUTOENCODER,
    autoencoder_config,
)
from svd_marker_tools.config import RETROSPECTIVE_SAMPLE_CLEAN_CSV
from svd_marker_tools.utils import RNG_SEED, Cols

NIHSS_COL = Cols.NIHSS_24H_BINARY_GT4
MIN_DELTA = 1e-4  # minimum difference in loss to identify as an improvement

# %%
# load data
data_df = pd.read_csv(RETROSPECTIVE_SAMPLE_CLEAN_CSV, sep=";")

# %%
# assign training variables and create directories
timestamp = datetime.now().strftime("%Y%m%d_%H%M")
AUTOENCODER_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
run_output_dir = AUTOENCODER_OUTPUTS_DIR / f"svd_ae_output_{timestamp}"
run_output_dir.mkdir(parents=True, exist_ok=True)
epochs = autoencoder_config.epochs
batch_size = autoencoder_config.batch_size

if autoencoder_config.debug_mode:
    print("DEBUG MODE ENABLED: Overriding training settings.")
    epochs = 3

val_train_loss = []
val_eval_loss = []
val_recon_loss = []
val_aux_loss = []
val_aux_acc = []


# %%
@dataclass
class RunConfig:
    """Run-time config to be stored next to the model outputs."""

    rng_seed: int = RNG_SEED
    weight_decay: float = autoencoder_config.weight_decay
    warmup_epochs: int = autoencoder_config.warmup_epochs
    ramp_epochs: int = autoencoder_config.ramp_epochs
    nihss_col: str = NIHSS_COL
    timestamp: str = timestamp


run_config = RunConfig()

config_path = run_output_dir / "run_config.json"
with open(config_path, "w") as f:
    json.dump(
        {"training": asdict(autoencoder_config), "run": asdict(run_config)}, f, indent=2
    )

warmup_epochs = autoencoder_config.warmup_epochs
ramp_epochs = autoencoder_config.ramp_epochs
lambda_max = autoencoder_config.lambda_max

# due to a dynamically changing loss function, early stopping is only activated as soon as the final
# loss function is reached
early_stop_start_epoch = warmup_epochs + ramp_epochs


# %%
def lambda_aux(epoch: int) -> float:
    """Aux loss weight schedule."""
    if epoch < warmup_epochs:
        return 0.0

    if epoch < warmup_epochs + ramp_epochs:
        t = (epoch - warmup_epochs + 1) / float(ramp_epochs)
        return float(lambda_max) * float(t)

    return float(lambda_max)


# %%
def train():  # noqa: D103, PLR0915
    device = torch.device(
        "cuda"
        if torch.cuda.is_available() and autoencoder_config.device == "cuda"
        else "cpu"
    )
    print(f"Using device: {device}")

    torch.manual_seed(RNG_SEED)

    # preprocessing
    preproc_cfg = AEPreprocessConfig(
        feature_cols=SHARED_COLS_FOR_AUTOENCODER,
        nihss_col=NIHSS_COL,
        count_cols=FEATURE_COLS_COUNTS,
        ordinal_cols=FEATURE_COLS_ORDINAL,
        standardize_features=True,
        standardize_nihss=False,
    )

    # Build split deterministically (without random_split(range(...)) to keep typing happy)
    full_len = len(data_df)
    train_len = int(0.8 * full_len)
    val_len = full_len - train_len

    generator = torch.Generator().manual_seed(RNG_SEED)
    perm = torch.randperm(full_len, generator=generator).tolist()

    train_idx = perm[:train_len]
    val_idx = perm[train_len : train_len + val_len]

    df_train = data_df.iloc[train_idx].reset_index(drop=True)
    df_val = data_df.iloc[val_idx].reset_index(drop=True)

    # Fit preprocessor on TRAIN only, then persist it in run folder
    preproc = TabularAEPreprocessor(preproc_cfg).fit(df_train)
    preproc_path = run_output_dir / "preproc.json"
    preproc.save_json(preproc_path)

    train_dataset = SVDTabularDataset(df=df_train, preproc=preproc)
    val_dataset = SVDTabularDataset(df=df_val, preproc=preproc)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = TabularAutoencoderWithAuxHead(
        n_features=len(SHARED_COLS_FOR_AUTOENCODER),
        latent_dim=1,
        hidden_dim=16,
        dropout_p=0.0,
    ).to(device)

    # Loss, optimizer, and LR on plateau initialisation
    recon_criterion = nn.MSELoss()
    aux_criterion = nn.BCEWithLogitsLoss()

    optimizer = optim.AdamW(
        model.parameters(),
        lr=autoencoder_config.lr,
        weight_decay=autoencoder_config.weight_decay,
    )

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=autoencoder_config.patience_reduce_lr,
    )

    # Early stopping
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(epochs):
        lam = lambda_aux(epoch)

        # ----- TRAIN -----
        model.train()
        train_loss = 0.0
        train_recon = 0.0
        train_aux = 0.0

        n_train = 0

        for x, y in train_loader:
            x_gpu = x.to(device)
            y_gpu = y.to(device)

            optimizer.zero_grad()
            x_recon, nihss_logit, _z = model(x_gpu)

            loss_recon = recon_criterion(x_recon, x_gpu)
            loss_aux = aux_criterion(nihss_logit, y_gpu)
            loss = loss_recon + lam * loss_aux

            loss.backward()
            optimizer.step()

            bs = x_gpu.size(0)
            n_train += bs
            train_loss += loss.item() * bs
            train_recon += loss_recon.item() * bs
            train_aux += loss_aux.item() * bs

        train_loss /= n_train
        train_recon /= n_train
        train_aux /= n_train

        # ----- VALIDATE -----
        model.eval()
        val_loss = 0.0
        val_recon = 0.0
        val_aux = 0.0
        correct = 0
        total = 0
        n_val = 0

        with torch.no_grad():
            for x, y in val_loader:
                x_gpu = x.to(device)
                y_gpu = y.to(device)

                x_recon, nihss_logit, _z = model(x_gpu)

                loss_recon = recon_criterion(x_recon, x_gpu)
                loss_aux = aux_criterion(nihss_logit, y_gpu)
                loss = loss_recon + lam * loss_aux

                bs = x_gpu.size(0)
                n_val += bs
                val_loss += loss.item() * bs
                val_recon += loss_recon.item() * bs
                val_aux += loss_aux.item() * bs

                preds = (torch.sigmoid(nihss_logit) >= 0.5).float()  # noqa: PLR2004
                correct += (preds == y_gpu).sum().item()
                total += y_gpu.numel()

        val_loss /= n_val
        val_recon /= n_val
        val_aux /= n_val
        val_acc = correct / max(1, total)
        val_aux_acc.append(val_acc)

        # Logging
        current_lr = optimizer.param_groups[0]["lr"]
        val_train_loss.append(train_loss)
        val_eval_loss.append(val_loss)
        val_recon_loss.append(val_recon)
        val_aux_loss.append(val_aux)

        print(
            f"Epoch [{epoch+1}/{epochs}] "
            f"Train: {train_loss:.6f} (recon {train_recon:.6f}, aux {train_aux:.6f}) "
            f"Val: {val_loss:.6f} (recon {val_recon:.6f}, aux {val_aux:.6f}) "
            f"AuxAcc: {val_acc:.4f} "
            f"lam: {lam:.3f} "
            f"LR: {current_lr:.6f}"
        )

        scheduler.step(val_loss)

        # Early stopping (start only after ramp is finished)
        if epoch < early_stop_start_epoch:
            print(
                f"Early stopping disabled until epoch {early_stop_start_epoch} "
                f"(current {epoch+1})."
            )
        else:  # noqa: PLR5501
            if val_loss < best_val_loss - MIN_DELTA:
                print("Validation loss improved. Saving model...")
                best_val_loss = val_loss
                patience_counter = 0
                best_weights_path = (
                    run_output_dir / f"{timestamp}_best_model_weights.pt"
                )
                torch.save(model.state_dict(), best_weights_path)
            else:
                patience_counter += 1
                print(
                    "No improvement. Patience "
                    f"{patience_counter}/{autoencoder_config.patience_early_stopping}"
                )
                if patience_counter >= autoencoder_config.patience_early_stopping:
                    print("Early stopping triggered!")
                    break

    print(f"Training finished. Best validation loss: {best_val_loss:.6f}")

    metrics_path = run_output_dir / "metrics.json"
    metrics = {
        "train_loss": val_train_loss,
        "val_loss": val_eval_loss,
        "val_recon": val_recon_loss,
        "val_aux": val_aux_loss,
        "val_aux_acc": val_aux_acc,
    }
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # save final weights in lightweight model
    lightweight_path = run_output_dir / f"{timestamp}_best_model_weights.pt"
    torch.save(model.state_dict(), lightweight_path)


if __name__ == "__main__":
    train()

# %%
