import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import os
import warnings

# Suppress scientific notation and warnings
np.set_printoptions(suppress=True)
warnings.filterwarnings('ignore')


# ----------------- 1) Forward model (piecewise) --------------------
def forward_piecewise(z, D2, k2, D3, k3, C0, CL, z_interface=0.5):
    z = np.asarray(z)
    A2 = (CL - C0 + (k2 / D2) * z_interface * C0 + (k3 / D3) * (1 - z_interface) * CL) / (
            z_interface + (k2 / D2) * z_interface ** 2 + (k3 / D3) * (1 - z_interface) ** 2
    )
    B2 = C0 - (k2 / D2) * z_interface * A2
    A3 = A2
    B3 = CL - (k3 / D3) * (1 - z_interface) * A3

    C = np.where(
        z <= z_interface,
        A2 * z + B2 - k2 / D2 * (z_interface - z) * A2,
        A3 * z + B3 - k3 / D3 * (z - z_interface) * A3
    )
    return C


# ----------------- 2) Synthetic dataset with memory optimization -----------------
class InversePINNDataset(Dataset):
    def __init__(self, profile_configs, samples_per_profile=200, z_points=100, save_dir="synthetic_profiles"):
        self.z = np.linspace(0, 1, z_points)
        self.sample_indices = []
        self.save_dir = save_dir

        # Create main save directory
        os.makedirs(save_dir, exist_ok=True)

        total = 0
        for profile_name, param_ranges in profile_configs.items():
            # Create profile-specific subfolder
            profile_dir = os.path.join(save_dir, profile_name)
            os.makedirs(profile_dir, exist_ok=True)

            for i in range(samples_per_profile):
                # Instead of storing all data in memory, just store paths
                self.sample_indices.append((profile_name, total))

                # Sample parameters
                D2 = np.random.uniform(*param_ranges["D2"])
                D3 = np.random.uniform(*param_ranges["D3"])
                k2 = np.random.uniform(*param_ranges["k2"])
                k3 = np.random.uniform(*param_ranges["k3"])
                C0 = np.random.uniform(*param_ranges["C0"])
                CL = np.random.uniform(*param_ranges["CL"])

                # Generate concentration profile
                C = forward_piecewise(self.z, D2, k2, D3, k3, C0, CL)

                # Save to file immediately without storing in memory
                save_path = os.path.join(profile_dir, f"profile_{total:04d}.npz")
                np.savez(save_path,
                         z=self.z,
                         C=C.astype(np.float32),
                         params=np.array([D2, k2, D3, k3, C0, CL], dtype=np.float32))
                total += 1

    def __len__(self):
        return len(self.sample_indices)

    def __getitem__(self, idx):
        profile_name, idx_val = self.sample_indices[idx]
        data = np.load(os.path.join(self.save_dir, profile_name, f"profile_{idx_val:04d}.npz"))
        return torch.tensor(data['C']), torch.tensor(data['params'])


# ----------------- 3) Improved Neural Network Model -------------------------
class InversePINNModel(nn.Module):
    def __init__(self, input_dim=100, hidden_dim=256, output_dim=6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)


# ----------------- 4) LightningModule ------------------------------
class InversePINNLightning(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.criterion = nn.MSELoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.criterion(y_hat, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        val_loss = self.criterion(y_hat, y)
        self.log("val_loss", val_loss, prog_bar=True)
        return val_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1
            }
        }


# ----------------- 5) LightningDataModule --------------------------
class InversePINNDataModule(pl.LightningDataModule):
    def __init__(self, dataset, batch_size=16):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size

    def setup(self, stage=None):
        n = len(self.dataset)
        train_len = int(0.8 * n)
        val_len = n - train_len
        self.train_dataset, self.val_dataset = random_split(self.dataset, [train_len, val_len])

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=0,
            pin_memory=True
        )


# ----------------- 6) Run training and validation ------------------
if __name__ == "__main__":
    # Configuration for different profiles
    profile_configs = {
        f"Profile_{i}": {
            "D2": (0.8e4, 1.2e4),
            "D3": (0.8e4, 1.2e4),
            "k2": (0.05, 0.15),
            "k3": (0.05, 0.15),
            "C0": (10, 30),
            "CL": (50, 70)
        }
        for i in range(1, 76)
    }

    # Create dataset with memory optimization
    print("Generating dataset and saving profiles...")
    dataset = InversePINNDataset(
        profile_configs=profile_configs,
        samples_per_profile=100,
        z_points=100,
        save_dir="synthetic_profiles"
    )
    print(f"Created dataset with {len(dataset)} samples")

    # Setup model and data
    model = InversePINNModel()
    lightning_model = InversePINNLightning(model)
    dm = InversePINNDataModule(dataset, batch_size=16)

    # Configure callbacks
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=20,
        verbose=True,
        mode="min"
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath="checkpoints",
        filename="best_model",
        save_top_k=1,
        mode="min"
    )

    # Initialize Trainer
    trainer = pl.Trainer(
        max_epochs=500,
        callbacks=[early_stop, checkpoint_callback],
        accelerator="auto",
        devices="auto",
        log_every_n_steps=10,
        accumulate_grad_batches=4,
        precision='16-mixed'
    )

    print("Starting training...")
    trainer.fit(model=lightning_model, datamodule=dm)

    # Load best model for evaluation
    print("Loading best model for evaluation...")
    best_lightning = InversePINNLightning.load_from_checkpoint(
        checkpoint_callback.best_model_path,
        model=model
    )
    best_model = best_lightning.model.cpu()  # Move to CPU for evaluation
    best_model.eval()

    # ----------------- 7) Parity plots ---------------------
    print("Creating parity plots...")
    val_loader = dm.val_dataloader()
    Y_true_list, Y_pred_list = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            yh = best_model(xb)
            Y_true_list.append(yb)
            Y_pred_list.append(yh)

    Y_true = torch.cat(Y_true_list).cpu().numpy()
    Y_pred = torch.cat(Y_pred_list).cpu().numpy()
    param_names = ["D2", "k2", "D3", "k3", "C0", "CL"]

    # Normalize parameters for better visualization
    param_scales = {
        "D2": 1e4, "D3": 1e4,
        "k2": 1, "k3": 1,
        "C0": 1, "CL": 1
    }

    plt.figure(figsize=(15, 10))
    for i, name in enumerate(param_names):
        plt.subplot(2, 3, i + 1)

        scale = param_scales.get(name, 1)
        true_vals = Y_true[:, i] / scale
        pred_vals = Y_pred[:, i] / scale

        plt.scatter(true_vals, pred_vals, alpha=0.6, edgecolors='b', facecolors='none')
        min_val = min(true_vals.min(), pred_vals.min())
        max_val = max(true_vals.max(), pred_vals.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')

        plt.xlabel(f"True {name}")
        plt.ylabel(f"Predicted {name}")
        plt.title(f"Parity: {name} " + (f"(x{scale})" if scale != 1 else ""))
        plt.grid(True)
        plt.axis('square')
        plt.xlim(min_val - 0.05 * (max_val - min_val), max_val + 0.05 * (max_val - min_val))
        plt.ylim(min_val - 0.05 * (max_val - min_val), max_val + 0.05 * (max_val - min_val))

    plt.tight_layout()
    plt.savefig("parity_plots.png", dpi=300)
    print("Saved parity_plots.png")

    # ----------------- 8) Profile Reconstruction -------------------
    print("Creating profile reconstructions...")
    z_grid = np.linspace(0, 1, 400)
    n = 75
    idxs = np.random.choice(len(Y_true), size=n, replace=False)

    plt.figure(figsize=(10, 6))
    for idx in idxs:
        true_params = dict(zip(param_names, Y_true[idx]))
        pred_params = dict(zip(param_names, Y_pred[idx]))
        C_true = forward_piecewise(z_grid, **true_params)
        C_pred = forward_piecewise(z_grid, **pred_params)
        plt.plot(z_grid, C_true, 'k--', alpha=0.8)
        plt.plot(z_grid, C_pred, alpha=0.7)

    plt.xlabel("Depth z (normalized)")
    plt.ylabel("O₂ concentration")
    plt.title("True (dashed black) vs Reconstructed (solid) Profiles")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Calculate and print average relative error
    relative_errors = []
    for i in range(len(param_names)):
        abs_error = np.abs(Y_true[:, i] - Y_pred[:, i])
        rel_error = abs_error / (np.abs(Y_true[:, i]) + 1e-8)
        relative_errors.append(np.mean(rel_error))

    print("\nAverage Relative Errors:")
    for i, name in enumerate(param_names):
        print(f"{name}: {relative_errors[i]:.4f} ({relative_errors[i] * 100:.2f}%)")

    print("Done!")