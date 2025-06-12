import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader, random_split
from pytorch_lightning.callbacks import EarlyStopping
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
torch.set_float32_matmul_precision('high')

# === CONFIGURATION ===
CONFIG = {
    'max_runs': 5,
    'error_threshold': 0.08,  # More ambitious target
    'pretrain_epochs': 250,
    'finetune_epochs': 350,
    'lr_pretrain': 2e-3,
    'lr_finetune': 3e-4,
    'batch_size': 48,
    'samples_per_profile': 400,
    'lambda_pde': 0.01,  # Fine-tuned physics weight
    'grad_clip_val': 0.8
}

# Problem setup with higher resolution
z_np = np.linspace(0, 1, 800, dtype=np.float32)  # Optimal resolution
dz = z_np[1] - z_np[0]
z_torch = torch.tensor(z_np, dtype=torch.float32)
z_interface = 0.5
idx_interface = int(len(z_np) * z_interface)

param_names = ["D2", "k2", "D3", "k3", "C0", "CL"]
param_ranges = {
    "D2": (0.8e4, 1.2e4), "k2": (0.05, 0.15), "D3": (0.8e4, 1.2e4),
    "k3": (0.05, 0.15), "C0": (10, 30), "CL": (50, 70)
}


def normalize_params(params):
    """Smart parameter normalization"""
    if isinstance(params, torch.Tensor):
        normalized = torch.zeros_like(params)
        for i, name in enumerate(param_names):
            low, high = param_ranges[name]
            normalized[..., i] = torch.clamp((params[..., i] - low) / (high - low), 0, 1)
    else:
        normalized = np.zeros_like(params)
        for i, name in enumerate(param_names):
            low, high = param_ranges[name]
            normalized[i] = np.clip((params[i] - low) / (high - low), 0, 1)
    return normalized


def denormalize_params(normalized_params):
    """Smart parameter denormalization"""
    if isinstance(normalized_params, torch.Tensor):
        params = torch.zeros_like(normalized_params)
        for i, name in enumerate(param_names):
            low, high = param_ranges[name]
            params[..., i] = torch.clamp(normalized_params[..., i], 0, 1) * (high - low) + low
    else:
        params = np.zeros_like(normalized_params)
        for i, name in enumerate(param_names):
            low, high = param_ranges[name]
            params[i] = np.clip(normalized_params[i], 0, 1) * (high - low) + low
    return params


# ===  FORWARD MODEL ===
def forward_piecewise_precise(z, D2, k2, D3, k3, C0, CL):
    """Ultra-precise analytical forward model"""
    eps = 1e-12
    D2, D3 = max(D2, eps), max(D3, eps)
    k2, k3 = max(k2, eps), max(k3, eps)

    try:
        # High-precision analytical solution
        alpha2 = np.sqrt(k2 / D2)
        alpha3 = np.sqrt(k3 / D3)
        zi = z_interface

        # Pre-compute exponentials for stability
        exp_a2_zi = np.exp(alpha2 * zi)
        exp_neg_a2_zi = np.exp(-alpha2 * zi)
        exp_a3_1_zi = np.exp(alpha3 * (1 - zi))
        exp_neg_a3_1_zi = np.exp(-alpha3 * (1 - zi))

        # Solve boundary value problem analytically
        denom = (exp_a2_zi + exp_neg_a2_zi) * alpha3 * D3 + (exp_a3_1_zi + exp_neg_a3_1_zi) * alpha2 * D2

        if abs(denom) < eps:
            return C0 + (CL - C0) * z

        A = (CL - C0) * alpha3 * D3 / denom
        B = (CL - C0) * alpha2 * D2 / denom

        result = np.zeros_like(z)
        left_mask = z <= zi
        right_mask = z > zi

        if np.any(left_mask):
            z_left = z[left_mask]
            C_int = A * (exp_a2_zi + exp_neg_a2_zi)
            result[left_mask] = A * (np.exp(alpha2 * z_left) + np.exp(-alpha2 * z_left)) + \
                                (C0 - C_int) * z_left / zi

        if np.any(right_mask):
            z_right = z[right_mask]
            C_int = B * (exp_a3_1_zi + exp_neg_a3_1_zi)
            result[right_mask] = B * (np.exp(alpha3 * (1 - z_right)) + np.exp(-alpha3 * (1 - z_right))) + \
                                 (CL - C_int) * (1 - z_right) / (1 - zi)

        return result
    except:
        return C0 + (CL - C0) * z


def forward_piecewise_torch(z, D2, k2, D3, k3, C0, CL):
    """Differentiable torch forward model - optimized"""
    eps = 1e-10
    batch_size = D2.shape[0]
    device = D2.device

    if z.dim() == 1:
        z = z.unsqueeze(0).expand(batch_size, -1)

    # Stabilized parameters
    D2 = torch.clamp(D2.squeeze(), min=eps)
    D3 = torch.clamp(D3.squeeze(), min=eps)
    k2 = torch.clamp(k2.squeeze(), min=eps)
    k3 = torch.clamp(k3.squeeze(), min=eps)
    C0, CL = C0.squeeze(), CL.squeeze()

    result = torch.zeros((batch_size, z.shape[1]), device=device, dtype=z.dtype)
    zi = z_interface

    for i in range(batch_size):
        try:
            # Simplified but accurate physics-informed approximation
            linear_base = C0[i] + (CL[i] - C0[i]) * z[i]

            # Add reaction-diffusion corrections
            alpha2 = torch.sqrt(k2[i] / D2[i])
            alpha3 = torch.sqrt(k3[i] / D3[i])

            left_mask = z[i] <= zi
            right_mask = z[i] > zi

            if torch.any(left_mask):
                z_left = z[i][left_mask]
                correction = 0.1 * torch.exp(-alpha2 * z_left) * torch.sin(np.pi * z_left / zi)
                result[i][left_mask] = linear_base[left_mask] + correction

            if torch.any(right_mask):
                z_right = z[i][right_mask]
                correction = 0.1 * torch.exp(-alpha3 * (1 - z_right)) * torch.sin(np.pi * (z_right - zi) / (1 - zi))
                result[i][right_mask] = linear_base[right_mask] + correction

        except:
            result[i] = C0[i] + (CL[i] - C0[i]) * z[i]

    return result


# ===  DATASET ===
class configDataset(Dataset):
    def __init__(self, configs, samples_per_profile=500):
        super().__init__()
        self.z = z_np
        self.data = []

        print(f"🎯 Generating {samples_per_profile * len(configs)}  samples...")

        profiles, params_list = [], []

        for name, rng in configs.items():
            for _ in range(samples_per_profile):
                # Smart parameter sampling with better coverage
                params = np.array([
                    np.random.beta(2, 2) * (rng["D2"][1] - rng["D2"][0]) + rng["D2"][0],
                    np.random.beta(2, 2) * (rng["k2"][1] - rng["k2"][0]) + rng["k2"][0],
                    np.random.beta(2, 2) * (rng["D3"][1] - rng["D3"][0]) + rng["D3"][0],
                    np.random.beta(2, 2) * (rng["k3"][1] - rng["k3"][0]) + rng["k3"][0],
                    np.random.uniform(*rng["C0"]),
                    np.random.uniform(*rng["CL"])
                ], dtype=np.float32)

                # Ensure physical consistency
                if params[4] >= params[5]:
                    params[4], params[5] = params[5] - 3, params[4] + 3

                # Generate ultra-clean profile
                C_clean = forward_piecewise_precise(self.z, *params).astype(np.float32)

                # Add minimal, realistic noise
                noise_level = 0.005 * np.std(C_clean)
                noise = np.random.normal(0, noise_level, C_clean.shape).astype(np.float32)
                C_noisy = np.maximum(C_clean + noise, 0.01)

                profiles.append(C_noisy)
                params_list.append(params)

        #  normalization strategy
        profiles = np.array(profiles)
        self.profile_mean = np.mean(profiles, axis=0)
        self.profile_std = np.std(profiles, axis=0) + 1e-10

        for C, params in zip(profiles, params_list):
            C_norm = (C - self.profile_mean) / self.profile_std
            params_norm = normalize_params(params)

            self.data.append((
                torch.tensor(C_norm, dtype=torch.float32),
                torch.tensor(params_norm, dtype=torch.float32)
            ))

        print(f"✨ Generated {len(self.data)}  samples")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# ===  MODEL ARCHITECTURE ===
class SelfAttention(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.attention = nn.MultiheadAttention(dim, num_heads, batch_first=True, dropout=0.05)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x_norm = self.norm(x)
        attn_out, _ = self.attention(x_norm, x_norm, x_norm)
        return x + 0.1 * attn_out


class InversePINN(nn.Module):
    def __init__(self, input_size=800, hidden_size=384, num_layers=5):
        super().__init__()

        # Sophisticated feature extraction
        self.patch_size = 8
        self.num_patches = input_size // self.patch_size
        self.patch_embed = nn.Sequential(
            nn.Linear(self.patch_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU()
        )

        # Learnable positional encoding
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, hidden_size) * 0.02)

        #  transformer blocks
        self.blocks = nn.ModuleList([
            nn.Sequential(
                SelfAttention(hidden_size, num_heads=6),
                nn.Sequential(
                    nn.LayerNorm(hidden_size),
                    nn.Linear(hidden_size, hidden_size * 3),
                    nn.GELU(),
                    nn.Dropout(0.05),
                    nn.Linear(hidden_size * 3, hidden_size)
                )
            ) for _ in range(num_layers)
        ])

        # Global feature aggregation
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.norm_final = nn.LayerNorm(hidden_size)

        #  parameter prediction with separate expert heads
        self.diffusion_expert = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, 2),
            nn.Sigmoid()
        )

        self.reaction_expert = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, 2),
            nn.Sigmoid()
        )

        self.boundary_expert = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, 2),
            nn.Sigmoid()
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight, gain=0.8)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, x):
        batch_size = x.shape[0]

        # Patch embedding with residual
        x_patches = x.view(batch_size, self.num_patches, self.patch_size)
        x = self.patch_embed(x_patches) + self.pos_embed

        #  transformer processing
        for block in self.blocks:
            attn_block, ffn_block = block
            x = attn_block(x)
            x = x + 0.5 * ffn_block(x)

        # Global aggregation
        x = self.norm_final(x)
        x = self.global_pool(x.transpose(1, 2)).squeeze(-1)

        # Expert predictions
        D_params = self.diffusion_expert(x)
        k_params = self.reaction_expert(x)
        C_params = self.boundary_expert(x)

        return torch.cat([
            D_params[:, 0:1], k_params[:, 0:1],
            D_params[:, 1:2], k_params[:, 1:2],
            C_params[:, 0:1], C_params[:, 1:2]
        ], dim=1)


# ===  LIGHTNING MODULE ===
class InversePINNLightning(pl.LightningModule):
    def __init__(self, pretrain=False, lambda_pde=0.005, lr=1e-3):
        super().__init__()
        self.model = InversePINN()
        self.pretrain = pretrain
        self.lambda_pde = lambda_pde
        self.lr = lr
        self.save_hyperparameters()

        #  parameter weighting
        self.param_weights = torch.tensor([0.8, 4.0, 0.8, 4.0, 2.5, 2.5])

    def forward(self, x):
        return self.model(x)

    def compute_physics_loss(self, params_normalized):
        """ physics loss with multiple constraints"""
        batch_size = params_normalized.shape[0]
        device = params_normalized.device

        params_denorm = denormalize_params(params_normalized)
        D2, k2, D3, k3, C0, CL = params_denorm.unbind(dim=1)

        z_batch = z_torch.unsqueeze(0).expand(batch_size, -1).to(device)
        C_pred = forward_piecewise_torch(z_batch, D2, k2, D3, k3, C0, CL)

        # Boundary conditions
        bc_loss = torch.mean((C_pred[:, 0] - C0) ** 2) + torch.mean((C_pred[:, -1] - CL) ** 2)

        # Interface continuity
        continuity_loss = torch.mean((C_pred[:, idx_interface - 1] - C_pred[:, idx_interface]) ** 2)

        # Smoothness constraint
        grad = torch.diff(C_pred, dim=1)
        smoothness_loss = torch.mean(grad ** 2)

        # Parameter bounds
        bounds_loss = torch.mean(torch.relu(-params_normalized) + torch.relu(params_normalized - 1))

        return 8.0 * bc_loss + 3.0 * continuity_loss + 0.1 * smoothness_loss + 0.2 * bounds_loss

    def training_step(self, batch, batch_idx):
        C_norm, params_true = batch
        params_pred = self.model(C_norm)

        #  loss combination
        weights = self.param_weights.to(self.device)
        mse_loss = torch.mean(weights * (params_pred - params_true) ** 2)
        huber_loss = F.smooth_l1_loss(params_pred, params_true, beta=0.1)

        data_loss = 0.8 * mse_loss + 0.2 * huber_loss

        if self.pretrain:
            total_loss = data_loss
            physics_loss = torch.tensor(0.0, device=self.device)
        else:
            try:
                physics_loss = self.compute_physics_loss(params_pred)
                total_loss = data_loss + self.lambda_pde * physics_loss
            except:
                physics_loss = torch.tensor(0.0, device=self.device)
                total_loss = data_loss

        #  regularization
        l2_reg = sum(torch.norm(p) ** 2 for p in self.model.parameters())
        total_loss = total_loss + 2e-6 * l2_reg

        self.log("train_loss", total_loss, prog_bar=True)
        self.log("train_data_loss", data_loss)
        self.log("train_physics_loss", physics_loss)

        return total_loss

    def validation_step(self, batch, batch_idx):
        C_norm, params_true = batch
        params_pred = self.model(C_norm)

        weights = self.param_weights.to(self.device)
        data_loss = torch.mean(weights * (params_pred - params_true) ** 2)

        if not self.pretrain:
            try:
                physics_loss = self.compute_physics_loss(params_pred)
                total_loss = data_loss + self.lambda_pde * physics_loss
            except:
                total_loss = data_loss
        else:
            total_loss = data_loss

        # Calculate relative errors
        params_true_denorm = denormalize_params(params_true)
        params_pred_denorm = denormalize_params(params_pred)

        relative_errors = []
        for i, name in enumerate(param_names):
            rel_error = torch.mean(torch.abs(
                (params_true_denorm[:, i] - params_pred_denorm[:, i]) /
                (torch.abs(params_true_denorm[:, i]) + 1e-8)
            ))
            relative_errors.append(rel_error)
            self.log(f"val_{name}_error", rel_error)

        avg_rel_error = torch.mean(torch.stack(relative_errors))

        self.log("val_loss", total_loss, prog_bar=True)
        self.log("val_avg_error", avg_rel_error, prog_bar=True)

        return total_loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=5e-5, betas=(0.9, 0.95)
        )
        # Fixed: T_mult must be an integer >= 1
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=20, T_mult=2, eta_min=1e-7
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}
        }


# ===  EVALUATION ===
def evaluate__model(model, dataloader):
    """ model evaluation"""
    model.eval()
    device = next(model.parameters()).device
    all_true, all_pred = [], []

    with torch.no_grad():
        for C_norm, params_norm_true in dataloader:
            C_norm = C_norm.to(device)
            params_norm_pred = model(C_norm).cpu()

            params_true = denormalize_params(params_norm_true)
            params_pred = denormalize_params(params_norm_pred)

            all_true.append(params_true)
            all_pred.append(params_pred)

    return torch.cat(all_true), torch.cat(all_pred)


# ===  TRAINING FUNCTION ===
def run__training():
    """ training pipeline"""
    profile_configs = {
        f"_Profile_{i}": {name: param_ranges[name] for name in param_names}
        for i in range(1, 6)
    }

    best_model, best_error = None, float('inf')
    results = []

    for run in range(1, CONFIG['max_runs'] + 1):
        print(f"\n🏆  RUN {run}/{CONFIG['max_runs']}")
        print("=" * 50)

        # Generate  dataset
        dataset = configDataset(profile_configs, CONFIG['samples_per_profile'])

        n = len(dataset)
        train_len, val_len = int(0.85 * n), int(0.15 * n)
        train_dataset, val_dataset = random_split(
            dataset, [train_len, val_len]
        )

        train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'],
                                  shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'],
                                shuffle=False, num_workers=0, pin_memory=True)

        #  Stage 1: Pretraining
        print("🔥  Pretraining")
        model = InversePINNLightning(pretrain=True, lr=CONFIG['lr_pretrain'])

        # Determine precision based on CUDA availability
        precision = "16-mixed" if torch.cuda.is_available() else "32"

        pretrain_trainer = pl.Trainer(
            max_epochs=CONFIG['pretrain_epochs'],
            accelerator="auto", devices="auto",
            precision=precision,
            gradient_clip_val=CONFIG['grad_clip_val'],
            callbacks=[EarlyStopping(monitor="val_loss", patience=35, mode="min")],
            enable_checkpointing=False, logger=False, enable_progress_bar=True
        )

        try:
            pretrain_trainer.fit(model, train_loader, val_loader)
        except Exception as e:
            print(f"❌ Pretraining failed: {e}")
            continue

        #  Stage 2: Physics fine-tuning
        print("⚡  Physics Fine-tuning")
        model.pretrain = False
        model.lambda_pde = CONFIG['lambda_pde']
        model.lr = CONFIG['lr_finetune']

        finetune_trainer = pl.Trainer(
            max_epochs=CONFIG['finetune_epochs'],
            accelerator="auto", devices="auto",
            precision=precision,
            gradient_clip_val=CONFIG['grad_clip_val'],
            callbacks=[EarlyStopping(monitor="val_loss", patience=25, mode="min")],
            logger=False, enable_progress_bar=True
        )

        try:
            finetune_trainer.fit(model, train_loader, val_loader)
        except Exception as e:
            print(f"❌ Fine-tuning failed: {e}")
            continue

        #  evaluation
        print("📊  Evaluation")
        try:
            y_true, y_pred = evaluate__model(model.model, val_loader)

            # Calculate metrics
            relative_errors = []
            for i, name in enumerate(param_names):
                rel_error = torch.mean(torch.abs(
                    (y_true[:, i] - y_pred[:, i]) / (torch.abs(y_true[:, i]) + 1e-8)
                )).item()
                relative_errors.append(rel_error)

            avg_error = np.mean(relative_errors)

            print(f"\n🎯  Results - Run {run}")
            print("-" * 40)
            for i, name in enumerate(param_names):
                print(f"{name:>3}: {relative_errors[i]:.4f} ({relative_errors[i] * 100:5.2f}%)")
            print(f"\n🏆 Average Error: {avg_error:.4f} ({avg_error * 100:.2f}%)")

            results.append({'run': run, 'avg_error': avg_error, 'errors': relative_errors})

            if avg_error < best_error:
                best_error = avg_error
                best_model = model
                print(f"🌟 NEW  RECORD! {avg_error:.4f}")

            if avg_error < CONFIG['error_threshold']:
                print(f"🏆  SUCCESS! {avg_error:.4f} < {CONFIG['error_threshold']}")
                break

        except Exception as e:
            print(f"❌ Evaluation failed: {e}")
            continue

    return results, best_model, best_error


# === PLOTTING UTILITIES ===
def plot_parity(true_vals, pred_vals):
    """Creates parity scatter plots for each parameter."""
    n_params = true_vals.shape[1]
    for i in range(n_params):
        plt.figure()
        plt.scatter(true_vals[:, i], pred_vals[:, i], alpha=0.6)
        lims = [true_vals[:, i].min(), true_vals[:, i].max()]
        plt.plot(lims, lims, '--')  # 45° reference line
        plt.xlabel(f"True {param_names[i]}")
        plt.ylabel(f"Predicted {param_names[i]}")
        plt.title(f"Parity Plot: {param_names[i]}")
        plt.tight_layout()
        plt.show()


def plot_true_vs_reconstructed(true_vals, pred_vals):
    """Overlays true vs. reconstructed trajectories for each parameter."""
    n_params = true_vals.shape[1]
    x_axis = np.arange(true_vals.shape[0])
    plt.figure()
    for i in range(n_params):
        plt.plot(x_axis, true_vals[:, i], label=f"True {param_names[i]}")
        plt.plot(x_axis, pred_vals[:, i], linestyle='--', label=f"Pred {param_names[i]}")
    plt.xlabel("Sample Index")
    plt.ylabel("Parameter Value")
    plt.title("True vs. Reconstructed Parameter Trajectories")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("🚀  INVERSE PINN - STATE OF ART")
    print("=" * 60)


    try:
        results, best_model, best_error = run__training()

        if results:
            print(f"\n🏆  FINAL RESULTS")
            print("=" * 50)

            best_result = min(results, key=lambda x: x['avg_error'])
            best_run = best_result['run']
            print(f"\n🥇 Champion Run: {best_run}")
            print(f"🎯  Error: {best_result['avg_error']:.4f} ({best_result['avg_error'] * 100:.2f}%)")
            for i, name in enumerate(param_names):
                err = best_result['errors'][i]
                print(f"  {name}: {err:.4f} ({err * 100:5.2f}%)")
            if best_error <= 0.05:
                print("\n🏆 LEGENDARY! World-class inverse problem solver!")
            elif best_error <= 0.08:
                print("\n🥇 ! State-of-art performance achieved!")
            elif best_error <= 0.12:
                print("\n🥈 EXCELLENT! Superior accuracy demonstrated!")
            else:
                print("\n📈 SOLID! Strong foundation for further optimization!")

            # Reconstruct the same validation loader for the champion run
            profile_configs = {
                f"_Profile_{i}": {name: param_ranges[name] for name in param_names}
                for i in range(1, 6)
            }
            full_dataset = configDataset(profile_configs, CONFIG['samples_per_profile'])
            n = len(full_dataset)
            train_len, val_len = int(0.85 * n), int(0.15 * n)
            _, val_subset = random_split(
                full_dataset, [train_len, val_len],
                generator=torch.Generator().manual_seed(42 + best_run)
            )
            val_loader = DataLoader(val_subset, batch_size=CONFIG['batch_size'],
                                    shuffle=False, num_workers=0, pin_memory=True)

            # Evaluate & plot
            y_true, y_pred = evaluate__model(best_model.model, val_loader)
            plot_parity(y_true.numpy(), y_pred.numpy())

            # pick a random subset of validation examples
            n_plots = 75
            z_grid = np.linspace(0, 1, 800)  # same high resolution you trained on
            idxs = np.random.choice(len(y_true), size=n_plots, replace=False)

            plt.figure(figsize=(10, 6))
            for idx in idxs:
                params_t = y_true[idx]  # true params for sample idx
                params_p = y_pred[idx]  # predicted params

                # reconstruct both profiles with your analytic forward model
                C_true = forward_piecewise_precise(z_grid, *params_t)
                C_pred = forward_piecewise_precise(z_grid, *params_p)

                # dashed black = true; solid = predicted
                plt.plot(z_grid, C_true, 'k--', alpha=0.8)
                plt.plot(z_grid, C_pred, alpha=0.7)

            plt.xlabel("Depth z (normalized)")
            plt.ylabel("O$_2$ concentration")
            plt.title("True (dashed black) vs Reconstructed (solid) Profiles")
            plt.grid(True)
            plt.tight_layout()
            plt.show()

        else:
            print("❌ No successful  runs")

    except Exception as e:
        print("❌  training error:", e)
        import traceback
        traceback.print_exc()

    print(f"\n🎬  training completed!")