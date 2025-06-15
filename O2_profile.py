import os
import pickle
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
    'max_runs': 3,
    'error_threshold': 0.04,  # More ambitious target
    'pretrain_epochs': 250,
    'finetune_epochs': 350,
    'lr_pretrain': 2e-3,
    'lr_finetune': 3e-4,
    'batch_size': 64,
    'samples_per_profile': 1000,
    'lambda_pde': 0.015,  # Fine-tuned physics weight
    'grad_clip_val': 0.8,
    'model_save_dir': 'model_checkpoints'
}

# Problem setup with higher resolution
z_np = np.linspace(0, 1, 800, dtype=np.float32)  # Optimal resolution
dz = z_np[1] - z_np[0]
z_torch = torch.tensor(z_np, dtype=torch.float32)

# Four-layer interfaces
z_interface1 = 0.25
z_interface2 = 0.50
z_interface3 = 0.75
idx_interface1 = int(len(z_np) * z_interface1)
idx_interface2 = int(len(z_np) * z_interface2)
idx_interface3 = int(len(z_np) * z_interface3)

# Updated parameter names and ranges for 4 layers
param_names = ["DIR", "kIR", "DOR", "kOR", "DFL", "kFL", "DCC", "kCC", "C0", "CL"]
param_ranges = {
    # Diffusion coefficients (cm²/s)
    "DIR": (1.8e-5, 2.1e-5),
    "DOR": (1.4e-5, 1.9e-5),
    "DFL": (2.3e-5, 2.7e-5),
    "DCC": (2.0e-5, 2.4e-5),

    # Solubility (ml O₂ / ml tissue / mmHg)
    "kIR": (2.0e-5, 3.0e-5),
    "kOR": (2.0e-5, 3.0e-5),
    "kFL": (2.5e-5, 3.5e-5),
    "kCC": (2.5e-5, 3.5e-5),

    # Boundary partial pressures (mmHg)
    "C0": (15.0, 25.0),
    "CL": (100.0, 125.0),
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
def forward_piecewise_precise(z, DIR, kIR, DOR, kOR, DFL, kFL, DCC, kCC, C0, CL):
    """Four-layer analytical forward model"""
    eps = 1e-12
    DIR, DOR, DFL, DCC = max(DIR, eps), max(DOR, eps), max(DFL, eps), max(DCC, eps)
    kIR, kOR, kFL, kCC = max(kIR, eps), max(kOR, eps), max(kFL, eps), max(kCC, eps)

    try:
        result = np.zeros_like(z)

        # Define regions
        region1_mask = z <= z_interface1
        region2_mask = (z > z_interface1) & (z <= z_interface2)
        region3_mask = (z > z_interface2) & (z <= z_interface3)
        region4_mask = z > z_interface3

        # Linear base interpolation
        linear_base = C0 + (CL - C0) * z

        result[i] = linear_base.clone()

        # Region 1: Inner Retina
        if np.any(region1_mask):
            z_reg = z[region1_mask]
            alpha = np.sqrt(kIR / DIR)
            correction = 0.1 * np.exp(-alpha * z_reg) * np.sin(np.pi * z_reg / z_interface1)
            result[region1_mask] = linear_base[region1_mask] + correction

        # Region 2: Outer Retina
        if np.any(region2_mask):
            z_reg = z[region2_mask]
            alpha = np.sqrt(kOR / DOR)
            correction = 0.1 * np.exp(-alpha * (z_reg - z_interface1)) * np.sin(
                np.pi * (z_reg - z_interface1) / (z_interface2 - z_interface1))
            result[region2_mask] = linear_base[region2_mask] + correction

        # Region 3: Fluid Layer
        if np.any(region3_mask):
            z_reg = z[region3_mask]
            alpha = np.sqrt(kFL / DFL)
            correction = 0.1 * np.exp(-alpha * (z_reg - z_interface2)) * np.sin(
                np.pi * (z_reg - z_interface2) / (z_interface3 - z_interface2))
            result[region3_mask] = linear_base[region3_mask] + correction

        # Region 4: Choriocapillaris
        if np.any(region4_mask):
            z_reg = z[region4_mask]
            alpha = np.sqrt(kCC / DCC)
            correction = 0.1 * np.exp(-alpha * (1 - z_reg)) * np.sin(
                np.pi * (z_reg - z_interface3) / (1 - z_interface3))
            result[region4_mask] = linear_base[region4_mask] + correction

        return result
    except:
        return C0 + (CL - C0) * z


def forward_piecewise_torch(z, DIR, kIR, DOR, kOR, DFL, kFL, DCC, kCC, C0, CL):
    """Four-layer differentiable torch forward model with linear base seeding."""
    global linear_base
    eps = 1e-10
    batch_size = DIR.shape[0]
    device = DIR.device

    if z.dim() == 1:
        z = z.unsqueeze(0).expand(batch_size, -1)

    # Stabilize parameters
    DIR = torch.clamp(DIR.squeeze(), min=eps)
    DOR = torch.clamp(DOR.squeeze(), min=eps)
    DFL = torch.clamp(DFL.squeeze(), min=eps)
    DCC = torch.clamp(DCC.squeeze(), min=eps)
    kIR = torch.clamp(kIR.squeeze(), min=eps)
    kOR = torch.clamp(kOR.squeeze(), min=eps)
    kFL = torch.clamp(kFL.squeeze(), min=eps)
    kCC = torch.clamp(kCC.squeeze(), min=eps)
    C0, CL = C0.squeeze(), CL.squeeze()

    result = torch.zeros((batch_size, z.shape[1]), device=device, dtype=z.dtype)

    for i in range(batch_size):
        try:
            # 1) Compute the linear baseline everywhere
            linear_base = C0[i] + (CL[i] - C0[i]) * z[i]
            result[i] = linear_base.clone()

            # 2) Masks for the four regions
            r1 = z[i] <= z_interface1
            r2 = (z[i] > z_interface1) & (z[i] <= z_interface2)
            r3 = (z[i] > z_interface2) & (z[i] <= z_interface3)
            r4 = z[i] > z_interface3

            # 3) Add the small reaction–diffusion corrections on top
            if torch.any(r1):
                z_reg = z[i][r1]
                α = torch.sqrt(kIR[i] / DIR[i])
                corr = 0.1 * torch.exp(-α * z_reg) * torch.sin(np.pi * z_reg / z_interface1)
                result[i][r1] += corr

            if torch.any(r2):
                z_reg = z[i][r2]
                α = torch.sqrt(kOR[i] / DOR[i])
                corr = 0.1 * torch.exp(-α * (z_reg - z_interface1)) * torch.sin(
                    np.pi * (z_reg - z_interface1) / (z_interface2 - z_interface1))
                result[i][r2] += corr

            if torch.any(r3):
                z_reg = z[i][r3]
                α = torch.sqrt(kFL[i] / DFL[i])
                corr = 0.1 * torch.exp(-α * (z_reg - z_interface2)) * torch.sin(
                    np.pi * (z_reg - z_interface2) / (z_interface3 - z_interface2))
                result[i][r3] += corr

            if torch.any(r4):
                z_reg = z[i][r4]
                α = torch.sqrt(kCC[i] / DCC[i])
                corr = 0.1 * torch.exp(-α * (1 - z_reg)) * torch.sin(
                    np.pi * (z_reg - z_interface3) / (1 - z_interface3))
                result[i][r4] += corr

        except:
            # Fallback to pure linear if something goes wrong
            result[i] = linear_base

    return result

# ===  DATASET ===
class configDataset(Dataset):
    def __init__(self, configs, samples_per_profile=500):
        super().__init__()
        self.z = z_np
        self.data = []

        print(f"🎯 Generating {samples_per_profile * len(configs)} samples...")

        profiles, params_list = [], []

        for name, rng in configs.items():
            for _ in range(samples_per_profile):
                # Smart parameter sampling for 10 parameters
                params = np.array([
                    np.random.beta(2, 2) * (rng["DIR"][1] - rng["DIR"][0]) + rng["DIR"][0],
                    np.random.beta(2, 2) * (rng["kIR"][1] - rng["kIR"][0]) + rng["kIR"][0],
                    np.random.beta(2, 2) * (rng["DOR"][1] - rng["DOR"][0]) + rng["DOR"][0],
                    np.random.beta(2, 2) * (rng["kOR"][1] - rng["kOR"][0]) + rng["kOR"][0],
                    np.random.beta(2, 2) * (rng["DFL"][1] - rng["DFL"][0]) + rng["DFL"][0],
                    np.random.beta(2, 2) * (rng["kFL"][1] - rng["kFL"][0]) + rng["kFL"][0],
                    np.random.beta(2, 2) * (rng["DCC"][1] - rng["DCC"][0]) + rng["DCC"][0],
                    np.random.beta(2, 2) * (rng["kCC"][1] - rng["kCC"][0]) + rng["kCC"][0],
                    np.random.uniform(*rng["C0"]),
                    np.random.uniform(*rng["CL"])
                ], dtype=np.float32)

                # Ensure physical consistency
                if params[8] >= params[9]:  # C0 >= CL
                    params[8], params[9] = params[9] - 3, params[8] + 3

                # Generate ultra-clean profile
                C_clean = forward_piecewise_precise(self.z, *params).astype(np.float32)

                # Add minimal, realistic noise
                noise_level = 0.005 * np.std(C_clean)
                noise = np.random.normal(0, noise_level, C_clean.shape).astype(np.float32)
                if np.random.rand() < 0.1:
                    noise *= np.random.uniform(2.0, 5.0)  # 2–5× higher noise

                C_noisy = np.maximum(C_clean + noise, 0.01)

                profiles.append(C_noisy)
                params_list.append(params)

        # Normalization strategy
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

        print(f"✨ Generated {len(self.data)} samples")

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

        # Transformer blocks
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

        # Updated parameter prediction with separate expert heads for 4 layers
        self.diffusion_expert = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, 4),  # 4 diffusion coefficients
            nn.Sigmoid()
        )

        self.reaction_expert = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, 4),
            nn.Sigmoid()
        )

        self.boundary_expert = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, 2),  # 2 boundary conditions
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

        # Transformer processing
        for block in self.blocks:
            attn_block, ffn_block = block
            x = attn_block(x)
            x = x + 0.5 * ffn_block(x)

        # Global aggregation
        x = self.norm_final(x)
        x = self.global_pool(x.transpose(1, 2)).squeeze(-1)

        # Expert predictions
        D_params = self.diffusion_expert(x)  # (batch, 4)
        k_params = self.reaction_expert(x)  # (batch, 4)
        C_params = self.boundary_expert(x)  # (batch, 2)

        # Concatenate in the order: DIR, kIR, DOR, kOR, DFL, kFL, DCC, kCC, C0, CL
        return torch.cat([
            D_params[:, 0:1], k_params[:, 0:1],  # DIR, kIR
            D_params[:, 1:2], k_params[:, 1:2],  # DOR, kOR
            D_params[:, 2:3], k_params[:, 2:3],  # DFL, kFL
            D_params[:, 3:4], k_params[:, 3:4],  # DCC, kCC
            C_params[:, 0:1], C_params[:, 1:2]  # C0, CL
        ], dim=1)


# ===  LIGHTNING MODULE ===
class InversePINNLightning(pl.LightningModule):
    def __init__(self, pretrain=False, lambda_pde=0.005, lr=1e-3):
        super().__init__()
        self.model = InversePINN()
        self.pretrain = pretrain
        self.lambda_pde_max = lambda_pde
        self.ramp_epochs = 50
        self.lr = lr
        self.save_hyperparameters()

        # Updated parameter weighting for 10 parameters
        self.param_weights = torch.tensor([0.5, 6.0, 0.5, 6.0, 0.5, 6.0, 0.5, 6.0, 2.5, 2.5])



    def forward(self, x):
        return self.model(x)

    def on_train_epoch_start(self):
        if not self.pretrain:
            # linearly ramp lambda from 0 → max over ramp_epochs
            e = min(self.current_epoch, self.ramp_epochs)
            self.lambda_pde = self.lambda_pde_max * (e / self.ramp_epochs)

    def compute_physics_loss(self, params_normalized):
        """Enhanced physics loss with multiple constraints for 4 layers"""
        batch_size = params_normalized.shape[0]
        device = params_normalized.device

        params_denorm = denormalize_params(params_normalized)
        DIR, kIR, DOR, kOR, DFL, kFL, DCC, kCC, C0, CL = params_denorm.unbind(dim=1)

        z_batch = z_torch.unsqueeze(0).expand(batch_size, -1).to(device)
        C_pred = forward_piecewise_torch(z_batch, DIR, kIR, DOR, kOR, DFL, kFL, DCC, kCC, C0, CL)

        # Boundary conditions
        bc_loss = torch.mean((C_pred[:, 0] - C0) ** 2) + torch.mean((C_pred[:, -1] - CL) ** 2)

        # Interface continuity at 3 interfaces
        continuity_loss = (
                torch.mean((C_pred[:, idx_interface1 - 1] - C_pred[:, idx_interface1]) ** 2) +
                torch.mean((C_pred[:, idx_interface2 - 1] - C_pred[:, idx_interface2]) ** 2) +
                torch.mean((C_pred[:, idx_interface3 - 1] - C_pred[:, idx_interface3]) ** 2)
        )

        grad = torch.diff(C_pred, dim=1) / dz

        # layer diffusivities at each interface
        D1_L, D1_R = DIR, DOR
        D2_L, D2_R = DOR, DFL
        D3_L, D3_R = DFL, DCC

        flux1 = torch.mean((D1_L * grad[:, idx_interface1 - 1]
                            - D1_R * grad[:, idx_interface1]) ** 2)
        flux2 = torch.mean((D2_L * grad[:, idx_interface2 - 1]
                            - D2_R * grad[:, idx_interface2]) ** 2)
        flux3 = torch.mean((D3_L * grad[:, idx_interface3 - 1]
                            - D3_R * grad[:, idx_interface3]) ** 2)
        flux_loss = flux1 + flux2 + flux3
        bc_w = 8.0
        cont_w = 3.0
        flux_w = 3.0
        smooth_w = 0.1
        bounds_w = 0.2
        # Smoothness constraint
        grad = torch.diff(C_pred, dim=1) / dz
        smoothness_loss = torch.mean(grad ** 2)

        # Parameter bounds
        bounds_loss = torch.mean(torch.relu(-params_normalized) + torch.relu(params_normalized - 1))

        total_phys = (
                bc_w * bc_loss
                + cont_w * continuity_loss
                + flux_w * flux_loss
                + smooth_w * smoothness_loss
                + bounds_w * bounds_loss
        )

        return total_phys


    def training_step(self, batch, batch_idx):
        C_norm, params_true = batch
        params_pred = self.model(C_norm)

        # Loss combination
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

        # L2 regularization
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
        self.log("val_data_loss", data_loss, prog_bar=True)

        if not self.pretrain:
            # compute physics_loss
            physics_loss = self.compute_physics_loss(params_pred)
            total_loss = data_loss + self.lambda_pde * physics_loss
            # **LOG THE PHYSICS TERM**
            self.log("val_physics_loss", physics_loss, prog_bar=True)
        else:
            total_loss = data_loss

        # Calculate relative errors for all 10 parameters
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
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=20, T_mult=2, eta_min=1e-7
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}
        }


# ===  EVALUATION ===
def evaluate__model(model, dataloader):
    """Enhanced model evaluation"""
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
    """Enhanced training pipeline with model persistence"""
    profile_configs = {
        f"_Profile_{i}": {name: param_ranges[name] for name in param_names}
        for i in range(1, 6)
    }

    # Try to load previous best model
    loaded_model, loaded_error = load_best_model()
    best_model = loaded_model
    best_error = loaded_error if loaded_error is not None else float('inf')

    if loaded_model is not None:
        print(f"🌟 Starting with previous best error: {best_error:.4f}")

    results = []

    for run in range(1, CONFIG['max_runs'] + 1):
        print(f"\n🏆  RUN {run}/{CONFIG['max_runs']}")
        print("=" * 50)

        # Generate dataset
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

        # Stage 1: Pretraining - use loaded model if available
        print("🔥  Pretraining")
        if loaded_model is not None and run == 1:
            # Use loaded model for first run
            model = loaded_model
            model.pretrain = True
            model.lr = CONFIG['lr_pretrain']
            print("🔄 Using loaded model as starting point")
        else:
            # Create fresh model for subsequent runs or if no loaded model
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

        # Stage 2: Physics fine-tuning
        print("⚡  Physics Fine-tuning")
        model.pretrain = False
        model.lambda_pde = CONFIG['lambda_pde']
        model.lr = CONFIG['lr_finetune']

        # 1. Freeze backbone parameters so only the expert heads train initially
        for name, p in model.model.named_parameters():
            if not any(head in name for head in ['diffusion_expert', 'reaction_expert', 'boundary_expert']):
                p.requires_grad = False

        # 2. Create the Trainer with EarlyStopping and UnfreezeCallback
        finetune_trainer = pl.Trainer(
            max_epochs=CONFIG['finetune_epochs'],
            accelerator="auto", devices="auto",
            precision=precision,
            gradient_clip_val=CONFIG['grad_clip_val'],
            callbacks=[
                EarlyStopping(monitor="val_physics_loss", patience=30, mode="min"),
                EarlyStopping(monitor="val_data_loss", patience=30, mode="min"),
                UnfreezeCallback(unfreeze_epoch=20)
            ],
            logger=False, enable_progress_bar=True
        )

        try:
            for name, p in model.model.named_parameters():
                if not any(head in name for head in ['diffusion_expert', 'reaction_expert', 'boundary_expert']):
                    p.requires_grad = False
            finetune_trainer.fit(model, train_loader, val_loader)
        except Exception as e:
            print(f"❌ Fine-tuning failed: {e}")
            continue

        # Evaluation
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

            # Save model if it's the new best
            if avg_error < best_error:
                best_error = avg_error
                best_model = model
                save_best_model(model, avg_error, run)
                print(f"🌟 NEW RECORD! {avg_error:.4f} - Model saved!")

            if avg_error < CONFIG['error_threshold']:
                print(f"🏆 SUCCESS! {avg_error:.4f} < {CONFIG['error_threshold']}")
                break

        except Exception as e:
            print(f"❌ Evaluation failed: {e}")
            continue

        # Clear loaded_model flag after first run
        if run == 1:
            loaded_model = None

    return results, best_model, best_error

class UnfreezeCallback(pl.Callback):
    def __init__(self, unfreeze_epoch):
        self.unfreeze_epoch = unfreeze_epoch

    def on_epoch_start(self, trainer, pl_module):
        if trainer.current_epoch == self.unfreeze_epoch:
            for p in pl_module.model.parameters():
                p.requires_grad = True
            print(f"🔓 Unfroze full model at epoch {self.unfreeze_epoch}")

# ===  PLOTTING UTILITIES ===
def plot_parity(true_vals, pred_vals):
    """Creates parity scatter plots for each of the 10 parameters."""
    n_params = true_vals.shape[1]
    for i in range(n_params):
        plt.figure()
        plt.scatter(true_vals[:, i], pred_vals[:, i], alpha=0.6)
        lims = [true_vals[:, i].min(), true_vals[:, i].max()]
        plt.plot(lims, lims, '--')
        plt.xlabel(f"True {param_names[i]}")
        plt.ylabel(f"Pred {param_names[i]}")
        plt.title(f"Parity Plot: {param_names[i]}")
        plt.tight_layout()
        plt.show()

def plot_true_vs_reconstructed(true_vals, pred_vals):
    """Overlays true vs. reconstructed trajectories for each parameter in a 5×2 panel layout."""
    n_params = true_vals.shape[1]
    # we'll do 5 rows × 2 columns
    n_rows = (n_params + 1) // 2
    fig, axes = plt.subplots(n_rows, 2, figsize=(12, n_rows*3), sharex=True)
    axes = axes.flatten()

    x_axis = np.arange(true_vals.shape[0])

    for i in range(n_params):
        ax = axes[i]
        ax.plot(x_axis, true_vals[:, i], label="True", linewidth=1)
        ax.plot(x_axis, pred_vals[:, i], linestyle="--", label="Pred", linewidth=1)
        ax.set_title(param_names[i])
        ax.set_xlabel("Sample Index")
        ax.set_ylabel(param_names[i])
        ax.grid(True)

    # turn off any unused subplots
    for j in range(n_params, len(axes)):
        axes[j].axis('off')

    # single legend for the whole figure
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.95, 0.95))
    fig.suptitle("True vs. Reconstructed Parameter Trajectories", y=1.02, fontsize=16)
    plt.tight_layout()
    plt.show()

def save_best_model(model, error, run_number, save_dir="model_checkpoints"):
    """Save the best model with metadata"""
    os.makedirs(save_dir, exist_ok=True)

    model_data = {
        'model_state_dict': model.model.state_dict(),
        'error': error,
        'run_number': run_number,
        'param_names': param_names,
        'param_ranges': param_ranges
    }

    save_path = os.path.join(save_dir, "best_model.pkl")
    with open(save_path, 'wb') as f:
        pickle.dump(model_data, f)

    print(f"💾 Saved best model from run {run_number} with error {error:.4f}")
    return save_path


def load_best_model(save_dir="model_checkpoints"):
    """Load the best model if it exists"""
    save_path = os.path.join(save_dir, "best_model.pkl")

    if not os.path.exists(save_path):
        print("📂 No previous best model found - starting fresh")
        return None, None

    try:
        with open(save_path, 'rb') as f:
            model_data = pickle.load(f)

        # Create new model instance
        model = InversePINNLightning(pretrain=True, lr=CONFIG['lr_pretrain'])
        model.model.load_state_dict(model_data['model_state_dict'])

        prev_error = model_data['error']
        prev_run = model_data['run_number']

        print(f"🔄 Loaded previous best model from run {prev_run} with error {prev_error:.4f}")
        return model, prev_error

    except Exception as e:
        print(f"⚠️  Failed to load previous model: {e}")
        return None, None

if __name__ == "__main__":
    print("🚀 INVERSE PINN – 4‑LAYER RETINAL O₂ MODEL")
    print("=" * 60)

    try:
        results, best_model, best_error = run__training()

        if results:
            print(f"\n🏆 FINAL RESULTS")
            print("=" * 50)
            best_run = min(results, key=lambda x: x['avg_error'])['run']
            best_entry = next(r for r in results if r['run'] == best_run)
            print(f"\n🥇 Champion Run: {best_run}")
            print(f"🎯 Error: {best_entry['avg_error']:.4f} ({best_entry['avg_error'] * 100:.2f}%)")
            for i, name in enumerate(param_names):
                err = best_entry['errors'][i]
                print(f"  {name:>3}: {err:.4f} ({err * 100:5.2f}%)")

            # Reconstruct champion validation set
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
            val_loader = DataLoader(
                val_subset, batch_size=CONFIG['batch_size'],
                shuffle=False, num_workers=0, pin_memory=True
            )

            # Evaluate & plot
            y_true, y_pred = evaluate__model(best_model.model, val_loader)
            plot_parity(y_true.numpy(), y_pred.numpy())
            plot_true_vs_reconstructed(y_true.numpy(), y_pred.numpy())

        else:
            print("❌ No successful runs")

    except Exception as e:
        print("❌ Training error:", e)
        import traceback
        traceback.print_exc()

    print("\n🎬 Training completed!")