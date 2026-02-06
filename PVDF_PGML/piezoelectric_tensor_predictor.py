#!/usr/bin/env python3
"""
predict_d33_margin_enhanced_updated.py
Prediction script consistent with the enhanced training file:
- Vectorized Fourier features using freqs registered as buffer
- Loads model state and optional EMA shadow saved in checkpoint ('ema_shadow')
- Applies the same base + margin clamp when PVDF_Beta_Fraction > method ref
- Robust loading of preprocessor and model tensors
- Improved piezoelectric tensor estimation using empirical relations and component-specific contributions
"""
import os
import math
import json
import importlib.util
from copy import deepcopy
from typing import Union, Dict, Any
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.compose import ColumnTransformer  # Import this to add to safe globals

# Add this line to allow ColumnTransformer in torch.load
torch.serialization.add_safe_globals([ColumnTransformer])

# ---------------- Config (match training defaults where relevant) ----------------
CFG = {
    "beta_margin_scale": 200.0,
    "min_increase_when_beta_gt": 0.5,
    "enforce_base_when_beta_gt_method_ref": True,
    "method_beta_ref_map": {
        "Electrospinning": 0.5496,
        "Solution casting": 0.5,
        "Poling": 0.5,
        "Sol-gel": 0.5,
        "default": 0.5
    }
}

# ---------------- Load properties ----------------
def load_properties():
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Try multiple locations
    possible_paths = [
        os.path.join(script_dir, 'materials_properties.py'),
        os.path.join(script_dir, 'properties.json'),
        'materials_properties.py',
        'properties.json'
    ]
    
    for path in possible_paths:
        try:
            if os.path.exists(path):
                if path.endswith('.py'):
                    spec = importlib.util.spec_from_file_location("materials_properties", path)
                    mod = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(mod)
                    if hasattr(mod, 'properties'):
                        return mod.properties
                elif path.endswith('.json'):
                    with open(path, 'r') as f:
                        props = json.load(f)
                    if isinstance(props, dict):
                        return props
        except Exception as e:
            continue
    
    raise FileNotFoundError(f"Could not find materials_properties.py or properties.json. Tried: {possible_paths}")

# Load properties (will raise error if not found)
properties = load_properties()

# Get PVDF properties
pvdf_beta_ref = float(properties['PVDF'].get('Beta'))
pvdf_baseline_default = float(properties['PVDF'].get('Piezoelectric Coefficient (d33)'))
pvdf_baseline_by_method = {
    "Electrospinning": 18.0,
    "Solution casting": 6.0,
    "Poling": 6.0,
    "Sol-gel": 6.0
}
METHOD_BETA_REF_MAP = CFG["method_beta_ref_map"]

# Factors for piezoelectric components based on literature ratios for PVDF
NU_PVDF = properties['PVDF'].get('Poissons Ratio')
SQRT_2_1_NU = math.sqrt(2 * (1 + NU_PVDF))
FACTOR_D31 = 20.0 / 32.0
FACTOR_D32 = 1.50 / 32.0
SHEAR_FACTOR_D15 = (27.0 / 32.0) / SQRT_2_1_NU
SHEAR_FACTOR_D24 = (23.0 / 32.0) / SQRT_2_1_NU

# ---------------- Helpers (same physics formulas as training) ----------------
def estimate_sample_beta(dopant, vol_frac_pct):
    beta0 = pvdf_beta_ref
    try:
        if dopant in properties and 'Beta_at_1p5pct' in properties[dopant]:
            beta_1p5 = float(properties[dopant]['Beta_at_1p5pct'])
            if vol_frac_pct <= 0:
                return beta0
            scale = vol_frac_pct / 1.5
            est = beta0 + scale * (beta_1p5 - beta0)
            lo = min(beta0, beta_1p5) - 0.05
            hi = max(beta0, beta_1p5) + 0.05
            return float(np.clip(est, lo, hi))
    except Exception:
        pass
    return beta0

def yamada_dielectric(epsilon_pvdf, epsilon_dopant, vol_frac):
    num = 1 + 3 * vol_frac * (epsilon_dopant - epsilon_pvdf)
    den = 3 * epsilon_pvdf + (epsilon_dopant - epsilon_pvdf) * (1 - vol_frac) + 1e-9
    return epsilon_pvdf * (num / den)

def maxwell_eucken(k_pvdf, k_dopant, vol_frac):
    den = (2 * k_pvdf + k_dopant - vol_frac * (k_dopant - k_pvdf) + 1e-9)
    return k_pvdf * (2 * k_pvdf + k_dopant + 2 * vol_frac * (k_dopant - k_pvdf)) / den

def halpin_tsai(E_pvdf, E_dopant, vol_frac, aspect_ratio=1):
    ratio = (E_dopant / (E_pvdf + 1e-12))
    eta = (ratio - 1) / (ratio + aspect_ratio + 1e-9)
    den = (1 - eta * vol_frac + 1e-9)
    return E_pvdf * (1 + aspect_ratio * eta * vol_frac) / den

def modified_rom(nu_pvdf, nu_dopant, vol_frac):
    return 0.9 * (nu_pvdf * (1 - vol_frac) + nu_dopant * vol_frac)

def calculate_composite_properties(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    method_map = CFG.get('method_beta_ref_map', {})
    for idx, row in df.iterrows():
        dopant = row['Dopants']
        vol_frac = float(row['Dopants fr'])/100.0
        eps_pvdf = properties['PVDF'].get('Dielectric Constant')
        eps_dopant = properties.get(dopant, {}).get('Dielectric Constant', eps_pvdf)
        df.at[idx, 'Effective Dielectric Constant'] = yamada_dielectric(eps_pvdf, eps_dopant, vol_frac)
        k_pvdf = properties['PVDF'].get('Thermal Conductivity')
        k_dopant = properties.get(dopant, {}).get('Thermal Conductivity', k_pvdf)
        df.at[idx, 'Effective Thermal Conductivity'] = maxwell_eucken(k_pvdf, k_dopant, vol_frac)
        E_pvdf = properties['PVDF'].get("Youngs Modulus")
        E_dopant = properties.get(dopant, {}).get('Youngs Modulus', E_pvdf)
        aspect = 10 if dopant in ['CNT','GO','G','Nfs','MXenes'] else 1
        df.at[idx, "Effective Youngs Modulus"] = halpin_tsai(E_pvdf, E_dopant, vol_frac, aspect)
        nu_pvdf = properties['PVDF'].get("Poissons Ratio")
        nu_dopant = properties.get(dopant, {}).get('Poissons Ratio', nu_pvdf)
        df.at[idx, "Effective Poissons Ratio"] = modified_rom(nu_pvdf, nu_dopant, vol_frac)
        rho_pvdf = properties['PVDF'].get("Density")
        rho_dopant = properties.get(dopant, {}).get('Density', rho_pvdf)
        df.at[idx, "Effective Density"] = rho_pvdf * (1 - vol_frac) + rho_dopant * vol_frac
        sp_heat_pvdf = properties['PVDF'].get("Specific Heat Capacity")
        sp_heat_dopant = properties.get(dopant, {}).get('Specific Heat Capacity', sp_heat_pvdf)
        df.at[idx, "Effective Specific Heat Capacity"] = sp_heat_pvdf * (1 - vol_frac) + sp_heat_dopant * vol_frac
        df.at[idx, "PVDF_Beta_Fraction"] = estimate_sample_beta(dopant, float(row['Dopants fr']))
        method = row.get('Fabrication Method', None)
        method_key = method if (pd.notna(method) and method is not None) else 'default'
        method_ref = method_map.get(method_key, method_map.get('default', pvdf_beta_ref))
        df.at[idx, 'PVDF_Beta_Method_Ref'] = float(method_ref)
        df.at[idx, 'PVDF_Beta_HigherThanMethodRef'] = float(df.at[idx, "PVDF_Beta_Fraction"] > method_ref)
    return df

def add_advanced_features(df: pd.DataFrame, fraction_weight=1.0) -> pd.DataFrame:
    df = df.copy()
    df['Dopant_Piezoelectric Coefficient (d33)'] = df['Dopants'].apply(lambda x: properties.get(x, {}).get('Piezoelectric Coefficient (d33)', 0.0))
    df['Dopant_Dielectric Constant'] = df['Dopants'].apply(lambda x: properties.get(x, {}).get('Dielectric Constant', properties['PVDF'].get('Dielectric Constant')))
    df['Dopant_Youngs Modulus'] = df['Dopants'].apply(lambda x: properties.get(x, {}).get('Youngs Modulus', properties['PVDF'].get('Youngs Modulus')))
    df['Thermal_Electronic'] = df['Effective Thermal Conductivity'] * df['Effective Dielectric Constant']
    df['Mech_Compat'] = df["Effective Youngs Modulus"] / properties['PVDF'].get("Youngs Modulus")
    df['Dopants fr'] = df['Dopants fr'].astype(float) * float(fraction_weight)
    df['Stiffness_Impact'] = df['Dopants fr'] * df["Dopant_Youngs Modulus"]
    df['PZT_Interaction'] = df.apply(lambda x: x['Dopant_Piezoelectric Coefficient (d33)'] * x['Dopants fr'] if x['Dopants'] == 'PZT' else 0.0, axis=1)
    df['GO_Interaction'] = df.apply(lambda x: x['Dopant_Dielectric Constant'] * x['Dopants fr'] if x['Dopants'] == 'GO' else 0.0, axis=1)
    df['BaTiO3_Interaction'] = df.apply(lambda x: x['Dopant_Piezoelectric Coefficient (d33)'] * x['Dopants fr'] if x['Dopants'] == 'BaTiO3' else 0.0, axis=1)
    def pvdf_base_for_row(r):
        meth = r.get('Fabrication Method', None)
        if pd.isna(meth) or meth is None:
            return pvdf_baseline_default
        return pvdf_baseline_by_method.get(meth, pvdf_baseline_default)
    df['PVDF_d33_baseline'] = df.apply(pvdf_base_for_row, axis=1).astype(float)
    df['Filler_Category'] = df['Dopants'].apply(lambda x: properties.get(x, {}).get('Filler_Category', 'Unknown'))
    if 'PVDF_Beta_Method_Ref' not in df.columns:
        df = calculate_composite_properties(df)
    return df

# ---------------- Model classes consistent with training ----------------
class EnhancedAttention(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert dim % num_heads == 0
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.scale = self.head_dim ** -0.5

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x = self.activation(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return self.norm2(x + residual)

class PhysicsConstrainedNN(nn.Module):
    def __init__(self, input_dim, numeric_cols, feature_indices, properties, num_dopants, emb_dim=64, dropout=0.2, use_learned_d33=True):
        super().__init__()
        self.feature_indices = dict(feature_indices)
        self.numeric_cols = list(numeric_cols)
        self.numeric_idx = {name:i for i,name in enumerate(self.numeric_cols)}
        self.properties = properties
        self.base_d33_pvdf = abs(properties['PVDF'].get('Piezoelectric Coefficient (d33)'))
        self.dopant_embedding = nn.Embedding(num_dopants, emb_dim)
        nn.init.normal_(self.dopant_embedding.weight, mean=0, std=0.02)
        self.use_learned_d33 = use_learned_d33
        if use_learned_d33:
            self.learned_d33 = nn.Embedding(num_dopants, 1)
            nn.init.xavier_uniform_(self.learned_d33.weight)
        else:
            self.learned_d33 = None
        # freqs as buffer and vectorized Fourier dimension
        freqs = torch.tensor([1,2,4,8,16,32,64], dtype=torch.float32)
        self.register_buffer('freqs', freqs)
        fourier_dim = len(freqs) * 2
        total_input_dim = input_dim + emb_dim + fourier_dim
        self.input_proj = nn.Linear(total_input_dim, 256)
        self.res_blocks = nn.ModuleList([ResidualBlock(256, dropout) for _ in range(3)])
        self.piezo_in_dim = 5
        self.piezo_branch = nn.Sequential(
            nn.Linear(self.piezo_in_dim, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, 128), nn.LayerNorm(128), nn.GELU()
        )
        self.attention = EnhancedAttention(128, num_heads=4)
        self.fusion = nn.Sequential(
            nn.Linear(256 + 128, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.LayerNorm(128), nn.GELU()
        )
        self.output = nn.Linear(128, 1)
        self.register_buffer('y_mean', torch.tensor(0.0))
        self.register_buffer('y_std', torch.tensor(1.0))

    def set_target_scaler(self, mean, std, device='cpu'):
        self.y_mean = torch.tensor(float(mean), device=device)
        self.y_std = torch.tensor(float(std), device=device)

    def forward(self, x_std, X_orig_numeric, dop_idx):
        x = x_std.clone()
        fr_idx = int(self.numeric_idx['Dopants fr'])
        fr = x[:, fr_idx:fr_idx+1] # (B,1)
        device = x.device
        freqs = self.freqs.to(device) # (F,)
        angles = 2.0 * math.pi * fr * freqs.unsqueeze(0) # (B,F)
        sins = torch.sin(angles)
        coss = torch.cos(angles)
        fourier = torch.cat([sins, coss], dim=1) # (B, 2F)
        emb = self.dopant_embedding(dop_idx)
        piezo_idxs = [
            int(self.numeric_idx['Dopant_Piezoelectric Coefficient (d33)']),
            int(self.numeric_idx['Stiffness_Impact']),
            int(self.numeric_idx['PZT_Interaction']),
            int(self.numeric_idx['GO_Interaction']),
            int(self.numeric_idx['BaTiO3_Interaction'])
        ]
        piezo_feats = x[:, piezo_idxs]
        x_in = torch.cat([x, emb, fourier], dim=1)
        x = self.input_proj(x_in)
        for block in self.res_blocks:
            x = block(x)
        h_piezo = self.piezo_branch(piezo_feats).unsqueeze(1)
        h_piezo = self.attention(h_piezo).squeeze(1)
        h_comb = torch.cat([x, h_piezo], dim=1)
        h_fused = self.fusion(h_comb)
        delta_scaled = self.output(h_fused).squeeze(1)
        if self.use_learned_d33 and (self.learned_d33 is not None):
            learned_raw = self.learned_d33(dop_idx).squeeze(1)
            learned_pos = F.softplus(learned_raw)
            delta_scaled = delta_scaled + learned_pos
        return delta_scaled

    def compute_base_d33_tensor(self, X_orig_numeric_tensor, use_nucleation=False):
        device = X_orig_numeric_tensor.device
        idx_vol = int(self.numeric_idx['Dopants fr'])
        vol_frac = X_orig_numeric_tensor[:, idx_vol] / (100.0 * 1.0 + 1e-12)
        phi_m = 1.0 - vol_frac; phi_f = vol_frac
        d_dopant_input = X_orig_numeric_tensor[:, int(self.numeric_idx['Dopant_Piezoelectric Coefficient (d33)'])]
        if 'PVDF_d33_baseline' in self.numeric_idx:
            pvdf_baseline = X_orig_numeric_tensor[:, int(self.numeric_idx['PVDF_d33_baseline'])]
        else:
            pvdf_baseline = torch.full_like(phi_m, fill_value=float(self.base_d33_pvdf), device=device)
        if 'PVDF_Beta_Fraction' in self.numeric_idx:
            beta_vals = X_orig_numeric_tensor[:, int(self.numeric_idx['PVDF_Beta_Fraction'])]
            pvdf_baseline_scaled = pvdf_baseline * (beta_vals / (pvdf_beta_ref + 1e-12))
        else:
            pvdf_baseline_scaled = pvdf_baseline
        eps_f = X_orig_numeric_tensor[:, int(self.numeric_idx['Dopant_Dielectric Constant'])]
        eps_m = float(self.properties['PVDF'].get('Dielectric Constant'))
        L = 3.0 * eps_m / (eps_f + 2.0 * eps_m + 1e-9)
        L = torch.clamp(L, min=0.0, max=3.0)
        gamma = 0.5
        interfacial = gamma * (phi_f * (eps_f - eps_m) / (eps_f + 2.0 * eps_m + 1e-9)) * pvdf_baseline_scaled
        nucleation_term = torch.zeros_like(phi_f, device=device)
        filler_term = L * phi_f * d_dopant_input
        base_d33 = torch.abs(phi_m * pvdf_baseline_scaled + filler_term + interfacial + nucleation_term)
        base_scaled = (base_d33 - self.y_mean.to(device)) / (self.y_std.to(device) + 1e-9)
        return base_d33, base_scaled

    def scaled_to_orig(self, out_scaled):
        out_orig = out_scaled * self.y_std + self.y_mean
        return F.softplus(out_orig)

# ---------------- Utilities for checkpoint handling ----------------
def infer_model_shape_from_state(state_dict: Dict[str, Any]):
    cand_keys = [k for k in state_dict.keys() if 'dopant_embedding' in k and 'weight' in k]
    if not cand_keys:
        raise RuntimeError("Couldn't find dopant_embedding.weight in state dict.")
    emb_w = state_dict[cand_keys[0]]
    if isinstance(emb_w, torch.Tensor):
        emb_shape = tuple(emb_w.shape)
    else:
        emb_shape = tuple(np.array(emb_w).shape)
    num_dopants, emb_dim = int(emb_shape[0]), int(emb_shape[1])
    learned_present = any(['learned_d33.weight' in k for k in state_dict.keys()])
    return num_dopants, emb_dim, learned_present

def load_state_to_model(model: nn.Module, state_dict: Dict[str, Any], device: Union[str, torch.device] = 'cpu'):
    converted = {}
    for k, v in state_dict.items():
        if isinstance(v, torch.Tensor):
            converted[k] = v.to(device)
        else:
            try:
                converted[k] = torch.tensor(v, device=device)
            except Exception:
                converted[k] = v
    model.load_state_dict({k: converted[k] for k in converted.keys() if k in model.state_dict().keys()}, strict=False)
    return model

def apply_ema_shadow_if_present(model: nn.Module, ckpt: Dict[str, Any], device: Union[str, torch.device] = 'cpu'):
    if 'ema_shadow' not in ckpt:
        return model
    ema_shadow = ckpt['ema_shadow']
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in ema_shadow:
                v = ema_shadow[name]
                if not isinstance(v, torch.Tensor):
                    v = torch.tensor(v, device=device)
                else:
                    v = v.to(device)
                param.data.copy_(v)
    return model

# ---------------- Main prediction function ----------------
def predict_dataframe(checkpoint_path: str, df_input: Union[pd.DataFrame, Dict[str, Any]], device: str = 'cpu', beta_override: float = None):
    if isinstance(df_input, dict):
        df = pd.DataFrame([df_input])
    else:
        df = df_input.copy().reset_index(drop=True)
    if 'Dopants' not in df.columns or 'Dopants fr' not in df.columns:
        raise ValueError("Input must include 'Dopants' and 'Dopants fr' columns.")
    # load checkpoint with weights_only=False to allow loading of ColumnTransformer
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if 'model' not in ckpt:
        raise KeyError("Checkpoint missing 'model' key (expected a dict under key 'model').")
    state_dict = ckpt['model']
    # infer embedding size, learned flag
    num_dopants_ckpt, emb_dim_ckpt, learned_present = infer_model_shape_from_state(state_dict)
    preprocessor = ckpt.get('preprocessor', None)
    dopant_to_idx = ckpt.get('dopant_to_idx', {})
    numeric_cols = ckpt.get('numeric_cols', None)
    feature_indices = ckpt.get('feature_indices', {})
    input_dim = ckpt.get('input_dim', None)
    base_features = ckpt.get('base_features', None)
    fraction_weight = ckpt.get('fraction_weight', 1.0)
    y_mean = ckpt.get('y_mean', 0.0)
    y_std = ckpt.get('y_std', 1.0)
    if numeric_cols is None or base_features is None or input_dim is None:
        raise RuntimeError("Checkpoint missing required metadata (numeric_cols / base_features / input_dim).")
    # compute derived features
    df = calculate_composite_properties(df)
    df = add_advanced_features(df, fraction_weight=fraction_weight)
    # optionally override beta values (useful for hypothetical predictions)
    if beta_override is not None:
        df['PVDF_Beta_Fraction'] = float(beta_override)
        for idx in df.index:
            method = df.at[idx, 'Fabrication Method'] if 'Fabrication Method' in df.columns else None
            method_key = method if (pd.notna(method) and method is not None) else 'default'
            method_ref = METHOD_BETA_REF_MAP.get(method_key, METHOD_BETA_REF_MAP.get('default', pvdf_beta_ref))
            df.at[idx, 'PVDF_Beta_Method_Ref'] = method_ref
            df.at[idx, 'PVDF_Beta_HigherThanMethodRef'] = (df.at[idx, 'PVDF_Beta_Fraction'] > method_ref)
            base_val = pvdf_baseline_by_method.get(method_key, pvdf_baseline_default)
            df.at[idx, 'PVDF_d33_baseline'] = base_val * (df.at[idx, 'PVDF_Beta_Fraction'] / pvdf_beta_ref)
    # dopant indices
    df['dopant_idx'] = df['Dopants'].map(lambda x: dopant_to_idx.get(x, 0))
    if preprocessor is None:
        raise RuntimeError("Checkpoint preprocessor missing; cannot scale features for model input.")
    # prepare inputs
    X_pre = preprocessor.transform(df[base_features])
    X_orig_numeric = df[numeric_cols].values.astype(float)
    # instantiate model
    model = PhysicsConstrainedNN(
        input_dim=input_dim,
        numeric_cols=numeric_cols,
        feature_indices=feature_indices,
        properties=properties,
        num_dopants=num_dopants_ckpt,
        emb_dim=emb_dim_ckpt,
        dropout=0.0,
        use_learned_d33=learned_present
    )
    # load weights into model (robust)
    model = load_state_to_model(model, state_dict, device='cpu')
    # If EMA shadow exists, replace parameters with EMA weights
    if 'ema_shadow' in ckpt:
        apply_ema_shadow_if_present(model, ckpt, device='cpu')
    # move model to device and set scaler
    model = model.to(device)
    model.set_target_scaler(y_mean, y_std, device=device)
    model.eval()
    X_std_t = torch.tensor(X_pre, dtype=torch.float32, device=device)
    X_orig_t = torch.tensor(X_orig_numeric, dtype=torch.float32, device=device)
    dop_idx_t = torch.tensor(df['dopant_idx'].values.astype(int), dtype=torch.long, device=device)
    with torch.no_grad():
        delta_scaled = model(X_std_t, X_orig_t, dop_idx_t) # scaled residual
        base_d33, base_scaled = model.compute_base_d33_tensor(X_orig_t)
        out_scaled = base_scaled + delta_scaled
        out_pre_activation = out_scaled * model.y_std.to(device) + model.y_mean.to(device)
        # compute per-sample margin the same way training did
        beta_idx = numeric_cols.index('PVDF_Beta_Fraction')
        method_ref_idx = numeric_cols.index('PVDF_Beta_Method_Ref')
        beta_diff = X_orig_t[:, beta_idx] - X_orig_t[:, method_ref_idx]
        scale = float(CFG.get('beta_margin_scale', 200.0))
        min_inc = float(CFG.get('min_increase_when_beta_gt', 0.5))
        margin_pc = torch.clamp(scale * beta_diff, min=0.0)
        min_tensor = torch.full_like(margin_pc, fill_value=min_inc, device=device)
        margin_pc = torch.where(margin_pc > 0.0, torch.maximum(margin_pc, min_tensor), margin_pc)
        # enforce clamp to base+margin for samples where PVDF_Beta_HigherThanMethodRef = 1
        mask = (X_orig_t[:, numeric_cols.index('PVDF_Beta_HigherThanMethodRef')] > 0.5).to(device)
        if mask.any() and CFG.get('enforce_base_when_beta_gt_method_ref', True):
            out_pre_activation = torch.where(mask, torch.max(out_pre_activation, base_d33 + margin_pc), out_pre_activation)
            out_scaled = (out_pre_activation - model.y_mean.to(device)) / (model.y_std.to(device) + 1e-9)
        out_orig = model.scaled_to_orig(out_scaled) # apply softplus
        delta_orig = (delta_scaled * model.y_std.to(device)).cpu().numpy()
        base_d33_np = base_d33.cpu().numpy()
        pred_np = out_orig.cpu().numpy().flatten()
    # assemble results
    df_out = df.copy().reset_index(drop=True)
    df_out['PVDF_Beta_Fraction_used'] = df_out['PVDF_Beta_Fraction']
    df_out['physics_base_d33'] = base_d33_np
    df_out['learned_delta_d33'] = delta_orig.flatten()
    df_out['predicted_d33'] = pred_np.flatten()
    # Compute tensor components using more scientific coupling factor approach
    _eps0 = 8.8541878128e-12 # vacuum permittivity (F/m)
    def estimate_full_piezo_from_d33_and_effective_props(
        d33_pC, # predicted d33 in pC/N
        E_eff, # Effective Young's modulus (Pa)
        nu_eff, # Effective Poisson's ratio (unitless)
        eps_r_eff, # Effective relative permittivity (unitless)
        enforce_signs=True
    ):
        """
        Estimate d31, d32, d15, d24 (and return d33) in pC/N using a coupling-factor approach.
        """
        # safe minimal values
        E_eff = max(float(E_eff), 1e6) # Pa (avoid div-by-zero)
        nu_eff = float(nu_eff) if not math.isnan(nu_eff) else NU_PVDF
        eps_r_eff = max(float(eps_r_eff), 1e-6)
        # convert d33 -> C/N (use abs for calculation, apply sign later)
        d33_abs_C = abs(float(d33_pC)) * 1e-12
        # axial compliance s_axial ~ 1/E (approx for mode 33)
        s_axial = 1.0 / E_eff
        # shear compliance s_shear ~ 1/G where G = E/(2(1+nu))
        G = E_eff / (2.0 * (1.0 + nu_eff) + 1e-18)
        s_shear = 1.0 / max(G, 1e-18)
        # permittivity in F/m
        eps33 = _eps0 * eps_r_eff
        # coupling factor k_33 (dimensionless). add tiny floors to numeric stability
        denom = max(s_axial * eps33, 1e-36)
        k33 = d33_abs_C / math.sqrt(denom)
        # estimate absolute transverse components: d31,d32 (C/N)
        d31_abs_C = FACTOR_D31 * k33 * math.sqrt(s_axial * eps33)
        d32_abs_C = FACTOR_D32 * k33 * math.sqrt(s_axial * eps33)
        # estimate absolute shear components d15, d24 (C/N) using shear compliance
        d15_abs_C = SHEAR_FACTOR_D15 * k33 * math.sqrt(s_shear * eps33)
        d24_abs_C = SHEAR_FACTOR_D24 * k33 * math.sqrt(s_shear * eps33)
        # convert back to pC/N
        conv = 1e12
        d33_abs_p = float(d33_abs_C * conv)
        d31_abs_p = float(d31_abs_C * conv)
        d32_abs_p = float(d32_abs_C * conv)
        d15_abs_p = float(d15_abs_C * conv)
        d24_abs_p = float(d24_abs_C * conv)
        # Apply sign conventions typical for PVDF-like materials if requested
        if enforce_signs:
            d33_p = -d33_abs_p
            d31_p = +d31_abs_p
            d32_p = +d32_abs_p
            d15_p = -d15_abs_p
            d24_p = -d24_abs_p
        else:
            d33_p = d33_abs_p
            d31_p = d31_abs_p
            d32_p = d32_abs_p
            d15_p = d15_abs_p
            d24_p = d24_abs_p
        return {
            'd33': d33_p,
            'd31': d31_p,
            'd32': d32_p,
            'd15': d15_p,
            'd24': d24_p
        }

    def format_piezoelectric_tensor(tensor_components):
        matrix = [
            [0.0, 0.0, 0.0, 0.0, tensor_components.get('d15', 0.0), 0.0],
            [0.0, 0.0, 0.0, tensor_components.get('d24', 0.0), 0.0, 0.0],
            [tensor_components.get('d31', 0.0), tensor_components.get('d32', 0.0), tensor_components.get('d33', 0.0), 0.0, 0.0, 0.0]
        ]
        return matrix

    # Compute tensor components for each sample using coupling factor method
    for idx in df_out.index:
        E_eff = df_out.at[idx, 'Effective Youngs Modulus']
        nu_eff = df_out.at[idx, 'Effective Poissons Ratio']
        eps_r_eff = df_out.at[idx, 'Effective Dielectric Constant']
        comps = estimate_full_piezo_from_d33_and_effective_props(
            d33_pC=df_out.at[idx, 'predicted_d33'],
            E_eff=E_eff,
            nu_eff=nu_eff,
            eps_r_eff=eps_r_eff,
            enforce_signs=True
        )
        # Add to df_out
        for comp, val in comps.items():
            df_out.at[idx, f'phys_{comp}'] = val

    return df_out

# convenience wrapper for single-sample predict
def predict_sample(checkpoint_path: str = 'best_phys_resid_monotonic_improved_v2.pt',
                   dopant: str = 'SnO2',
                   frac: float = 1.5,
                   method: str = 'Electrospinning',
                   beta_fraction: float = None,
                   device: str = 'cpu'):
    df = pd.DataFrame({'Dopants':[dopant], 'Dopants fr':[frac], 'Fabrication Method':[method]})
    return predict_dataframe(checkpoint_path, df, device=device, beta_override=beta_fraction)

# ---------------- CLI example ----------------
if __name__ == "__main__":
    checkpoint = 'best_phys_resid_monotonic_improved_v2.pt'
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    # example usage
    res = predict_sample(checkpoint_path=checkpoint,
                         dopant='SnO2',
                         frac=1.5,
                         method='Electrospinning',
                         beta_fraction=0.5725,
                         device='cpu')
