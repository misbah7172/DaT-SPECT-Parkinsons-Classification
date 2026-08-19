"""
V11: Based on paper recommendations - add asymmetry features, probability shrinkage, MLP.
Key changes from v10:
1. Add comprehensive asymmetry features (left-right ratios)
2. Add posterior-anterior putamen ratios
3. Add probability shrinkage toward prior probability
4. Add small MLP as additional model
5. Better feature engineering focused on biological markers
"""
import json
import os
import shutil
import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from scipy.special import expit, logit
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
import xgboost as xgb
import lightgbm as lgb

SEEDS = [42, 777, 2024, 12345, 999]
N_FOLDS = 5
PVE_REF = 15.625


def load_data():
    labels = pd.read_csv("Dataset/train_labels.csv")
    site = pd.read_csv("Dataset/site_labels.csv")
    geom = pd.read_csv("Dataset/voxel_geometry.csv")
    sbr = pd.read_csv("Dataset/sbr_features_train.csv")
    phys = pd.read_csv("Dataset/phys_features_train.csv")
    morph = pd.read_csv("Dataset/sbr_features_morph_train.csv")
    atlas = pd.read_csv("Dataset/atlas_features_train.csv")

    merged = labels.merge(site, on="uid").merge(geom, on="uid")

    raw1 = [c for c in sbr.columns if c not in ("uid", "label")]
    merged = merged.merge(sbr[["uid"] + raw1], on="uid")

    phys_keep = [c for c in phys.columns if c != "uid"]
    merged = merged.merge(phys[["uid"] + phys_keep].rename(columns={c: f"ph_{c}" for c in phys_keep}), on="uid")

    morph_keep = [c for c in morph.columns if c != "uid"]
    merged = merged.merge(morph[["uid"] + morph_keep].rename(columns={c: f"mo_{c}" for c in morph_keep}), on="uid")

    atlas_keep = [c for c in atlas.columns if c != "uid"]
    merged = merged.merge(atlas[["uid"] + atlas_keep].rename(columns={c: f"at_{c}" for c in atlas_keep}), on="uid")

    merged = merged.dropna()
    y = merged["is_pathologic"].values
    groups = merged["pseudo_site"].astype(str).values

    fac = (PVE_REF / merged["voxel_vol"].values) ** (1.0 / 3.0)
    for c in [c for c in raw1 if c.startswith("sbr_") and c in merged.columns]:
        merged[c] = merged[c] * fac

    return merged, y, groups


def add_asymmetry_features(df):
    eps = 1e-6
    
    # Left-right asymmetry features from SBR
    if "sbr_left_putamen" in df.columns and "sbr_right_putamen" in df.columns:
        L = df["sbr_left_putamen"].values
        R = df["sbr_right_putamen"].values
        df["sbr_putamen_asym_ratio"] = (L - R) / (L + R + eps)
        df["sbr_putamen_min_max"] = np.minimum(L, R) / (np.maximum(L, R) + eps)
        df["sbr_putamen_mean"] = (L + R) / 2
        
    if "sbr_left_caudate" in df.columns and "sbr_right_caudate" in df.columns:
        L = df["sbr_left_caudate"].values
        R = df["sbr_right_caudate"].values
        df["sbr_caudate_asym_ratio"] = (L - R) / (L + R + eps)
        df["sbr_caudate_min_max"] = np.minimum(L, R) / (np.maximum(L, R) + eps)
        df["sbr_caudate_mean"] = (L + R) / 2
        
    if "sbr_left_total" in df.columns and "sbr_right_total" in df.columns:
        L = df["sbr_left_total"].values
        R = df["sbr_right_total"].values
        df["sbr_total_asym_ratio"] = (L - R) / (L + R + eps)
        df["sbr_total_mean"] = (L + R) / 2
    
    # Max features asymmetry
    if "sbr_left_putamen_max" in df.columns and "sbr_right_putamen_max" in df.columns:
        L = df["sbr_left_putamen_max"].values
        R = df["sbr_right_putamen_max"].values
        df["sbr_putamen_max_asym"] = (L - R) / (L + R + eps)
        
    if "sbr_left_caudate_max" in df.columns and "sbr_right_caudate_max" in df.columns:
        L = df["sbr_left_caudate_max"].values
        R = df["sbr_right_caudate_max"].values
        df["sbr_caudate_max_asym"] = (L - R) / (L + R + eps)
    
    # Median features asymmetry
    if "sbr_left_putamen_med" in df.columns and "sbr_right_putamen_med" in df.columns:
        L = df["sbr_left_putamen_med"].values
        R = df["sbr_right_putamen_med"].values
        df["sbr_putamen_med_asym"] = (L - R) / (L + R + eps)
        
    if "sbr_left_caudate_med" in df.columns and "sbr_right_caudate_med" in df.columns:
        L = df["sbr_left_caudate_med"].values
        R = df["sbr_right_caudate_med"].values
        df["sbr_caudate_med_asym"] = (L - R) / (L + R + eps)
    
    # Posterior-anterior ratios (key for Parkinson's)
    if "sbr_left_putamen_post" in df.columns and "sbr_left_putamen_ant" in df.columns:
        df["left_pa_ratio"] = df["sbr_left_putamen_post"].values / (df["sbr_left_putamen_ant"].values + eps)
    if "sbr_right_putamen_post" in df.columns and "sbr_right_putamen_ant" in df.columns:
        df["right_pa_ratio"] = df["sbr_right_putamen_post"].values / (df["sbr_right_putamen_ant"].values + eps)
    
    # Posterior asymmetry
    if "sbr_left_putamen_post" in df.columns and "sbr_right_putamen_post" in df.columns:
        L = df["sbr_left_putamen_post"].values
        R = df["sbr_right_putamen_post"].values
        df["post_putamen_asym"] = (L - R) / (L + R + eps)
    
    # Putamen-to-caudate ratios
    if "sbr_left_putamen" in df.columns and "sbr_left_caudate" in df.columns:
        df["left_pc_ratio_sbr"] = df["sbr_left_putamen"].values / (df["sbr_left_caudate"].values + eps)
    if "sbr_right_putamen" in df.columns and "sbr_right_caudate" in df.columns:
        df["right_pc_ratio_sbr"] = df["sbr_right_putamen"].values / (df["sbr_right_caudate"].values + eps)
    
    # Volume asymmetry
    if "vol_left_putamen" in df.columns and "vol_right_putamen" in df.columns:
        L = df["vol_left_putamen"].values
        R = df["vol_right_putamen"].values
        df["vol_putamen_asym_ratio"] = (L - R) / (L + R + eps)
        
    if "vol_left_caudate" in df.columns and "vol_right_caudate" in df.columns:
        L = df["vol_left_caudate"].values
        R = df["vol_right_caudate"].values
        df["vol_caudate_asym_ratio"] = (L - R) / (L + R + eps)
    
    # CV asymmetry
    if "cv_left_putamen" in df.columns and "cv_right_putamen" in df.columns:
        df["cv_putamen_diff"] = df["cv_left_putamen"].values - df["cv_right_putamen"].values
    if "cv_left_caudate" in df.columns and "cv_right_caudate" in df.columns:
        df["cv_caudate_diff"] = df["cv_left_caudate"].values - df["cv_right_caudate"].values
    
    # Atlas features asymmetry
    if "at_sbr_lp" in df.columns and "at_sbr_rp" in df.columns:
        L = df["at_sbr_lp"].values
        R = df["at_sbr_rp"].values
        df["at_putamen_asym_ratio"] = (L - R) / (L + R + eps)
        
    if "at_sbr_lc" in df.columns and "at_sbr_rc" in df.columns:
        L = df["at_sbr_lc"].values
        R = df["at_sbr_rc"].values
        df["at_caudate_asym_ratio"] = (L - R) / (L + R + eps)
    
    # Shape features asymmetry (from morph)
    for p in ["p95", "p97", "p99"]:
        x_col = f"mo_shape_centroid_x_{p}"
        if x_col in df.columns:
            df[f"mo_centroid_x_{p}_abs"] = np.abs(df[x_col].values)
    
    if "mo_shape_lr_centroid_distance_p95" in df.columns:
        df["mo_lr_dist_p95_abs"] = np.abs(df["mo_shape_lr_centroid_distance_p95"].values)
    
    return df


def build_X(merged):
    exclude = {"uid", "is_pathologic", "pseudo_site", "site_id", "sx", "sy", "sz", "voxel_side", "label"}
    num_cols = [c for c in merged.columns if c not in exclude and merged[c].dtype in (np.float64, np.int64, np.float32)]
    a = merged[num_cols].values.astype(np.float64)
    X = np.nan_to_num(
        np.concatenate([a, np.log1p(np.clip(a, 0, None))], axis=1),
        nan=0.0, posinf=0.0, neginf=0.0,
    )
    return X, num_cols


def optim_blend(M, y):
    def loss(w):
        p = M @ w
        return log_loss(y, np.clip(p, 1e-6, 1 - 1e-6))
    r = minimize(loss, np.ones(M.shape[1]) / M.shape[1],
                 method="L-BFGS-B", bounds=[(0, 1)] * M.shape[1])
    return r.x / r.x.sum()


def optimize_temperature(oof, y):
    oof = np.clip(oof, 1e-6, 1 - 1e-6)
    def loss(log_t):
        p = expit(logit(oof) / np.exp(log_t[0]))
        return log_loss(y, np.clip(p, 1e-6, 1 - 1e-6))
    r = minimize(loss, [0.0], method="L-BFGS-B", bounds=[(-2, 2)])
    return float(np.exp(r.x[0]))


def optimize_shrinkage(oof, y, prior):
    """Optimize probability shrinkage toward prior."""
    oof = np.clip(oof, 1e-6, 1 - 1e-6)
    def loss(alpha):
        p = (1 - alpha) * oof + alpha * prior
        return log_loss(y, np.clip(p, 1e-6, 1 - 1e-6))
    r = minimize_scalar(loss, bounds=(0.0, 0.5), method="bounded")
    return float(r.x)


class MLPWrapper:
    """Simple MLP wrapper for sklearn-like interface."""
    def __init__(self, input_dim, hidden_dims=(64, 32), dropout=0.3, lr=0.001, epochs=50, batch_size=128, seed=42):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.seed = seed
        self.model = None
        self.scaler = StandardScaler()
        
    def _build_model(self):
        import torch
        import torch.nn as nn
        
        class MLP(nn.Module):
            def __init__(self, input_dim, hidden_dims, dropout):
                super().__init__()
                layers = []
                prev_dim = input_dim
                for h in hidden_dims:
                    layers.extend([
                        nn.Linear(prev_dim, h),
                        nn.BatchNorm1d(h),
                        nn.ReLU(),
                        nn.Dropout(dropout),
                    ])
                    prev_dim = h
                layers.append(nn.Linear(prev_dim, 1))
                self.net = nn.Sequential(*layers)
                
            def forward(self, x):
                return self.net(x).squeeze(-1)
        
        torch.manual_seed(self.seed)
        return MLP(self.input_dim, self.hidden_dims, self.dropout)
    
    def fit(self, X, y):
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
        
        self.scaler.fit(X)
        Xs = self.scaler.transform(X)
        
        self.model = self._build_model()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-5)
        criterion = nn.BCEWithLogitsLoss()
        
        X_t = torch.FloatTensor(Xs)
        y_t = torch.FloatTensor(y)
        dataset = TensorDataset(X_t, y_t)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        self.model.train()
        for epoch in range(self.epochs):
            for xb, yb in loader:
                optimizer.zero_grad()
                loss = criterion(self.model(xb), yb)
                loss.backward()
                optimizer.step()
        
        return self
    
    def predict_proba(self, X):
        import torch
        self.model.eval()
        Xs = self.scaler.transform(X)
        with torch.no_grad():
            logits = self.model(torch.FloatTensor(Xs))
            probs = torch.sigmoid(logits).numpy()
        return np.column_stack([1 - probs, probs])


def get_models(seed, n_features):
    return {
        "lr": ("full", LogisticRegression(C=1.31, max_iter=3000, random_state=seed)),
        "xgb1": ("mi", xgb.XGBClassifier(
            n_estimators=800, max_depth=5, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.6, reg_lambda=3.0,
            min_child_weight=5, gamma=0.1,
            use_label_encoder=False, eval_metric="logloss",
            random_state=seed, n_jobs=-1, verbosity=0)),
        "xgb2": ("mi", xgb.XGBClassifier(
            n_estimators=500, max_depth=3, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.8, reg_lambda=1.0,
            min_child_weight=3, gamma=0.0,
            use_label_encoder=False, eval_metric="logloss",
            random_state=seed + 1, n_jobs=-1, verbosity=0)),
        "lgb1": ("mi", lgb.LGBMClassifier(
            n_estimators=800, max_depth=5, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.6, reg_lambda=3.0,
            min_child_samples=15, random_state=seed, n_jobs=-1, verbose=-1)),
        "lgb2": ("mi", lgb.LGBMClassifier(
            n_estimators=500, max_depth=3, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.8, reg_lambda=1.0,
            min_child_samples=5, random_state=seed + 1, n_jobs=-1, verbose=-1)),
        "et": ("mi", ExtraTreesClassifier(1000, max_depth=10, n_jobs=-1, random_state=seed)),
        "hgb": ("mi", HistGradientBoostingClassifier(
            max_iter=800, max_depth=5, learning_rate=0.03,
            l2_regularization=2.0, min_samples_leaf=15, random_state=seed)),
    }


def train_and_evaluate(X, y, groups, seeds):
    n_samples = len(y)
    n_features = X.shape[1]
    models_dict = get_models(42, n_features)
    model_names = list(models_dict.keys()) + ["mlp"]
    n_models = len(model_names)
    n_streams = n_models
    print(f"  {n_models} models (incl MLP), {n_features} features")

    oof = {i: np.zeros(n_samples) for i in range(n_streams)}
    counts = {i: np.zeros(n_samples) for i in range(n_streams)}

    for seed in seeds:
        for tr, va in StratifiedGroupKFold(N_FOLDS, shuffle=True, random_state=seed).split(X, y, groups):
            sc = StandardScaler().fit(X[tr])
            Xtr, Xva = sc.transform(X[tr]), sc.transform(X[va])

            mi = mutual_info_classif(Xtr, y[tr], random_state=seed)
            n_feat = Xtr.shape[1]
            mi_top = min(60, n_feat)
            sel = np.argsort(mi)[::-1][:mi_top]
            Xtr_t, Xva_t = Xtr[:, sel], Xva[:, sel]

            for si, (mname, (mode, model_factory)) in enumerate(models_dict.items()):
                if mode == "full":
                    model = model_factory.__class__(**model_factory.get_params()).fit(Xtr, y[tr])
                    pred = model.predict_proba(Xva)[:, 1]
                else:
                    model = model_factory.__class__(**model_factory.get_params()).fit(Xtr_t, y[tr])
                    pred = model.predict_proba(Xva_t)[:, 1]
                oof[si][va] += pred
                counts[si][va] += 1

            # Train MLP
            mlp = MLPWrapper(input_dim=mi_top, hidden_dims=(64, 32), dropout=0.3, lr=0.001, epochs=50, batch_size=128, seed=seed)
            mlp.fit(Xtr_t, y[tr])
            pred_mlp = mlp.predict_proba(Xva_t)[:, 1]
            oof[n_streams - 1][va] += pred_mlp
            counts[n_streams - 1][va] += 1

        print(f"  seed {seed} done")

    for i in range(n_streams):
        oof[i] = np.where(counts[i] > 0, oof[i] / counts[i], 0.5)

    return oof, model_names


def main():
    print("Loading data...")
    merged, y, groups = load_data()
    print(f"  n={len(y)}, pos={y.sum():.0f}/{len(y)} ({y.mean()*100:.1f}%)")

    # Add asymmetry features
    print("Adding asymmetry features...")
    merged = add_asymmetry_features(merged)
    print(f"  Total columns: {len(merged.columns)}")

    X, num_cols = build_X(merged)
    print(f"  X: {X.shape}")

    # Add site one-hot encoding
    le = LabelEncoder()
    site_ids = le.fit_transform(merged["pseudo_site"].astype(str).values).reshape(-1, 1)
    X = np.hstack([X, site_ids.astype(np.float64)])
    print(f"  X with site: {X.shape}")

    print("\nTraining OOF...")
    oof, model_names = train_and_evaluate(X, y, groups, SEEDS)

    M = np.column_stack([oof[i] for i in range(len(oof))])
    print(f"\nOOF matrix: {M.shape}")

    print("\nPer-stream metrics:")
    for i, name in enumerate(model_names):
        try:
            auc = roc_auc_score(y, M[:, i])
            ll = log_loss(y, np.clip(M[:, i], 1e-6, 1 - 1e-6))
            print(f"  {name:12s} AUC={auc:.4f}  LL={ll:.4f}")
        except:
            print(f"  {name:12s} degenerate")

    # Optimize blend
    w = optim_blend(M, y)
    blend = M @ w
    print(f"\nBLEND AUC={roc_auc_score(y, blend):.4f}  LL={log_loss(y, np.clip(blend, 1e-6, 1-1e-6)):.4f}")

    # Optimize temperature
    temp = optimize_temperature(blend, y)
    p_temp = expit(logit(np.clip(blend, 1e-6, 1 - 1e-6)) / temp)
    print(f"TEMP={temp:.4f} -> AUC={roc_auc_score(y, p_temp):.4f}  LL={log_loss(y, np.clip(p_temp, 1e-6, 1-1e-6)):.4f}")

    # Optimize shrinkage
    prior = y.mean()
    alpha = optimize_shrinkage(blend, y, prior)
    p_shrunk = (1 - alpha) * blend + alpha * prior
    print(f"SHRINK={alpha:.4f} -> AUC={roc_auc_score(y, p_shrunk):.4f}  LL={log_loss(y, np.clip(p_shrunk, 1e-6, 1-1e-6)):.4f}")

    # Combine temperature + shrinkage
    p_combined = (1 - alpha) * p_temp + alpha * prior
    print(f"COMBINED -> AUC={roc_auc_score(y, p_combined):.4f}  LL={log_loss(y, np.clip(p_combined, 1e-6, 1-1e-6)):.4f}")

    # Select best approach
    best_ll = float('inf')
    best_p = None
    best_method = ""
    
    for name, p in [("blend", blend), ("temp", p_temp), ("shrink", p_shrunk), ("combined", p_combined)]:
        ll = log_loss(y, np.clip(p, 1e-6, 1-1e-6))
        auc = roc_auc_score(y, p)
        if ll < best_ll:
            best_ll = ll
            best_p = p
            best_method = name
            best_auc = auc

    print(f"\nBEST: {best_method} AUC={best_auc:.4f}  LL={best_ll:.4f}")
    final = np.clip(best_p, 0.005, 0.995)

    # Greedy elimination
    print("\n--- Greedy elimination ---")
    best_ll_elim = log_loss(y, final)
    best_auc_elim = roc_auc_score(y, final)
    best_w_elim = w.copy()
    best_temp_elim = temp
    best_alpha_elim = alpha
    best_mask = np.ones(M.shape[1], dtype=bool)
    best_method_elim = best_method

    for _ in range(M.shape[1]):
        improved = False
        for j in range(M.shape[1]):
            if not best_mask[j]:
                continue
            mask = best_mask.copy()
            mask[j] = False
            Mc = M[:, mask]
            if Mc.shape[1] == 0:
                continue
            wc = optim_blend(Mc, y)
            bc = Mc @ wc
            
            # Try each method
            tc = optimize_temperature(bc, y)
            ptc = expit(logit(np.clip(bc, 1e-6, 1 - 1e-6)) / tc)
            
            alphac = optimize_shrinkage(bc, y, prior)
            psc = (1 - alphac) * bc + alphac * prior
            
            pcomb = (1 - alphac) * ptc + alphac * prior
            
            for mname, p in [("blend", bc), ("temp", ptc), ("shrink", psc), ("combined", pcomb)]:
                ll_c = log_loss(y, np.clip(p, 1e-6, 1-1e-6))
                if ll_c < best_ll_elim:
                    best_mask = mask.copy()
                    best_w_full = np.zeros(M.shape[1])
                    best_w_full[mask] = wc
                    best_w_elim = best_w_full
                    best_temp_elim = tc
                    best_alpha_elim = alphac
                    best_ll_elim = ll_c
                    best_auc_elim = roc_auc_score(y, p)
                    best_method_elim = mname
                    improved = True
                    print(f"  Dropped {model_names[j]} ({mname}) -> AUC={best_auc_elim:.4f} LL={best_ll_elim:.4f}")
        if not improved:
            break

    print(f"\nFINAL: AUC={best_auc_elim:.4f}  LL={best_ll_elim:.4f}")
    active = [(model_names[i], float(best_w_elim[i])) for i in range(len(model_names)) if best_mask[i]]
    for n, wt in sorted(active, key=lambda x: -x[1]):
        print(f"  {n:12s} w={wt:.4f}")

    return merged, y, groups, X, best_w_elim, best_temp_elim, best_alpha_elim, best_mask, model_names, best_auc_elim, best_ll_elim, best_method_elim


if __name__ == "__main__":
    merged, y, groups, X, best_w, best_temp, best_alpha, best_mask, model_names, final_auc, final_ll, best_method = main()

    seeds = SEEDS
    out_dir = "submission_v11"
    os.makedirs(os.path.join(out_dir, "weights"), exist_ok=True)

    print("\nTraining final models on full data...")
    n_features = X.shape[1]
    models_dict = get_models(42, n_features)

    for si, seed in enumerate(seeds):
        sc = StandardScaler().fit(X)
        Xs = sc.transform(X)
        mi = mutual_info_classif(Xs, y, random_state=seed)
        mi_top = min(60, n_features)
        sel = np.argsort(mi)[::-1][:mi_top]
        Xt = Xs[:, sel]

        for mname, (mode, model_factory) in models_dict.items():
            if mode == "full":
                model = model_factory.__class__(**model_factory.get_params()).fit(Xs, y)
            else:
                model = model_factory.__class__(**model_factory.get_params()).fit(Xt, y)
            joblib.dump(model, os.path.join(out_dir, f"weights/s{si}_{mname}.pkl"))

        # Train MLP
        mlp = MLPWrapper(input_dim=mi_top, hidden_dims=(64, 32), dropout=0.3, lr=0.001, epochs=50, batch_size=128, seed=seed)
        mlp.fit(Xt, y)
        joblib.dump(mlp, os.path.join(out_dir, f"weights/s{si}_mlp.pkl"))

        joblib.dump(sc, os.path.join(out_dir, f"weights/s{si}_scaler.pkl"))
        joblib.dump(sel, os.path.join(out_dir, f"weights/s{si}_mi.pkl"))
        print(f"  seed {seed} done")

    active_weights = [float(best_w[i]) for i in range(len(best_w)) if best_mask[i]]
    active_indices = [int(i) for i in range(len(best_w)) if best_mask[i]]

    cfg = {
        "n_seeds": len(seeds),
        "model_names": model_names,
        "blend_weights": active_weights,
        "blend_indices": active_indices,
        "temperature": float(best_temp),
        "shrinkage_alpha": float(best_alpha),
        "prior": float(y.mean()),
        "best_method": best_method,
        "clip_min": 0.005,
        "clip_max": 0.995,
        "mi_top": 60,
    }
    with open(os.path.join(out_dir, "model_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    with open(os.path.join(out_dir, "oof_metrics.json"), "w") as f:
        json.dump({
            "oof_auroc": round(float(final_auc), 4),
            "oof_ll": round(float(final_ll), 4),
            "active_models": [model_names[i] for i in range(len(model_names)) if best_mask[i]],
        }, f, indent=2)

    shutil.copy("src/sbr_extractor.py", os.path.join(out_dir, "sbr_extractor.py"))
    shutil.copy("submission_v7/sbr_extractor_phys.py", os.path.join(out_dir, "sbr_extractor_phys.py"))
    print(f"\nSaved to {out_dir}")
