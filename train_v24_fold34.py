"""V24 CNN training resume: fold 3+ with flushed output."""
import os, sys, time, json, gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import log_loss, roc_auc_score

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)

CACHE_DIR = "D:/DaT_cache/volumes_2mm"
LABELS_PATH = "E:/DaT/Dataset/train_labels.csv"
SITE_PATH = "E:/DaT/Dataset/site_labels.csv"
OUTPUT_DIR = "D:/DaT_cache/models"
os.makedirs(OUTPUT_DIR, exist_ok=True)

class _ResBlock3(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.b1 = nn.Sequential(nn.Conv3d(c, c, 3, padding=1), nn.BatchNorm3d(c), nn.ReLU())
        self.b2 = nn.Sequential(nn.Conv3d(c, c, 3, padding=1), nn.BatchNorm3d(c))
    def forward(self, x):
        return torch.relu(x + self.b2(self.b1(x)))

class _DeepNet(nn.Module):
    def __init__(self, net, head):
        super().__init__()
        self.net = net; self.head = head
    def forward(self, x):
        return self.head(self.net(x)).squeeze(-1)

def build_net(arch):
    if arch == "r":
        net = nn.Sequential(
            nn.Conv3d(1,16,3,padding=1),nn.BatchNorm3d(16),nn.ReLU(),nn.MaxPool3d(2),
            nn.Conv3d(16,32,3,padding=1),nn.BatchNorm3d(32),nn.ReLU(),nn.MaxPool3d(2),
            _ResBlock3(32),_ResBlock3(32),nn.Conv3d(32,64,3,padding=1),nn.BatchNorm3d(64),nn.ReLU(),
            nn.AdaptiveAvgPool3d((2,2,2)))
        head = nn.Sequential(nn.Flatten(),nn.Linear(64*8,192),nn.ReLU(),nn.Dropout(0.4),nn.Linear(192,1))
    else:
        net = nn.Sequential(
            nn.Conv3d(1,20,3,padding=1),nn.BatchNorm3d(20),nn.ReLU(),nn.MaxPool3d(2),
            nn.Conv3d(20,36,3,padding=1),nn.BatchNorm3d(36),nn.ReLU(),nn.MaxPool3d(2),
            _ResBlock3(36),_ResBlock3(36),nn.Conv3d(36,48,3,padding=1),nn.BatchNorm3d(48),nn.ReLU(),
            nn.AdaptiveAvgPool3d((2,2,2)))
        head = nn.Sequential(nn.Flatten(),nn.Linear(48*8,224),nn.ReLU(),nn.Dropout(0.4),nn.Linear(224,1))
    return _DeepNet(net, head)

class VolumeDataset(Dataset):
    def __init__(self, cache_dir, uids, labels=None, augment=False):
        self.cache_dir = cache_dir; self.uids = uids; self.labels = labels; self.augment = augment
    def __len__(self): return len(self.uids)
    def __getitem__(self, idx):
        vol = np.load(os.path.join(self.cache_dir, f"{self.uids[idx]}.npy"))
        vol = (vol - vol.mean()) / (vol.std() + 1e-8)
        vol = vol[np.newaxis, ...]
        if self.augment:
            if np.random.random() > 0.5: vol = vol[:,:,: ,::-1].copy()
            if np.random.random() > 0.5: vol = vol[:,::-1,:,:].copy()
            vol = vol * np.random.uniform(0.95, 1.05)
        vol = torch.from_numpy(vol.astype(np.float32))
        if self.labels is not None:
            return vol, torch.tensor(self.labels[idx], dtype=torch.float32)
        return vol

def train_model(model, train_loader, val_loader, device, epochs=80, lr=1e-4):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, total_steps=epochs*len(train_loader), pct_start=5/epochs)
    criterion = nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    best_val_loss = float("inf"); best_state = None; patience_counter = 0; warmup_done = False

    for epoch in range(epochs):
        model.train()
        for vol, lbl in train_loader:
            vol, lbl = vol.to(device), lbl.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                loss = criterion(model(vol), lbl)
            if torch.isnan(loss) or torch.isinf(loss): continue
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update(); scheduler.step()
        if epoch >= 5: warmup_done = True
        model.eval()
        vloss = 0; vcount = 0; vp = []; vt = []
        with torch.no_grad():
            for vol, lbl in val_loader:
                vol, lbl = vol.to(device), lbl.to(device)
                with torch.amp.autocast("cuda"):
                    loss = criterion(model(vol), lbl)
                if not (torch.isnan(loss) or torch.isinf(loss)):
                    vloss += loss.item()*len(lbl); vcount += len(lbl)
                vp.extend(torch.sigmoid(model(vol)).cpu().numpy())
                vt.extend(lbl.cpu().numpy())
        avg = vloss/max(vcount,1)
        if warmup_done and avg < best_val_loss:
            best_val_loss = avg; best_state = {k:v.clone() for k,v in model.state_dict().items()}; patience_counter = 0
        elif warmup_done: patience_counter += 1
        if (epoch+1)%10==0 or epoch==0:
            try: auc = roc_auc_score(np.array(vt), np.array(vp))
            except: auc = 0
            print(f"  Ep {epoch}: loss={avg:.4f} best={best_val_loss:.4f} auc={auc:.4f} pat={patience_counter}", flush=True)
        if patience_counter >= 15:
            print(f"  Early stop at ep {epoch}", flush=True); break
    if best_state: model.load_state_dict(best_state)
    return model, best_val_loss

def main():
    device = torch.device("cuda")
    print(f"Device: {torch.cuda.get_device_name(0)}", flush=True)

    labels_df = pd.read_csv(LABELS_PATH)
    available = [f.replace(".npy","") for f in os.listdir(CACHE_DIR) if f.endswith(".npy")]
    labels_df = labels_df[labels_df["uid"].isin(available)].reset_index(drop=True)
    try:
        site_df = pd.read_csv(SITE_PATH)
        labels_df = labels_df.merge(site_df, on="uid", how="left")
        if "sx" in labels_df.columns:
            labels_df["scanner_group"] = labels_df.apply(lambda r: f"{r['sx']:.1f}x{r['sy']:.1f}x{r['sz']:.1f}", axis=1)
        else: labels_df["scanner_group"] = "unknown"
    except: labels_df["scanner_group"] = "unknown"

    uids = labels_df["uid"].tolist()
    labels_arr = labels_df["is_pathologic"].values.astype(float)
    groups = labels_df["scanner_group"].values

    archs = ["r","r","r","big","big","big"]
    names = ["deep_r_42","deep_r_777","deep_r_2024","deep_big_2025","deep_big_1984","deep_big_100"]
    seeds = [42,777,2024,2025,1984,100]

    skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    splits = list(skf.split(labels_arr, labels_arr, groups))

    oof = np.zeros((len(labels_arr), 6))
    oof_mask = np.zeros(len(labels_arr), dtype=bool)

    for fold in range(3, 5):
        train_idx, val_idx = splits[fold]
        print(f"\n{'='*50} FOLD {fold} train={len(train_idx)} val={len(val_idx)} {'='*50}", flush=True)
        t_uids = [uids[i] for i in train_idx]
        t_labels = labels_arr[train_idx].tolist()
        v_uids = [uids[i] for i in val_idx]
        v_labels = labels_arr[val_idx]

        for mi, (arch, name, seed) in enumerate(zip(archs, names, seeds)):
            print(f"\n  {name} ({arch}, seed={seed})", flush=True)
            torch.manual_seed(seed); torch.cuda.manual_seed(seed)
            model = build_net(arch).to(device)
            print(f"    Params: {sum(p.numel() for p in model.parameters()):,}", flush=True)
            ds_tr = VolumeDataset(CACHE_DIR, t_uids, t_labels, augment=True)
            ds_va = VolumeDataset(CACHE_DIR, v_uids, v_labels.tolist(), augment=False)
            ld_tr = DataLoader(ds_tr, batch_size=12, shuffle=True, num_workers=0, pin_memory=True)
            ld_va = DataLoader(ds_va, batch_size=12, shuffle=False, num_workers=0, pin_memory=True)
            t0 = time.time()
            model, vl = train_model(model, ld_tr, ld_va, device, epochs=80)
            print(f"    Done: loss={vl:.4f} time={time.time()-t0:.0f}s", flush=True)
            torch.save({"arch":arch,"state":model.state_dict()}, os.path.join(OUTPUT_DIR, f"{name}_fold{fold}.pt"))
            model.eval()
            bs = 0
            with torch.no_grad():
                for vol, _ in ld_va:
                    vol = vol.to(device)
                    with torch.amp.autocast("cuda"):
                        preds = torch.sigmoid(model(vol)).cpu().numpy()
                    be = bs + len(preds)
                    oof[val_idx[bs:be], mi] = preds
                    bs = be
            del model; torch.cuda.empty_cache(); gc.collect()

        oof_mask[val_idx] = True
        fp = oof[val_idx].mean(axis=1)
        print(f"  Fold {fold}: LL={log_loss(v_labels, np.clip(fp,1e-7,1-1e-7)):.4f} AUC={roc_auc_score(v_labels, fp):.4f}", flush=True)

    valid = oof_mask.sum()
    overall = oof[oof_mask].mean(axis=1)
    true = labels_arr[oof_mask]
    ll = log_loss(true, np.clip(overall,1e-7,1-1e-7))
    auc = roc_auc_score(true, overall)
    print(f"\nOVERALL: LL={ll:.4f} AUC={auc:.4f} ({valid} scans)", flush=True)
    np.save(os.path.join(OUTPUT_DIR,"oof_preds.npy"), oof)
    np.save(os.path.join(OUTPUT_DIR,"oof_labels.npy"), labels_arr)
    np.save(os.path.join(OUTPUT_DIR,"oof_mask.npy"), oof_mask)
    print("SAVED", flush=True)

if __name__ == "__main__":
    main()
