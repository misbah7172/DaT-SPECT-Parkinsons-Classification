"""V25 OOF inference to compute metrics."""
import os, sys, gc, json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import log_loss, roc_auc_score

sys.stdout.reconfigure(line_buffering=True)

CACHE_DIR = "D:/DaT_cache/volumes_2mm"
LABELS_PATH = "E:/DaT/Dataset/train_labels.csv"
MODEL_DIR = "D:/DaT_cache/v25_models"
OUTPUT_DIR = "D:/DaT_cache/v25_models"

class _ResBlock(torch.nn.Module):
    def __init__(self, c):
        super().__init__()
        self.b1 = torch.nn.Sequential(torch.nn.Conv3d(c,c,3,padding=1),torch.nn.BatchNorm3d(c),torch.nn.ReLU(inplace=True))
        self.b2 = torch.nn.Sequential(torch.nn.Conv3d(c,c,3,padding=1),torch.nn.BatchNorm3d(c))
    def forward(self, x):
        return torch.relu(x + self.b2(self.b1(x)))

class _DownBlock(torch.nn.Module):
    def __init__(self, ci, co):
        super().__init__()
        self.conv = torch.nn.Sequential(torch.nn.Conv3d(ci,co,3,stride=2,padding=1),torch.nn.BatchNorm3d(co),torch.nn.ReLU(inplace=True))
        self.res = _ResBlock(co)
    def forward(self, x):
        return self.res(self.conv(x))

class _DeepNet(torch.nn.Module):
    def __init__(self, net, head):
        super().__init__()
        self.net = net; self.head = head
    def forward(self, x):
        return self.head(self.net(x)).squeeze(-1)

def build_net_v25(arch):
    if arch == "r":
        net = torch.nn.Sequential(
            torch.nn.Conv3d(1,16,3,padding=1),torch.nn.BatchNorm3d(16),torch.nn.ReLU(inplace=True),torch.nn.MaxPool3d(2),
            _ResBlock(16),_DownBlock(16,32),_DownBlock(32,64),
            torch.nn.Conv3d(64,128,3,padding=1),torch.nn.BatchNorm3d(128),torch.nn.ReLU(inplace=True),
            torch.nn.AdaptiveAvgPool3d((2,2,2)))
        head = torch.nn.Sequential(torch.nn.Flatten(),torch.nn.Linear(128*8,256),torch.nn.ReLU(inplace=True),torch.nn.Dropout(0.4),torch.nn.Linear(256,1))
    else:
        net = torch.nn.Sequential(
            torch.nn.Conv3d(1,20,3,padding=1),torch.nn.BatchNorm3d(20),torch.nn.ReLU(inplace=True),torch.nn.MaxPool3d(2),
            _ResBlock(20),_DownBlock(20,40),_DownBlock(40,80),
            torch.nn.Conv3d(80,160,3,padding=1),torch.nn.BatchNorm3d(160),torch.nn.ReLU(inplace=True),
            torch.nn.AdaptiveAvgPool3d((2,2,2)))
        head = torch.nn.Sequential(torch.nn.Flatten(),torch.nn.Linear(160*8,320),torch.nn.ReLU(inplace=True),torch.nn.Dropout(0.4),torch.nn.Linear(320,1))
    return _DeepNet(net, head)

class VolumeDataset(Dataset):
    def __init__(self, d, u, l=None):
        self.d=d; self.u=u; self.l=l
    def __len__(self): return len(self.u)
    def __getitem__(self, i):
        v=np.load(os.path.join(self.d,"{}.npy".format(self.u[i])))
        v=(v-v.mean())/(v.std()+1e-8); v=v[np.newaxis]
        v=torch.from_numpy(v.astype(np.float32))
        if self.l is not None:
            return v,torch.tensor(self.l[i],dtype=torch.float32)
        return v

def main():
    dev = torch.device("cuda")
    print("GPU: {}".format(torch.cuda.get_device_name(0)), flush=True)

    labels_df = pd.read_csv(LABELS_PATH)
    avail = [f.replace(".npy","") for f in os.listdir(CACHE_DIR) if f.endswith(".npy")]
    labels_df = labels_df[labels_df["uid"].isin(avail)].reset_index(drop=True)
    uids = labels_df["uid"].tolist()
    labels_arr = labels_df["is_pathologic"].values.astype(float)
    groups = labels_df["uid"].values

    archs = ["r","r","r","r","r","big","big","big","big","big"]
    seeds = [42,777,2024,1984,100,2025,314,271,1337,999]
    names = ["v25_{}_{}".format("r" if a=="r" else "big", s) for a,s in zip(archs,seeds)]

    skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    splits = list(skf.split(labels_arr, labels_arr, groups))

    oof_preds = np.zeros((len(labels_arr), len(names)))

    for fold in range(5):
        tr_idx, va_idx = splits[fold]
        print("\nFOLD {} val={}".format(fold, len(va_idx)), flush=True)
        v_uids = [uids[i] for i in va_idx]
        v_labels = labels_arr[va_idx]
        ds_va = VolumeDataset(CACHE_DIR, v_uids, v_labels.tolist())
        ld_va = DataLoader(ds_va, batch_size=8, shuffle=False, num_workers=0, pin_memory=True)

        for mi,(arch,seed,name) in enumerate(zip(archs,seeds,names)):
            ckpt_path = os.path.join(MODEL_DIR, "{}_fold{}.pt".format(name, fold))
            if not os.path.exists(ckpt_path):
                print("  MISSING: {}".format(ckpt_path), flush=True)
                continue
            ckpt = torch.load(ckpt_path, map_location="cpu")
            model = build_net_v25(arch).to(dev)
            model.load_state_dict(ckpt["state"])
            model.eval()

            vp = []
            with torch.no_grad():
                for v,_ in ld_va:
                    v = v.to(dev)
                    with torch.amp.autocast("cuda"):
                        preds = torch.sigmoid(model(v)).cpu().numpy()
                    vp.extend(preds)
            oof_preds[va_idx, mi] = vp
            print("  {}: done".format(name), flush=True)
            del model
            torch.cuda.empty_cache()
            gc.collect()

    # Save
    np.save(os.path.join(OUTPUT_DIR, "oof_preds.npy"), oof_preds)
    np.save(os.path.join(OUTPUT_DIR, "oof_labels.npy"), labels_arr)

    # Metrics
    mask = (oof_preds.sum(axis=1) != 0)
    ensemble = oof_preds[mask].mean(axis=1)
    true = labels_arr[mask]
    ll = log_loss(true, np.clip(ensemble, 1e-7, 1-1e-7))
    auc = roc_auc_score(true, ensemble)
    print("\n{} OVERALL {}".format("="*60, "="*60), flush=True)
    print("LL={:.4f} AUC={:.4f}".format(ll, auc), flush=True)

    for fold in range(5):
        _, vi = splits[fold]
        fp = oof_preds[vi].mean(axis=1)
        fl = log_loss(labels_arr[vi], np.clip(fp, 1e-7, 1-1e-7))
        fa = roc_auc_score(labels_arr[vi], fp)
        print("  Fold {}: LL={:.4f} AUC={:.4f}".format(fold, fl, fa), flush=True)

    with open(os.path.join(OUTPUT_DIR, "oof_metrics.json"), "w") as f:
        json.dump({"logloss": float(ll), "auc": float(auc)}, f, indent=2)
    print("Saved", flush=True)

if __name__ == "__main__":
    main()