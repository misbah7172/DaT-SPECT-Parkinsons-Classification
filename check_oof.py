"""Quick sanity check: evaluate fold 0 models."""
import os, sys, numpy as np, pandas as pd, torch, torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import log_loss, roc_auc_score

sys.stdout.reconfigure(line_buffering=True)
CACHE_DIR = "D:/DaT_cache/volumes_2mm"
MODEL_DIR = "D:/DaT_cache/models"
LABELS_PATH = "E:/DaT/Dataset/train_labels.csv"

class _ResBlock3(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.b1 = nn.Sequential(nn.Conv3d(c,c,3,padding=1),nn.BatchNorm3d(c),nn.ReLU())
        self.b2 = nn.Sequential(nn.Conv3d(c,c,3,padding=1),nn.BatchNorm3d(c))
    def forward(self, x): return torch.relu(x + self.b2(self.b1(x)))
class _DeepNet(nn.Module):
    def __init__(self, net, head):
        super().__init__()
        self.net = net; self.head = head
    def forward(self, x): return self.head(self.net(x)).squeeze(-1)
def build_net(arch):
    if arch == "r":
        net = nn.Sequential(nn.Conv3d(1,16,3,padding=1),nn.BatchNorm3d(16),nn.ReLU(),nn.MaxPool3d(2),nn.Conv3d(16,32,3,padding=1),nn.BatchNorm3d(32),nn.ReLU(),nn.MaxPool3d(2),_ResBlock3(32),_ResBlock3(32),nn.Conv3d(32,64,3,padding=1),nn.BatchNorm3d(64),nn.ReLU(),nn.AdaptiveAvgPool3d((2,2,2)))
        head = nn.Sequential(nn.Flatten(),nn.Linear(64*8,192),nn.ReLU(),nn.Dropout(0.4),nn.Linear(192,1))
    else:
        net = nn.Sequential(nn.Conv3d(1,20,3,padding=1),nn.BatchNorm3d(20),nn.ReLU(),nn.MaxPool3d(2),nn.Conv3d(20,36,3,padding=1),nn.BatchNorm3d(36),nn.ReLU(),nn.MaxPool3d(2),_ResBlock3(36),_ResBlock3(36),nn.Conv3d(36,48,3,padding=1),nn.BatchNorm3d(48),nn.ReLU(),nn.AdaptiveAvgPool3d((2,2,2)))
        head = nn.Sequential(nn.Flatten(),nn.Linear(48*8,224),nn.ReLU(),nn.Dropout(0.4),nn.Linear(224,1))
    return _DeepNet(net, head)

class VDS(Dataset):
    def __init__(self, d, u):
        self.d=d; self.u=u
    def __len__(self): return len(self.u)
    def __getitem__(self, i):
        v=np.load(os.path.join(self.d,"{}.npy".format(self.u[i])))
        v=(v-v.mean())/(v.std()+1e-8)
        return torch.from_numpy(v[np.newaxis].astype(np.float32))

labels_df = pd.read_csv(LABELS_PATH)
avail = [f.replace(".npy","") for f in os.listdir(CACHE_DIR) if f.endswith(".npy")]
labels_df = labels_df[labels_df["uid"].isin(avail)].reset_index(drop=True)
uids = labels_df["uid"].tolist()
labs = labels_df["is_pathologic"].values.astype(float)
groups = labels_df["uid"].values

skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
splits = list(skf.split(labs, labs, groups))
device = torch.device("cuda")

names = ["deep_r_42","deep_r_777","deep_r_2024","deep_big_2025","deep_big_1984","deep_big_100"]

for fold in range(5):
    _, val_idx = splits[fold]
    v_uids = [uids[i] for i in val_idx]
    v_labs = labs[val_idx]
    ds = VDS(CACHE_DIR, v_uids)
    ld = DataLoader(ds, batch_size=12, shuffle=False, num_workers=0)
    print("Fold {} ({} scans):".format(fold, len(val_idx)), flush=True)
    all_preds = []
    for name in names:
        ckpt_path = os.path.join(MODEL_DIR, "{}_fold{}.pt".format(name, fold))
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        model = build_net(ck["arch"]).to(device)
        model.load_state_dict(ck["state"])
        model.eval()
        ps = []
        with torch.no_grad():
            for b in ld:
                p = torch.sigmoid(model(b.to(device))).cpu().numpy()
                ps.append(p)
        preds = np.concatenate(ps)
        all_preds.append(preds)
        ll = log_loss(v_labs, np.clip(preds,1e-7,1-1e-7))
        auc = roc_auc_score(v_labs, preds)
        print("  {}: LL={:.4f} AUC={:.4f}".format(name, ll, auc), flush=True)
        del model; torch.cuda.empty_cache()
    avg = np.mean(all_preds, axis=0)
    ll = log_loss(v_labs, np.clip(avg,1e-7,1-1e-7))
    auc = roc_auc_score(v_labs, avg)
    print("  ENSEMBLE: LL={:.4f} AUC={:.4f}".format(ll, auc), flush=True)
