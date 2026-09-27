"""V25 training resume: finish fold 1 + folds 2-4."""
import os, sys, time, json, gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import log_loss, roc_auc_score

sys.stdout.reconfigure(line_buffering=True)

CACHE_DIR = "D:/DaT_cache/volumes_2mm"
LABELS_PATH = "E:/DaT/Dataset/train_labels.csv"
OUTPUT_DIR = "D:/DaT_cache/v25_models"

class _ResBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.b1 = nn.Sequential(nn.Conv3d(c,c,3,padding=1),nn.BatchNorm3d(c),nn.ReLU(inplace=True))
        self.b2 = nn.Sequential(nn.Conv3d(c,c,3,padding=1),nn.BatchNorm3d(c))
    def forward(self, x):
        return torch.relu(x + self.b2(self.b1(x)))

class _DownBlock(nn.Module):
    def __init__(self, ci, co):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv3d(ci,co,3,stride=2,padding=1),nn.BatchNorm3d(co),nn.ReLU(inplace=True))
        self.res = _ResBlock(co)
    def forward(self, x):
        return self.res(self.conv(x))

class _DeepNet(nn.Module):
    def __init__(self, net, head):
        super().__init__()
        self.net = net; self.head = head
    def forward(self, x):
        return self.head(self.net(x)).squeeze(-1)

def build_net_v25(arch):
    if arch == "r":
        net = nn.Sequential(
            nn.Conv3d(1,16,3,padding=1),nn.BatchNorm3d(16),nn.ReLU(inplace=True),nn.MaxPool3d(2),
            _ResBlock(16),_DownBlock(16,32),_DownBlock(32,64),
            nn.Conv3d(64,128,3,padding=1),nn.BatchNorm3d(128),nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((2,2,2)))
        head = nn.Sequential(nn.Flatten(),nn.Linear(128*8,256),nn.ReLU(inplace=True),nn.Dropout(0.4),nn.Linear(256,1))
    else:
        net = nn.Sequential(
            nn.Conv3d(1,20,3,padding=1),nn.BatchNorm3d(20),nn.ReLU(inplace=True),nn.MaxPool3d(2),
            _ResBlock(20),_DownBlock(20,40),_DownBlock(40,80),
            nn.Conv3d(80,160,3,padding=1),nn.BatchNorm3d(160),nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((2,2,2)))
        head = nn.Sequential(nn.Flatten(),nn.Linear(160*8,320),nn.ReLU(inplace=True),nn.Dropout(0.4),nn.Linear(320,1))
    return _DeepNet(net, head)

class VolumeDataset(Dataset):
    def __init__(self, d, u, l=None, aug=False):
        self.d=d; self.u=u; self.l=l; self.aug=aug
    def __len__(self): return len(self.u)
    def __getitem__(self, i):
        v=np.load(os.path.join(self.d,"{}.npy".format(self.u[i])))
        v=(v-v.mean())/(v.std()+1e-8); v=v[np.newaxis]
        if self.aug:
            if np.random.random()>0.5: v=v[:,:,: ,::-1].copy()
            if np.random.random()>0.5: v=v[:,::-1,:,:].copy()
            if np.random.random()>0.5: v=v[:,:,::-1,:].copy()
            v=v*np.random.uniform(0.93,1.07)+np.random.uniform(-0.02,0.02)
        v=torch.from_numpy(v.astype(np.float32))
        if self.l is not None:
            return v,torch.tensor(self.l[i],dtype=torch.float32)
        return v

def mixup_data(x, y, alpha=0.2):
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0)).to(x.device)
    return lam*x+(1-lam)*x[idx], y, y[idx], lam

class LabelSmoothingBCE(nn.Module):
    def __init__(self, s=0.05):
        super().__init__(); self.s=s
    def forward(self, logits, targets):
        t = targets*(1-self.s)+0.5*self.s
        return nn.functional.binary_cross_entropy_with_logits(logits, t)

def train_model(model, tr_loader, va_loader, dev, epochs=100, lr=1e-4, accum=2):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = LabelSmoothingBCE(0.05)
    scaler = torch.amp.GradScaler("cuda")
    best_vl=float("inf"); best_st=None; pat=0
    for ep in range(epochs):
        model.train(); opt.zero_grad(); el=0; nb=0
        for v,l in tr_loader:
            v,l = v.to(dev), l.to(dev)
            vm,la,lb,lam = mixup_data(v,l)
            with torch.amp.autocast("cuda"):
                loss = (lam*crit(model(vm),la)+(1-lam)*crit(model(vm),lb))/accum
            scaler.scale(loss).backward()
            el+=loss.item()*accum; nb+=1
            if nb%accum==0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
                scaler.step(opt); scaler.update(); opt.zero_grad()
        sched.step()
        model.eval(); vl_sum=0; vc=0; vp=[]; vt=[]
        with torch.no_grad():
            for v,l in va_loader:
                v,l = v.to(dev), l.to(dev)
                with torch.amp.autocast("cuda"):
                    logits=model(v); loss=crit(logits,l)
                if not (torch.isnan(loss) or torch.isinf(loss)):
                    vl_sum+=loss.item()*len(l); vc+=len(l)
                vp.extend(torch.sigmoid(logits).cpu().numpy())
                vt.extend(l.cpu().numpy())
        avg_vl = vl_sum/max(vc,1)
        if avg_vl < best_vl:
            best_vl=avg_vl; best_st={k:v.clone() for k,v in model.state_dict().items()}; pat=0
        else: pat+=1
        if (ep+1)%10==0 or ep==0:
            try: auc=roc_auc_score(np.array(vt),np.array(vp))
            except: auc=0
            print("  Ep {}: train={:.4f} val={:.4f} best={:.4f} auc={:.4f} pat={}".format(
                ep, el/max(nb,1), avg_vl, best_vl, auc, pat), flush=True)
        if pat>=20:
            print("  Early stop ep {}".format(ep), flush=True); break
    if best_st: model.load_state_dict(best_st)
    return model, best_vl

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

    # Determine which models still need training
    done = set()
    for f in os.listdir(OUTPUT_DIR):
        if f.endswith(".pt"):
            # e.g. v25_r_42_fold0.pt
            fold = int(f.replace(".pt","").split("fold")[1])
            name = f.replace(".pt","").split("_fold")[0]
            done.add((fold, name))
    print("Already done: {} models".format(len(done)), flush=True)

    for fold in range(5):
        tr_idx, va_idx = splits[fold]
        print("\n{} FOLD {} train={} val={} {}".format("="*50, fold, len(tr_idx), len(va_idx), "="*50), flush=True)
        t_uids = [uids[i] for i in tr_idx]
        t_labels = labels_arr[tr_idx].tolist()
        v_uids = [uids[i] for i in va_idx]
        v_labels = labels_arr[va_idx]

        for mi,(arch,seed,name) in enumerate(zip(archs,seeds,names)):
            if (fold, name) in done:
                print("  {} fold{} already done, skipping".format(name, fold), flush=True)
                continue
            print("\n  {} (arch={}, seed={})".format(name, arch, seed), flush=True)
            torch.manual_seed(seed); torch.cuda.manual_seed(seed)
            model = build_net_v25(arch).to(dev)
            print("    Params: {:,}".format(sum(p.numel() for p in model.parameters())), flush=True)
            ds_tr = VolumeDataset(CACHE_DIR, t_uids, t_labels, aug=True)
            ds_va = VolumeDataset(CACHE_DIR, v_uids, v_labels.tolist(), aug=False)
            ld_tr = DataLoader(ds_tr, batch_size=8, shuffle=True, num_workers=0, pin_memory=True)
            ld_va = DataLoader(ds_va, batch_size=8, shuffle=False, num_workers=0, pin_memory=True)
            t0=time.time()
            model,vl = train_model(model, ld_tr, ld_va, dev, epochs=100)
            print("    loss={:.4f} time={:.0f}s".format(vl, time.time()-t0), flush=True)
            torch.save({"arch":arch,"state":model.state_dict()}, os.path.join(OUTPUT_DIR,"{}_fold{}.pt".format(name,fold)))
            del model; torch.cuda.empty_cache(); gc.collect()

    print("\n{} ALL DONE {}".format("="*60, "="*60), flush=True)

if __name__ == "__main__":
    main()
