import os, time, numpy as np
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import log_loss, roc_auc_score

DATA=r'E:\DaT\cnn3d'; OUT=r'E:\DaT\submission_trials'
os.makedirs(OUT, exist_ok=True)
LOG=os.path.join(OUT,'ms_trial.log')
X = np.load(os.path.join(DATA,'X_reg.npy'))
y = np.load(os.path.join(DATA,'y.npy')).ravel().astype(np.int64)
groups = np.load(r'E:\DaT\v26_oof\groups.npy', allow_pickle=True).ravel()
device='cuda'
EPOCHS=35; BATCH=8; SEED=42

def log(m):
    with open(LOG,'a') as f: f.write(f'[{time.strftime("%H:%M:%S")}] {m}\n')

class ResBlock3(nn.Module):
    def __init__(s,c):
        super().__init__()
        s.b1=nn.Sequential(nn.Conv3d(c,c,3,padding=1),nn.BatchNorm3d(c),nn.ReLU())
        s.b2=nn.Sequential(nn.Conv3d(c,c,3,padding=1),nn.BatchNorm3d(c))
        s.relu=nn.ReLU()
    def forward(s,x): return s.relu(x+s.b2(s.b1(x)))

class NetMS(nn.Module):
    """multi-scale: coarse ctx (17x20x21) + fine detail (34x40x42)"""
    def __init__(s):
        super().__init__()
        s.ctx=nn.Sequential(
            nn.Conv3d(1,12,3,padding=1),nn.BatchNorm3d(12),nn.ReLU(),nn.MaxPool3d(2))
        s.fine=nn.Sequential(
            nn.Conv3d(1,16,3,padding=1),nn.BatchNorm3d(16),nn.ReLU(),nn.MaxPool3d(2),
            nn.Conv3d(16,32,3,padding=1),nn.BatchNorm3d(32),nn.ReLU(),nn.MaxPool3d(2))
        s.merge=nn.Sequential(
            nn.Conv3d(12+32,48,3,padding=1),nn.BatchNorm3d(48),nn.ReLU(),
            ResBlock3(48),ResBlock3(48),
            nn.Conv3d(48,64,3,padding=1),nn.BatchNorm3d(64),nn.ReLU(),
            nn.AdaptiveAvgPool3d((2,2,2)))
        s.head=nn.Sequential(nn.Flatten(),nn.Linear(64*8,224),nn.ReLU(),nn.Dropout(0.4),nn.Linear(224,1))
    def forward(s,x):
        cf = s.fine(x)
        cx = nn.functional.avg_pool3d(x,kernel_size=2)
        c  = s.ctx(cx)
        m  = torch.cat([c,cf],dim=1)
        return s.head(s.merge(m)).squeeze(-1)

torch.manual_seed(SEED); np.random.seed(SEED)
Xc = nn.functional.avg_pool3d(torch.from_numpy(X).float().unsqueeze(1),2).squeeze(1).numpy()
print('fine',X.shape,'ctx',Xc.shape)
SPLIT=StratifiedGroupKFold(5,shuffle=True,random_state=42)
FOLDS=list(SPLIT.split(np.zeros(len(y)),y,groups))
oof=np.zeros(len(y)); counts=np.zeros(len(y))
log(f'MS start {device}')
for fold,(tr,va) in enumerate(FOLDS):
    Xt=torch.from_numpy(X[tr]).float().unsqueeze(1)
    dl=DataLoader(TensorDataset(Xt,torch.from_numpy(y[tr]).float()),batch_size=BATCH,shuffle=True)
    model=NetMS().to(device)
    opt=optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
    sched=optim.lr_scheduler.CosineAnnealingLR(opt,T_max=EPOCHS)
    crit=nn.BCEWithLogitsLoss()
    scaler=torch.amp.GradScaler('cuda')
    for ep in range(EPOCHS):
        model.train()
        for xb,yb in dl:
            xb=xb.to(device); yb=yb.to(device)
            opt.zero_grad()
            with torch.autocast('cuda'):
                loss=crit(model(xb),yb)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        sched.step()
    model.eval()
    pred=np.zeros(len(va))
    Xva=torch.from_numpy(X[va]).float().unsqueeze(1).to(device)
    with torch.no_grad():
        for s in range(0,len(va),48):
            e=min(s+48,len(va))
            p=torch.sigmoid(model(Xva[s:e]).squeeze(-1)).float().cpu().numpy()
            for dx,dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                t=torch.roll(Xva[s:e],dx,dims=2); t=torch.roll(t,dy,dims=1)
                p+=torch.sigmoid(model(t).squeeze(-1)).float().cpu().numpy()
            for dz in (-1,1):
                t=torch.roll(Xva[s:e],dz,dims=3)
                p+=torch.sigmoid(model(t).squeeze(-1)).float().cpu().numpy()
            pred[s:e]=p
    pred/=7
    oof[va]=pred; counts[va]=1
    ll=log_loss(y[va],np.clip(pred,1e-6,1-1e-6))
    log(f'f{fold}: LL={ll:.4f} AUC={roc_auc_score(y[va],pred):.4f}')
    print(f'f{fold}: LL={ll:.4f} AUC={roc_auc_score(y[va],pred):.4f}',flush=True)
ll=log_loss(y,np.clip(oof,1e-6,1-1e-6))
log(f'MS OOF r42: LL={ll:.4f} AUC={roc_auc_score(y,oof):.4f}  [baseline r42 1ch: LL=0.4469 AUC=0.9060]')
print('MS OOF r42:',f'LL={ll:.4f} AUC={roc_auc_score(y,oof):.4f}')
