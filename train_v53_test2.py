import os, time
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import models
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import log_loss, roc_auc_score

print('DEBUG: Starting script')
DATA = r'E:\DaT\cnn3d'
X = np.load(os.path.join(DATA, 'X_reg.npy'))
y = np.load(os.path.join(DATA, 'y.npy')).ravel().astype(np.int64)
groups = np.load(r'E:\DaT\v26_oof\groups.npy', allow_pickle=True).ravel()
print('DEBUG: Data loaded')
OUT = r'E:\DaT\submission_v26\weights_v53_test2'
os.makedirs(OUT, exist_ok=True)
LOG = os.path.join(r'E:\DaT\submission_v26', 'train_v53_test2.log')
print('DEBUG: Output dir set')

EPOCHS = 2
BATCH = 8
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('DEBUG: Device:', device)
JOBS_R = [42, 777, 2024, 100]
JOBS_B = [2025, 1984, 11, 22, 33, 44, 55, 66]
SPLIT = StratifiedGroupKFold(5, shuffle=True, random_state=42)
FOLDS = list(SPLIT.split(np.zeros(len(y)), y, groups))
print('DEBUG: Number of folds:', len(FOLDS))

def log(m):
    with open(LOG, 'a') as f:
        f.write(f'[{time.strftime("%H:%M:%S")}] {m}\n')
    print(m, flush=True)

print('DEBUG: About to define classes')
class SliceWrapper(nn.Module):
    def __init__(self, base_model_name='resnet18', pretrained=True, agg_method='mean', feature_dim=512, hidden_dim=256, dropout=0.4):
        super().__init__()
        self.agg_method = agg_method
        print(f'DEBUG: SliceWrapper init, base={base_model_name}')
        if base_model_name == 'resnet18':
            self.base = models.resnet18(pretrained=pretrained)
        elif base_model_name == 'resnet34':
            self.base = models.resnet34(pretrained=pretrained)
        else:
            raise ValueError(f'Unsupported base model: {base_model_name}')
        with torch.no_grad():
            w = self.base.conv1.weight.data
            w_avg = w.mean(dim=1, keepdim=True)
            self.base.conv1.weight = nn.Parameter(w_avg)
            if self.base.conv1.bias is not None:
                self.base.conv1.bias = nn.Parameter(self.base.conv1.bias.data.mean(dim=0, keepdim=True))
        self.base.fc = nn.Identity()
        if base_model_name in ['resnet18', 'resnet34']:
            self.feature_dim = 512
        else:
            raise ValueError(f'Unknown feature dim for {base_model_name}')
        self.head = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, x):
        B, C, D, H, W = x.shape
        features = []
        for d in range(D):
            slice_img = x[:, :, d, :, :]
            feat = self.base(slice_img)
            features.append(feat)
        features = torch.stack(features, dim=0)
        if self.agg_method == 'mean':
            agg_feat = features.mean(dim=0)
        elif self.agg_method == 'max':
            agg_feat = features.max(dim=0)[0]
        else:
            raise ValueError(f'Unsupported agg_method: {self.agg_method}')
        logits = self.head(agg_feat)
        return logits.squeeze(-1)

def make_model(arch='r'):
    print('DEBUG: make_model called')
    return SliceWrapper(base_model_name='resnet18', pretrained=True)

print('DEBUG: Classes defined')
if not os.path.exists(LOG):
    open(LOG, 'w').close()
log(f'v53_test2 start device={device}')
print('DEBUG: Log written')

print('DEBUG: Setting JOBS')
JOBS = [('r', s) for s in JOBS_R] + [('big', s) for s in JOBS_B]
print('DEBUG: JOBS set:', JOBS)
print('DEBUG: Setting total')
total = len(JOBS) * len(FOLDS)
print('DEBUG: total set:', total)
done = 0
oof_maps = {}
print('DEBUG: Before loop')
print('DEBUG: About to enter loop')
for arch, seed in JOBS:
    print(f'DEBUG: Inside loop arch={arch} seed={seed}')
    oof_arr = np.zeros(len(y)); cnt = np.zeros(len(y))
    for fold, (tr, va) in enumerate(FOLDS):
        t0 = time.time()
        pred, ll, auc = train_fold(arch, seed, fold, tr, va)
        oof_arr[va] += pred; cnt[va] += 1
        done += 1
        log(f'{arch}_{seed} f{fold}: LL={ll:.4f} AUC={auc:.4f} ({time.time()-t0:.0f}s, {done}/{total})')
    oof_arr /= np.maximum(cnt, 1)
    np.save(os.path.join(OUT, f'fold_oof_{arch}_{seed}.npy'), oof_arr)
    log(f'{arch}_{seed} OOF: LL={log_loss(y, np.clip(oof_arr,1e-6,1-1e-6)):.4f} AUC={roc_auc_score(y, oof_arr):.4f}')
print('DEBUG: After loop')
files = [os.path.join(OUT, f'fold_oof_{a}_{s}.npy') for a, s in JOBS]
print('DEBUG: Files set')
ens = np.mean([np.load(f) for f in files], axis=0)
print('DEBUG: Ensemble computed')
np.save(os.path.join(OUT, 'fold_oof.npy'), ens)
print('DEBUG: Ensemble saved')
log(f'ENSEMBLE OOF (12 combos): LL={log_loss(y, np.clip(ens,1e-6,1-1e-6)):.4f} AUC={roc_auc_score(y, ens):.4f}')
print('DEBUG: Final log written')
log('v53_test2 done')
print('DEBUG: Script ending')
