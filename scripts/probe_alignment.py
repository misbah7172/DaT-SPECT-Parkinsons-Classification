import warnings

warnings.filterwarnings("ignore")
import glob
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, ".")
from src.sbr_extractor_atlas import load_aligned

templ = np.load("Dataset/atlas_template.npy")
files = glob.glob("Dataset/DaT_Parkinsons_Challenge_-_niftis.zip/*.nii.gz")[:120]
lab = pd.read_csv("Dataset/train_labels.csv").set_index("uid")

rows = {}
for p in files:
    sm = load_aligned(p)
    uid = os.path.basename(p).replace(".nii.gz", "")
    thr = np.percentile(sm[sm > 0], 95)
    m = (sm >= thr)
    # overlap of this scan's striatal mask with template
    rows[uid] = {"overlap": float((m & (templ >= 0.4)).sum() / max(m.sum(), 1)),
                 "dice": float((2 * (m & (templ >= 0.4)).sum()) / max(((m).sum() + (templ >= 0.4).sum()), 1))}

df = pd.DataFrame(rows).T
Y = lab.loc[df.index, "is_pathologic"]
print("overlap mean", round(float(df.overlap.mean()), 3), "dice mean", round(float(df.dice.mean()), 3))