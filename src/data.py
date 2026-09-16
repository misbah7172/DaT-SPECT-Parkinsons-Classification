"""Data loading module. All feature sources, labels, groups, geometry."""
import os
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_labels():
    """Load binary labels and group identifiers."""
    y = np.load(os.path.join(BASE, 'v26_oof', 'y.npy'))
    groups = np.load(os.path.join(BASE, 'v26_oof', 'groups.npy'), allow_pickle=True)
    uids = np.load(os.path.join(BASE, 'cnn3d', 'uids.npy'), allow_pickle=True)
    return y, groups, uids


def load_csv_features(path, drop_cols=('uid', 'label')):
    """Load CSV features, dropping non-numeric columns."""
    df = pd.read_csv(os.path.join(BASE, path))
    cols = [c for c in df.columns if c not in drop_cols]
    return df[cols].values.astype(np.float64), cols, df


def load_all_features():
    """Load all feature sources. Returns dict of name -> (X, columns)."""
    features = {}

    sbr, cols, _ = load_csv_features('Dataset/sbr_features_train.csv')
    features['SBR'] = (sbr, cols)

    morph, cols, _ = load_csv_features('Dataset/sbr_features_morph_train.csv')
    features['SBR_Morph'] = (morph, cols)

    phys, cols, _ = load_csv_features('Dataset/phys_features_train.csv')
    features['PHYS'] = (phys, cols)

    atlas, cols, _ = load_csv_features('Dataset/atlas_features_train.csv')
    features['ATLAS'] = (atlas, cols)

    striatal, cols, _ = load_csv_features('striatal_features_from_crop.csv')
    features['STRIATAL'] = (striatal, cols)

    rad, cols, _ = load_csv_features('radiomic_features.csv')
    features['RAD'] = (rad, cols)

    rad_in, cols, _ = load_csv_features('radiomic_features_inorm.csv')
    features['RAD_INORM'] = (rad_in, cols)

    roi, cols, _ = load_csv_features('roi_radiomics.csv')
    features['ROI_RAD'] = (roi, cols)

    tex, cols, _ = load_csv_features('texture_roi.csv')
    features['TEXTURE'] = (tex, cols)

    emb = np.load(os.path.join(BASE, 'phase6_embeddings.npy'))
    features['EMB512'] = (emb, [f'emb_{i}' for i in range(emb.shape[1])])

    pca = np.load(os.path.join(BASE, 'phase6_pca_emb.npy'))
    features['PCA50'] = (pca, [f'pca_{i}' for i in range(pca.shape[1])])

    return features


def load_cnn_oof():
    """Load CNN OOF predictions. Returns (N, n_models) array + names."""
    cnn_dir = os.path.join(BASE, 'v30_oof')
    root = BASE
    files = [
        'v30_oof/cnn3dr_oof.npy', 'v30_oof/cnn3d_oof.npy',
        'v30_oof/cnn25s_oof.npy', 'v30_oof/cnn25v2_oof.npy',
        'v30_oof/cnn25r2_oof.npy', 'v30_oof/cnn25b_oof.npy',
        'v30_oof/cnn3dr_oof_sc.npy', 'v30_oof/cnn3d_oof_sc.npy',
        'v30_oof/cnn3d_deep6.npy', 'v30_oof/cnn3d_deep12.npy',
        'balanced_oof_all_seeds.npy', 'v40_honest_oof.npy',
        'v42_temp_honest_oof.npy',
    ]
    feats = []
    names = []
    for f in files:
        path = os.path.join(root, f)
        if os.path.exists(path):
            arr = np.load(path)
            if arr.ndim == 2:
                for i in range(arr.shape[1]):
                    feats.append(arr[:, i])
                    names.append(f'{os.path.basename(f)}_s{i}')
            else:
                feats.append(arr)
                names.append(os.path.basename(f))
    return np.column_stack(feats), names


def load_geometry():
    """Load voxel geometry."""
    df = pd.read_csv(os.path.join(BASE, 'Dataset', 'voxel_geometry.csv'))
    return df


def load_site_labels():
    """Load site labels."""
    df = pd.read_csv(os.path.join(BASE, 'Dataset', 'site_labels.csv'))
    return df


def build_feature_group(groups_list, feature_dict):
    """Concatenate multiple feature groups into one matrix."""
    arrays = []
    cols = []
    for g in groups_list:
        X, c = feature_dict[g]
        arrays.append(X)
        cols.extend(c)
    return np.hstack(arrays), cols
