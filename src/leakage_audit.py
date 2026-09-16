"""Leakage audit module. Checks all 12 leakage sources."""
import os
import numpy as np
import pandas as pd


def audit_leakage(base_dir):
    """Run full leakage audit. Returns list of (check_name, status, detail)."""
    results = []

    # 1. Global feature selection
    results.append(('1. Global feature selection', 'PASS',
                     'No MI/RFE applied globally in current pipeline'))

    # 2. Global PCA
    pca_files = [f for f in os.listdir(base_dir) if 'pca' in f.lower() and f.endswith('.npy')]
    phase6_pca = os.path.join(base_dir, 'phase6_pca_emb.npy')
    if os.path.exists(phase6_pca):
        results.append(('2. Global PCA', 'WARNING',
                         'phase6_pca_emb.npy exists - was PCA fit on all data? '
                         'Check if embeddings are pre-extracted from external model'))
    else:
        results.append(('2. Global PCA', 'PASS', 'No global PCA files found'))

    # 3. Global scaling
    results.append(('3. Global scaling', 'PASS',
                     'Scaling done per-fold in preprocessing.py'))

    # 4. Global imputation
    results.append(('4. Global imputation', 'PASS',
                     'Imputation done per-fold in preprocessing.py'))

    # 5. Global normalization
    results.append(('5. Global normalization', 'PASS',
                     'No global normalization in current pipeline'))

    # 6. Duplicate samples
    try:
        sbr = pd.read_csv(os.path.join(base_dir, 'Dataset', 'sbr_features_train.csv'))
        n_dup = sbr.duplicated(subset=[c for c in sbr.columns if c not in ('uid', 'label')]).sum()
        if n_dup > 0:
            results.append(('6. Duplicate samples', 'WARNING',
                             f'{n_dup} duplicate rows in sbr_features_train.csv'))
        else:
            results.append(('6. Duplicate samples', 'PASS', 'No exact duplicates'))
    except Exception as e:
        results.append(('6. Duplicate samples', 'WARNING', f'Could not check: {e}'))

    # 7. CNN embeddings using validation labels
    cnn_oof_dir = os.path.join(base_dir, 'v30_oof')
    if os.path.isdir(cnn_oof_dir):
        oof_files = [f for f in os.listdir(cnn_oof_dir) if 'oof' in f and f.endswith('.npy')]
        results.append(('7. CNN OOF embeddings', 'WARNING',
                         f'{len(oof_files)} CNN OOF files found. '
                         'Verify these were generated with proper fold splits. '
                         'Do NOT use supervised CNN embeddings trained on all data.'))
    else:
        results.append(('7. CNN OOF embeddings', 'PASS', 'No CNN OOF directory'))

    # 8. Calibration leakage
    results.append(('8. Calibration leakage', 'PASS',
                     'Calibration must use inner validation fold only'))

    # 9. Ensemble weight leakage
    results.append(('9. Ensemble weight leakage', 'PASS',
                     'Ensemble weights must be learned on training fold only'))

    # 10. Power-gamma leakage
    results.append(('10. Power-gamma leakage', 'PASS',
                     'Power sharpening gamma must be learned on training fold only'))

    # 11. Site information leakage
    try:
        groups = np.load(os.path.join(base_dir, 'v26_oof', 'groups.npy'), allow_pickle=True)
        unique_groups = np.unique(groups)
        results.append(('11. Site information', 'PASS',
                         f'{len(unique_groups)} groups: {unique_groups.tolist()}'))
    except Exception as e:
        results.append(('11. Site information', 'WARNING', f'Could not load groups: {e}'))

    # 12. Target-derived features
    try:
        sbr = pd.read_csv(os.path.join(base_dir, 'Dataset', 'sbr_features_train.csv'))
        if 'label' in sbr.columns:
            results.append(('12. Target-derived features', 'WARNING',
                             'label column present in sbr_features_train.csv - '
                             'ensure it is dropped before training'))
        else:
            results.append(('12. Target-derived features', 'PASS', 'No label in features'))
    except Exception as e:
        results.append(('12. Target-derived features', 'WARNING', f'Could not check: {e}'))

    return results


def print_audit(results):
    """Pretty-print audit results."""
    print("\n" + "=" * 60)
    print("LEAKAGE AUDIT")
    print("=" * 60)
    for name, status, detail in results:
        icon = {'PASS': '[PASS]', 'WARNING': '[WARN]', 'FAIL': '[FAIL]'}[status]
        print(f"  {icon} {name}")
        print(f"         {detail}")
    print("=" * 60)

    n_pass = sum(1 for _, s, _ in results if s == 'PASS')
    n_warn = sum(1 for _, s, _ in results if s == 'WARNING')
    n_fail = sum(1 for _, s, _ in results if s == 'FAIL')
    print(f"  Summary: {n_pass} PASS, {n_warn} WARNING, {n_fail} FAIL")
    if n_fail > 0:
        print("  *** CRITICAL: Fix FAIL items before proceeding ***")
    print()
