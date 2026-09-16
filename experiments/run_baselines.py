"""Phase 1+2: Leakage audit + reproduce baselines with site-aware CV.
All preprocessing fit on train fold only. No global operations."""
import sys, os, json, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import load_labels, load_all_features, load_cnn_oof, build_feature_group
from src.validation import get_fold_splits, print_fold_stats
from src.metrics import compute_all_metrics, compute_site_metrics, bootstrap_ci
from src.preprocessing import FoldSafePreprocessor, TreePreprocessor
from src.tabular_models import fit_with_search, predict_safe
from src.calibration import platt_scale, isotonic_calibrate, temperature_scale, fit_power_gamma, power_sharpen
from src.leakage_audit import audit_leakage, print_audit

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def main():
    os.makedirs(os.path.join(BASE, 'experiments'), exist_ok=True)
    os.makedirs(os.path.join(BASE, 'oof'), exist_ok=True)
    os.makedirs(os.path.join(BASE, 'reports'), exist_ok=True)

    # ===== PHASE 1: LEAKAGE AUDIT =====
    print("\n" + "=" * 70)
    print("PHASE 1: LEAKAGE AUDIT")
    print("=" * 70)
    results_audit = audit_leakage(BASE)
    print_audit(results_audit)

    # Save audit
    with open(os.path.join(BASE, 'reports', 'leakage_audit.txt'), 'w') as f:
        for name, status, detail in results_audit:
            f.write(f"{status}: {name}\n  {detail}\n")

    # ===== LOAD DATA =====
    print("Loading data...")
    y, groups, uids = load_labels()
    features = load_all_features()
    cnn_oof, cnn_names = load_cnn_oof()
    N = len(y)
    print(f"  N={N}, prevalence={y.mean():.3f}, sites={len(np.unique(groups))}")

    # ===== DEFINE FEATURE SETS =====
    feature_configs = {
        'SBR': ['SBR'],
        'SBR_PHYS': ['SBR', 'PHYS'],
        'SBR_CNN': ['SBR'],  # + cnn_oof
        'SBR_PHYS_CNN': ['SBR', 'PHYS'],  # + cnn_oof
        'SBR_PCA': ['SBR'],  # + pca50
        'SBR_PHYS_PCA': ['SBR', 'PHYS'],  # + pca50
        'SBR_CNN_PCA': ['SBR'],  # + cnn_oof + pca50
        'SBR_PHYS_CNN_PCA': ['SBR', 'PHYS'],  # + cnn_oof + pca50
        'PHYS_RAD_CNN': ['PHYS', 'RAD_INORM', 'ROI_RAD', 'TEXTURE'],  # + cnn_oof
    }

    def build_X(config_name):
        groups_list = feature_configs[config_name]
        X_base, cols_base = build_feature_group(groups_list, features)
        extras = []
        if 'CNN' in config_name:
            extras.append(cnn_oof)
        if 'PCA' in config_name:
            extras.append(features['PCA50'][0])
        if extras:
            X = np.hstack([X_base] + extras)
        else:
            X = X_base
        return X

    # ===== MODELS TO TEST =====
    model_names = ['LR', 'Ridge', 'ET', 'XGB', 'LGB']

    # ===== PHASE 2: REPRODUCE BASELINES =====
    print("\n" + "=" * 70)
    print("PHASE 2: REPRODUCE BASELINES (SITE-AWARE CV)")
    print("=" * 70)

    all_experiments = []

    for feat_name in feature_configs:
        X = build_X(feat_name)
        print(f"\n--- Feature set: {feat_name} ({X.shape[1]} features) ---")

        splits = get_fold_splits(X, y, groups, cv_type='site_aware')
        print_fold_stats(splits, y, groups, 'site_aware')

        for model_name in model_names:
            t0 = time.time()
            oof_raw = np.zeros(N)
            oof_platt = np.zeros(N)
            oof_isotonic = np.zeros(N)
            oof_temp = np.zeros(N)
            oof_gamma = np.zeros(N)
            fold_metrics = []

            for fold_idx, (trn_idx, val_idx) in enumerate(splits):
                X_trn, X_val = X[trn_idx], X[val_idx]
                y_trn, y_val = y[trn_idx], y[val_idx]

                # Preprocess: fit on train only
                if model_name in ('LR', 'Ridge'):
                    pp = FoldSafePreprocessor(do_impute=True, do_scale=True)
                else:
                    pp = TreePreprocessor()
                X_trn_p, _ = pp.fit_transform(X_trn)
                X_val_p = pp.transform(X_val)

                # Fit model with search
                model, _ = fit_with_search(model_name, X_trn_p, y_trn, X_val_p)
                pred_val = predict_safe(model, X_val_p)
                oof_raw[val_idx] = pred_val

                # Calibration: fit on inner train split
                n_inner = len(trn_idx)
                inner_trn = trn_idx[:n_inner * 8 // 10]
                inner_cal = trn_idx[n_inner * 8 // 10:]

                X_inner_trn, _ = pp.fit_transform(X[inner_trn])
                X_inner_cal = pp.transform(X[inner_cal])
                model_inner, _ = fit_with_search(model_name, X_inner_trn, y[inner_trn], X_inner_cal)
                pred_inner_trn = predict_safe(model_inner, X_inner_trn)
                pred_inner_cal = predict_safe(model_inner, X_inner_cal)

                # Fit calibration on inner cal, apply to val
                oof_platt[val_idx] = platt_scale(pred_inner_cal, y[inner_cal], pred_val)
                oof_isotonic[val_idx] = isotonic_calibrate(pred_inner_cal, y[inner_cal], pred_val)
                oof_temp[val_idx] = temperature_scale(pred_inner_cal, y[inner_cal], pred_val)

                gamma = fit_power_gamma(pred_inner_cal, y[inner_cal])
                oof_gamma[val_idx] = power_sharpen(pred_val, gamma)

                fold_m = compute_all_metrics(y_val, pred_val, prefix=f'f{fold_idx}_')
                fold_metrics.append(fold_m)

            elapsed = time.time() - t0

            # Overall metrics
            for tag, oof in [('raw', oof_raw), ('platt', oof_platt),
                             ('isotonic', oof_isotonic), ('temp', oof_temp),
                             ('gamma', oof_gamma)]:
                m = compute_all_metrics(y, oof, prefix='')
                site_m = compute_site_metrics(y, oof, groups)
                site_aucs = [v['auc'] for v in site_m.values() if not np.isnan(v.get('auc', np.nan))]
                site_lls = [v['logloss'] for v in site_m.values() if not np.isnan(v.get('logloss', np.nan))]

                exp_result = {
                    'feature_set': feat_name,
                    'model': model_name,
                    'calibration': tag,
                    'n_features': X.shape[1],
                    'cv_type': 'site_aware',
                    'n_folds': len(splits),
                    'auc': m['auc'],
                    'logloss': m['logloss'],
                    'brier': m['brier'],
                    'accuracy': m['accuracy'],
                    'f1': m['f1'],
                    'site_auc_mean': np.mean(site_aucs) if site_aucs else float('nan'),
                    'site_auc_std': np.std(site_aucs) if site_aucs else float('nan'),
                    'site_logloss_mean': np.mean(site_lls) if site_lls else float('nan'),
                    'site_logloss_std': np.std(site_lls) if site_lls else float('nan'),
                    'elapsed': elapsed,
                }
                all_experiments.append(exp_result)

                if tag == 'raw':
                    # Save OOF
                    np.save(os.path.join(BASE, 'oof',
                            f'{model_name.lower()}_{feat_name.lower()}_raw.npy'), oof)

                    marker = ' ***' if m['logloss'] < 0.35 else ''
                    print(f"  {model_name:5s} {feat_name:20s} raw: "
                          f"AUC={m['auc']:.4f} LL={m['logloss']:.4f} "
                          f"Brier={m['brier']:.4f} ({elapsed:.0f}s){marker}")

    # ===== SAVE RESULTS =====
    import pandas as pd
    df = pd.DataFrame(all_experiments)
    df.to_csv(os.path.join(BASE, 'experiments', 'results.csv'), index=False)

    # Top 10 by AUC
    print("\n\n" + "=" * 70)
    print("TOP 10 BY AUROC (raw probabilities)")
    print("=" * 70)
    raw = df[df['calibration'] == 'raw'].sort_values('auc', ascending=False)
    print(raw[['model', 'feature_set', 'n_features', 'auc', 'logloss', 'brier']].head(10).to_string())

    print("\n\nTOP 10 BY LOGLOSS (raw probabilities)")
    print("=" * 70)
    raw_ll = df[df['calibration'] == 'raw'].sort_values('logloss')
    print(raw_ll[['model', 'feature_set', 'n_features', 'auc', 'logloss', 'brier']].head(10).to_string())

    # Site-wise for best model
    best = raw.iloc[0]
    print(f"\n\nSITE-WISE ANALYSIS: {best['model']} + {best['feature_set']}")
    print("=" * 70)
    X_best = build_X(best['feature_set'])
    splits = get_fold_splits(X_best, y, groups, cv_type='site_aware')

    oof_best = np.zeros(N)
    for trn_idx, val_idx in splits:
        X_trn, X_val = X_best[trn_idx], X_best[val_idx]
        y_trn = y[trn_idx]
        pp = TreePreprocessor() if best['model'] in ('XGB', 'LGB', 'ET') else FoldSafePreprocessor(True, True)
        X_trn_p, _ = pp.fit_transform(X_trn)
        X_val_p = pp.transform(X_val)
        model, _ = fit_with_search(best['model'], X_trn_p, y_trn, X_val_p)
        oof_best[val_idx] = predict_safe(model, X_val_p)

    site_m = compute_site_metrics(y, oof_best, groups)
    print(f"{'Site':<20} {'N':>5} {'Pos':>5} {'AUC':>7} {'LL':>7} {'Brier':>7} {'MeanP':>7}")
    print("-" * 65)
    for site, m in sorted(site_m.items(), key=lambda x: x[1].get('auc', 0)):
        print(f"{str(site):<20} {m['n']:>5} {m['pos_count']:>5} "
              f"{m.get('auc', float('nan')):>7.4f} {m.get('logloss', float('nan')):>7.4f} "
              f"{m.get('brier', float('nan')):>7.4f} {m['mean_pred']:>7.4f}")

    # Bootstrap CI for best
    print("\n\nBOOTSTRAP CI for best model:")
    auc_mean, auc_lo, auc_hi = bootstrap_ci(y, oof_best, lambda yt, yp: __import__('sklearn.metrics', fromlist=['roc_auc_score']).roc_auc_score(yt, yp))
    ll_mean, ll_lo, ll_hi = bootstrap_ci(y, oof_best, lambda yt, yp: __import__('sklearn.metrics', fromlist=['log_loss']).log_loss(yt, yp))
    print(f"  AUC: {auc_mean:.4f} [{auc_lo:.4f}, {auc_hi:.4f}]")
    print(f"  LL:  {ll_mean:.4f} [{ll_lo:.4f}, {ll_hi:.4f}]")

    # Save final report
    report = {
        'best_single_model': f"{best['model']} + {best['feature_set']}",
        'validation': 'Site-aware StratifiedGroupKFold',
        'auc': float(best['auc']),
        'logloss': float(best['logloss']),
        'brier': float(best['brier']),
        'auc_95ci': [float(auc_lo), float(auc_hi)],
        'll_95ci': [float(ll_lo), float(ll_hi)],
        'site_auc_mean': float(best['site_auc_mean']),
        'site_auc_std': float(best['site_auc_std']),
        'site_logloss_mean': float(best['site_logloss_mean']),
        'site_logloss_std': float(best['site_logloss_std']),
    }
    with open(os.path.join(BASE, 'reports', 'final_report.json'), 'w') as f:
        json.dump(report, f, indent=2)

    print("\n\nDone. Results saved to experiments/results.csv and reports/final_report.json")


if __name__ == '__main__':
    main()
