"""Tabular model definitions. All models use conservative regularization."""
import numpy as np
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import ExtraTreesClassifier
import xgboost as xgb
import lightgbm as lgb


def get_model(model_name, random_state=42):
    """Return a model instance with conservative defaults."""
    if model_name == 'LR':
        return LogisticRegression(C=1.0, max_iter=2000, random_state=random_state)
    elif model_name == 'Ridge':
        return RidgeClassifier(alpha=1.0, random_state=random_state)
    elif model_name == 'ET':
        return ExtraTreesClassifier(
            n_estimators=500, max_depth=6, min_samples_leaf=5,
            random_state=random_state, n_jobs=-1)
    elif model_name == 'XGB':
        return xgb.XGBClassifier(
            n_estimators=600, max_depth=4, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.7, min_child_weight=5,
            reg_alpha=0.1, reg_lambda=5.0,
            objective='binary:logistic', eval_metric='logloss',
            use_label_encoder=False, random_state=random_state, n_jobs=-1)
    elif model_name == 'LGB':
        return lgb.LGBMClassifier(
            n_estimators=600, max_depth=4, learning_rate=0.03,
            num_leaves=15, subsample=0.8, colsample_bytree=0.7,
            min_child_samples=20, reg_alpha=0.1, reg_lambda=5.0,
            objective='binary', metric='binary_logloss',
            random_state=random_state, n_jobs=-1, verbose=-1)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def get_search_space(model_name):
    """Return hyperparameter search space for model selection."""
    if model_name == 'XGB':
        return [
            {'n_estimators': 400, 'max_depth': 3, 'learning_rate': 0.05,
             'subsample': 0.8, 'colsample_bytree': 0.7, 'min_child_weight': 5,
             'reg_alpha': 0.1, 'reg_lambda': 5.0},
            {'n_estimators': 600, 'max_depth': 4, 'learning_rate': 0.03,
             'subsample': 0.8, 'colsample_bytree': 0.7, 'min_child_weight': 5,
             'reg_alpha': 0.1, 'reg_lambda': 5.0},
            {'n_estimators': 800, 'max_depth': 2, 'learning_rate': 0.02,
             'subsample': 0.85, 'colsample_bytree': 0.6, 'min_child_weight': 10,
             'reg_alpha': 1.0, 'reg_lambda': 10.0},
            {'n_estimators': 400, 'max_depth': 4, 'learning_rate': 0.05,
             'subsample': 0.7, 'colsample_bytree': 0.8, 'min_child_weight': 3,
             'reg_alpha': 0.0, 'reg_lambda': 1.0},
            {'n_estimators': 800, 'max_depth': 3, 'learning_rate': 0.02,
             'subsample': 1.0, 'colsample_bytree': 1.0, 'min_child_weight': 10,
             'reg_alpha': 0.1, 'reg_lambda': 10.0},
        ]
    elif model_name == 'LGB':
        return [
            {'n_estimators': 400, 'max_depth': 3, 'learning_rate': 0.05,
             'num_leaves': 7, 'subsample': 0.8, 'colsample_bytree': 0.7,
             'min_child_samples': 20, 'reg_alpha': 0.1, 'reg_lambda': 5.0},
            {'n_estimators': 600, 'max_depth': 4, 'learning_rate': 0.03,
             'num_leaves': 15, 'subsample': 0.8, 'colsample_bytree': 0.7,
             'min_child_samples': 20, 'reg_alpha': 0.1, 'reg_lambda': 5.0},
            {'n_estimators': 800, 'max_depth': -1, 'learning_rate': 0.02,
             'num_leaves': 31, 'subsample': 0.85, 'colsample_bytree': 0.6,
             'min_child_samples': 40, 'reg_alpha': 1.0, 'reg_lambda': 10.0},
            {'n_estimators': 400, 'max_depth': 2, 'learning_rate': 0.05,
             'num_leaves': 7, 'subsample': 0.7, 'colsample_bytree': 0.8,
             'min_child_samples': 60, 'reg_alpha': 0.0, 'reg_lambda': 1.0},
            {'n_estimators': 800, 'max_depth': 3, 'learning_rate': 0.02,
             'num_leaves': 15, 'subsample': 1.0, 'colsample_bytree': 1.0,
             'min_child_samples': 40, 'reg_alpha': 0.1, 'reg_lambda': 10.0},
        ]
    elif model_name == 'LR':
        return [{'C': 0.1}, {'C': 1.0}, {'C': 10.0}]
    elif model_name == 'Ridge':
        return [{'alpha': 0.1}, {'alpha': 1.0}, {'alpha': 10.0}]
    elif model_name == 'ET':
        return [
            {'n_estimators': 500, 'max_depth': 6, 'min_samples_leaf': 5},
            {'n_estimators': 500, 'max_depth': 8, 'min_samples_leaf': 3},
            {'n_estimators': 300, 'max_depth': 10, 'min_samples_leaf': 10},
        ]
    else:
        return [{}]


def fit_with_search(model_name, X_trn, y_trn, X_val, random_state=42):
    """Train model with simple search, return best model and its train AUC."""
    from sklearn.metrics import roc_auc_score
    space = get_search_space(model_name)
    best_score = -1
    best_model = None

    for params in space:
        try:
            model = get_model(model_name, random_state=random_state)
            model.set_params(**params)

            if model_name in ('LR', 'Ridge'):
                model.fit(X_trn, y_trn)
                pred_trn = model.decision_function(X_trn)
                if pred_trn.ndim > 1:
                    pred_trn = pred_trn[:, 1]
                pred_trn = 1 / (1 + np.exp(-pred_trn))
            else:
                model.fit(X_trn, y_trn)
                pred_trn = model.predict_proba(X_trn)[:, 1]

            score = roc_auc_score(y_trn, pred_trn)
            if score > best_score:
                best_score = score
                best_model = model
        except Exception:
            continue

    if best_model is None:
        best_model = get_model(model_name, random_state=random_state)
        best_model.fit(X_trn, y_trn)

    return best_model, best_score


def predict_safe(model, X):
    """Predict probabilities, handling both proba and decision_function."""
    if hasattr(model, 'predict_proba'):
        return model.predict_proba(X)[:, 1]
    else:
        scores = model.decision_function(X)
        return 1 / (1 + np.exp(-scores))
