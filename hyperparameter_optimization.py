import optuna
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

def optimize_catboost(X, y, n_trials=100):
    """Optimize CatBoost hyperparameters using Optuna"""
    
    def objective(trial):
        params = {
            'iterations': trial.suggest_int('iterations', 500, 3000),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
            'depth': trial.suggest_int('depth', 4, 10),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
            'bootstrap_type': trial.suggest_categorical('bootstrap_type', ['Bernoulli', 'MVS']),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'random_seed': 42,
            'eval_metric': 'AUC',
            'early_stopping_rounds': 100,
            'verbose': 0
        }
        
        # Cross-validation
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = []
        
        for train_idx, valid_idx in skf.split(X, y):
            X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
            y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
            
            model = CatBoostClassifier(**params)
            model.fit(
                X_train, y_train,
                eval_set=(X_valid, y_valid),
                use_best_model=True,
                verbose=0
            )
            
            pred = model.predict_proba(X_valid)[:, 1]
            score = roc_auc_score(y_valid, pred)
            scores.append(score)
        
        return np.mean(scores)
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    
    return study.best_params

def optimize_lightgbm(X, y, n_trials=100):
    """Optimize LightGBM hyperparameters using Optuna"""
    
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 500, 3000),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
            'max_depth': trial.suggest_int('max_depth', 4, 10),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
            'min_child_weight': trial.suggest_float('min_child_weight', 1e-5, 1e-1, log=True),
            'random_state': 42,
            'metric': 'auc',
            'n_jobs': -1
        }
        
        # Cross-validation
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = []
        
        for train_idx, valid_idx in skf.split(X, y):
            X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
            y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
            
            model = LGBMClassifier(**params)
            model.fit(X_train, y_train)
            
            pred = model.predict_proba(X_valid)[:, 1]
            score = roc_auc_score(y_valid, pred)
            scores.append(score)
        
        return np.mean(scores)
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    
    return study.best_params

def optimize_xgboost(X, y, n_trials=100):
    """Optimize XGBoost hyperparameters using Optuna"""
    
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 500, 3000),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
            'max_depth': trial.suggest_int('max_depth', 4, 10),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_weight': trial.suggest_float('min_child_weight', 1e-5, 1e-1, log=True),
            'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
            'random_state': 42,
            'eval_metric': 'auc',
            'n_jobs': -1
        }
        
        # Cross-validation
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = []
        
        for train_idx, valid_idx in skf.split(X, y):
            X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
            y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
            
            model = XGBClassifier(**params)
            model.fit(X_train, y_train)
            
            pred = model.predict_proba(X_valid)[:, 1]
            score = roc_auc_score(y_valid, pred)
            scores.append(score)
        
        return np.mean(scores)
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    
    return study.best_params

def optimize_random_forest(X, y, n_trials=50):
    """Optimize Random Forest hyperparameters using Optuna"""
    
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
            'max_depth': trial.suggest_int('max_depth', 5, 15),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
            'random_state': 42,
            'n_jobs': -1
        }
        
        # Cross-validation
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = []
        
        for train_idx, valid_idx in skf.split(X, y):
            X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
            y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
            
            model = RandomForestClassifier(**params)
            model.fit(X_train, y_train)
            
            pred = model.predict_proba(X_valid)[:, 1]
            score = roc_auc_score(y_valid, pred)
            scores.append(score)
        
        return np.mean(scores)
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    
    return study.best_params

# Example usage (uncomment to run optimization)
"""
# Load your data here
# X = your_features
# y = your_target

print("🔍 Optimizing CatBoost...")
best_catboost_params = optimize_catboost(X, y, n_trials=50)
print("Best CatBoost params:", best_catboost_params)

print("🔍 Optimizing LightGBM...")
best_lightgbm_params = optimize_lightgbm(X, y, n_trials=50)
print("Best LightGBM params:", best_lightgbm_params)

print("🔍 Optimizing XGBoost...")
best_xgboost_params = optimize_xgboost(X, y, n_trials=50)
print("Best XGBoost params:", best_xgboost_params)

print("🔍 Optimizing Random Forest...")
best_rf_params = optimize_random_forest(X, y, n_trials=25)
print("Best Random Forest params:", best_rf_params)
"""