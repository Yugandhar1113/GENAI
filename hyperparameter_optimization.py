import pandas as pd
import numpy as np
import optuna
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# Load preprocessed data (assuming it's saved from the main script)
def load_preprocessed_data():
    """Load preprocessed data for hyperparameter optimization"""
    # This would load the data that was preprocessed in the main script
    # For now, we'll assume the data is available
    pass

def objective_catboost(trial):
    """Optuna objective for CatBoost hyperparameter optimization"""
    
    # Define hyperparameter search space
    params = {
        'iterations': trial.suggest_int('iterations', 1000, 3000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'depth': trial.suggest_int('depth', 4, 10),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
        'bootstrap_type': trial.suggest_categorical('bootstrap_type', ['Bernoulli', 'MVS']),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'random_seed': 42,
        'eval_metric': 'AUC',
        'early_stopping_rounds': 150,
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

def objective_lightgbm(trial):
    """Optuna objective for LightGBM hyperparameter optimization"""
    
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 1000, 3000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'max_depth': trial.suggest_int('max_depth', 4, 10),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
        'min_child_weight': trial.suggest_float('min_child_weight', 1e-3, 1e-1, log=True),
        'random_state': 42,
        'metric': 'auc',
        'n_jobs': -1
    }
    
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

def objective_xgboost(trial):
    """Optuna objective for XGBoost hyperparameter optimization"""
    
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 1000, 3000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'max_depth': trial.suggest_int('max_depth', 4, 10),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'random_state': 42,
        'eval_metric': 'auc',
        'n_jobs': -1,
        'tree_method': 'hist'
    }
    
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

def optimize_hyperparameters():
    """Run hyperparameter optimization for all models"""
    
    print("🔍 Starting hyperparameter optimization...")
    
    # Optimize CatBoost
    print("\n🎯 Optimizing CatBoost...")
    study_catboost = optuna.create_study(direction='maximize')
    study_catboost.optimize(objective_catboost, n_trials=50)
    
    print(f"Best CatBoost AUC: {study_catboost.best_value:.5f}")
    print(f"Best CatBoost params: {study_catboost.best_params}")
    
    # Optimize LightGBM
    print("\n🎯 Optimizing LightGBM...")
    study_lightgbm = optuna.create_study(direction='maximize')
    study_lightgbm.optimize(objective_lightgbm, n_trials=50)
    
    print(f"Best LightGBM AUC: {study_lightgbm.best_value:.5f}")
    print(f"Best LightGBM params: {study_lightgbm.best_params}")
    
    # Optimize XGBoost
    print("\n🎯 Optimizing XGBoost...")
    study_xgboost = optuna.create_study(direction='maximize')
    study_xgboost.optimize(objective_xgboost, n_trials=50)
    
    print(f"Best XGBoost AUC: {study_xgboost.best_value:.5f}")
    print(f"Best XGBoost params: {study_xgboost.best_params}")
    
    # Save optimization results
    results = {
        'catboost': {
            'best_score': study_catboost.best_value,
            'best_params': study_catboost.best_params
        },
        'lightgbm': {
            'best_score': study_lightgbm.best_value,
            'best_params': study_lightgbm.best_params
        },
        'xgboost': {
            'best_score': study_xgboost.best_value,
            'best_params': study_xgboost.best_params
        }
    }
    
    return results

if __name__ == "__main__":
    # This script should be run after the main preprocessing
    print("🚀 Hyperparameter optimization script")
    print("Make sure to run the main ensemble script first to preprocess the data")
    
    # Uncomment the following lines when you have the preprocessed data
    # results = optimize_hyperparameters()
    # print("\n✅ Hyperparameter optimization completed!")