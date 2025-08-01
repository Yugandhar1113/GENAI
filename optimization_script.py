import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.preprocessing import StandardScaler
import optuna
import warnings
warnings.filterwarnings('ignore')

print("🔧 Advanced Optimization Script for Home Credit Default Risk")
print("=" * 60)

# Load the preprocessed data (assuming it's saved from the main script)
def load_preprocessed_data():
    """Load preprocessed data or create it if not available"""
    try:
        # Try to load preprocessed data
        X = pd.read_csv("preprocessed_X.csv")
        y = pd.read_csv("preprocessed_y.csv")
        X_test = pd.read_csv("preprocessed_X_test.csv")
        return X, y['TARGET'], X_test
    except:
        print("Preprocessed data not found. Please run the main script first.")
        return None, None, None

## 1. Feature Selection Optimization
def optimize_feature_selection(X, y, X_test):
    """Optimize feature selection using multiple methods"""
    print("🎯 Optimizing feature selection...")
    
    # Method 1: Statistical feature selection
    selector_kbest = SelectKBest(score_func=f_classif, k='all')
    selector_kbest.fit(X, y)
    kbest_scores = selector_kbest.scores_
    
    # Method 2: Tree-based feature importance
    rf_selector = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_selector.fit(X, y)
    rf_importance = rf_selector.feature_importances_
    
    # Method 3: Extra Trees feature importance
    et_selector = ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    et_selector.fit(X, y)
    et_importance = et_selector.feature_importances_
    
    # Combine feature importance scores
    feature_scores = pd.DataFrame({
        'feature': X.columns,
        'kbest_score': kbest_scores,
        'rf_importance': rf_importance,
        'et_importance': et_importance
    })
    
    # Calculate combined score
    feature_scores['combined_score'] = (
        feature_scores['kbest_score'] * 0.3 +
        feature_scores['rf_importance'] * 0.4 +
        feature_scores['et_importance'] * 0.3
    )
    
    # Select top features
    feature_scores = feature_scores.sort_values('combined_score', ascending=False)
    top_features = feature_scores.head(int(X.shape[1] * 0.8))['feature'].tolist()
    
    print(f"Selected {len(top_features)} features out of {X.shape[1]}")
    
    return X[top_features], X_test[top_features], top_features

## 2. Hyperparameter Optimization with Optuna
def optimize_catboost(X, y, n_trials=50):
    """Optimize CatBoost hyperparameters using Optuna"""
    print("🔍 Optimizing CatBoost hyperparameters...")
    
    def objective(trial):
        params = {
            'iterations': trial.suggest_int('iterations', 500, 3000),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
            'depth': trial.suggest_int('depth', 4, 12),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'random_seed': 42,
            'eval_metric': 'AUC',
            'verbose': 0
        }
        
        # Cross-validation
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = []
        
        for train_idx, valid_idx in skf.split(X, y):
            X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
            y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
            
            model = CatBoostClassifier(**params)
            model.fit(X_train, y_train, eval_set=(X_valid, y_valid), verbose=0)
            pred = model.predict_proba(X_valid)[:, 1]
            score = roc_auc_score(y_valid, pred)
            scores.append(score)
        
        return np.mean(scores)
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    
    print(f"Best CatBoost AUC: {study.best_value:.5f}")
    print(f"Best parameters: {study.best_params}")
    
    return study.best_params

def optimize_lightgbm(X, y, n_trials=50):
    """Optimize LightGBM hyperparameters using Optuna"""
    print("🔍 Optimizing LightGBM hyperparameters...")
    
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 500, 3000),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
            'max_depth': trial.suggest_int('max_depth', 4, 12),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
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
    
    print(f"Best LightGBM AUC: {study.best_value:.5f}")
    print(f"Best parameters: {study.best_params}")
    
    return study.best_params

def optimize_xgboost(X, y, n_trials=50):
    """Optimize XGBoost hyperparameters using Optuna"""
    print("🔍 Optimizing XGBoost hyperparameters...")
    
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 500, 3000),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
            'max_depth': trial.suggest_int('max_depth', 4, 12),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
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
    
    print(f"Best XGBoost AUC: {study.best_value:.5f}")
    print(f"Best parameters: {study.best_params}")
    
    return study.best_params

## 3. Advanced Ensemble with Optimized Models
def create_optimized_ensemble(X, y, X_test, optimized_params):
    """Create ensemble with optimized hyperparameters"""
    print("🎯 Creating optimized ensemble...")
    
    # Get optimized models
    models = {
        'catboost_opt': CatBoostClassifier(**optimized_params['catboost'], verbose=0),
        'lightgbm_opt': LGBMClassifier(**optimized_params['lightgbm']),
        'xgboost_opt': XGBClassifier(**optimized_params['xgboost'])
    }
    
    # K-fold training
    n_folds = 5
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    oof_predictions = {name: np.zeros(X.shape[0]) for name in models}
    test_predictions = {name: np.zeros(X_test.shape[0]) for name in models}
    
    for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y)):
        print(f"  📊 Fold {fold + 1}/{n_folds}")
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
        
        for name, model in models.items():
            if 'catboost' in name:
                model.fit(X_train, y_train, eval_set=(X_valid, y_valid), verbose=0)
            else:
                model.fit(X_train, y_train)
            
            oof_predictions[name][valid_idx] = model.predict_proba(X_valid)[:, 1]
            test_predictions[name] += model.predict_proba(X_test)[:, 1] / n_folds
    
    # Calculate model scores
    model_scores = {}
    for name in models:
        auc = roc_auc_score(y, oof_predictions[name])
        model_scores[name] = auc
        print(f"  {name}: AUC = {auc:.5f}")
    
    return oof_predictions, test_predictions, model_scores

## 4. Advanced Blending Optimization
def optimize_blend_weights(oof_predictions, y, test_predictions, n_trials=100):
    """Optimize blending weights using Optuna"""
    print("🎛️ Optimizing blend weights...")
    
    def objective(trial):
        weights = []
        for i in range(len(oof_predictions)):
            weights.append(trial.suggest_float(f'weight_{i}', 0, 1))
        
        # Normalize weights
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        # Calculate blended OOF predictions
        blended_oof = np.zeros(len(y))
        for i, (name, preds) in enumerate(oof_predictions.items()):
            blended_oof += weights[i] * preds
        
        return roc_auc_score(y, blended_oof)
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    
    # Get optimal weights
    optimal_weights = []
    for i in range(len(oof_predictions)):
        optimal_weights.append(study.best_params[f'weight_{i}'])
    
    optimal_weights = np.array(optimal_weights)
    optimal_weights = optimal_weights / optimal_weights.sum()
    
    print(f"Optimal blend weights: {optimal_weights}")
    print(f"Best blend AUC: {study.best_value:.5f}")
    
    return optimal_weights

## 5. Main Optimization Pipeline
def main_optimization():
    """Main optimization pipeline"""
    print("🚀 Starting optimization pipeline...")
    
    # Load data
    X, y, X_test = load_preprocessed_data()
    if X is None:
        print("Please run the main script first to create preprocessed data.")
        return
    
    # Step 1: Feature selection
    X_selected, X_test_selected, selected_features = optimize_feature_selection(X, y, X_test)
    
    # Step 2: Hyperparameter optimization
    print("\n" + "="*50)
    print("HYPERPARAMETER OPTIMIZATION")
    print("="*50)
    
    optimized_params = {}
    optimized_params['catboost'] = optimize_catboost(X_selected, y, n_trials=30)
    optimized_params['lightgbm'] = optimize_lightgbm(X_selected, y, n_trials=30)
    optimized_params['xgboost'] = optimize_xgboost(X_selected, y, n_trials=30)
    
    # Step 3: Create optimized ensemble
    print("\n" + "="*50)
    print("OPTIMIZED ENSEMBLE CREATION")
    print("="*50)
    
    oof_predictions, test_predictions, model_scores = create_optimized_ensemble(
        X_selected, y, X_test_selected, optimized_params
    )
    
    # Step 4: Optimize blend weights
    print("\n" + "="*50)
    print("BLEND WEIGHT OPTIMIZATION")
    print("="*50)
    
    optimal_weights = optimize_blend_weights(oof_predictions, y, test_predictions, n_trials=50)
    
    # Step 5: Generate final predictions
    print("\n" + "="*50)
    print("FINAL PREDICTION GENERATION")
    print("="*50)
    
    # Create final blended prediction
    final_prediction = np.zeros(X_test_selected.shape[0])
    for i, (name, preds) in enumerate(test_predictions.items()):
        final_prediction += optimal_weights[i] * preds
    
    # Load test IDs
    test_ids = pd.read_csv("../input/home-credit-default-risk/application_test.csv")['SK_ID_CURR']
    
    # Create submission
    submission = pd.DataFrame({
        'SK_ID_CURR': test_ids,
        'TARGET': final_prediction
    })
    
    submission.to_csv("optimized_ensemble_submission.csv", index=False)
    
    print("\n✅ Optimization complete!")
    print("📁 Generated: optimized_ensemble_submission.csv")
    
    # Save optimization results
    results = {
        'selected_features': selected_features,
        'optimized_params': optimized_params,
        'model_scores': model_scores,
        'optimal_weights': optimal_weights.tolist()
    }
    
    import json
    with open('optimization_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("📁 Saved optimization results to: optimization_results.json")

if __name__ == "__main__":
    main_optimization()