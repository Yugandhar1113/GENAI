import pandas as pd
import numpy as np
import optuna
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

class HyperparameterOptimizer:
    def __init__(self, X, y, n_folds=5, random_state=42):
        self.X = X
        self.y = y
        self.n_folds = n_folds
        self.random_state = random_state
        self.kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    
    def objective_catboost(self, trial):
        """Optimize CatBoost hyperparameters"""
        params = {
            'iterations': trial.suggest_int('iterations', 1000, 3000),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
            'depth': trial.suggest_int('depth', 4, 10),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
            'border_count': trial.suggest_int('border_count', 32, 255),
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 1),
            'random_strength': trial.suggest_float('random_strength', 0, 1),
            'eval_metric': 'AUC',
            'early_stopping_rounds': 200,
            'random_seed': self.random_state,
            'thread_count': -1,
            'verbose': 0
        }
        
        cv_scores = []
        for fold, (train_idx, valid_idx) in enumerate(self.kf.split(self.X, self.y)):
            X_train, X_valid = self.X.iloc[train_idx], self.X.iloc[valid_idx]
            y_train, y_valid = self.y.iloc[train_idx], self.y.iloc[valid_idx]
            
            model = CatBoostClassifier(**params)
            model.fit(
                X_train, y_train,
                eval_set=(X_valid, y_valid),
                use_best_model=True,
                verbose=0
            )
            
            pred = model.predict_proba(X_valid)[:, 1]
            score = roc_auc_score(y_valid, pred)
            cv_scores.append(score)
        
        return np.mean(cv_scores)
    
    def objective_lightgbm(self, trial):
        """Optimize LightGBM hyperparameters"""
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 1000, 3000),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
            'max_depth': trial.suggest_int('max_depth', 4, 10),
            'num_leaves': trial.suggest_int('num_leaves', 16, 128),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'metric': 'auc',
            'random_state': self.random_state,
            'n_jobs': -1
        }
        
        cv_scores = []
        for fold, (train_idx, valid_idx) in enumerate(self.kf.split(self.X, self.y)):
            X_train, X_valid = self.X.iloc[train_idx], self.X.iloc[valid_idx]
            y_train, y_valid = self.y.iloc[train_idx], self.y.iloc[valid_idx]
            
            model = LGBMClassifier(**params)
            model.fit(
                X_train, y_train,
                eval_set=[(X_valid, y_valid)],
                callbacks=[LGBMClassifier().early_stopping(100)]
            )
            
            pred = model.predict_proba(X_valid)[:, 1]
            score = roc_auc_score(y_valid, pred)
            cv_scores.append(score)
        
        return np.mean(cv_scores)
    
    def objective_xgboost(self, trial):
        """Optimize XGBoost hyperparameters"""
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 1000, 3000),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
            'max_depth': trial.suggest_int('max_depth', 4, 10),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.6, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'eval_metric': 'auc',
            'random_state': self.random_state,
            'n_jobs': -1
        }
        
        cv_scores = []
        for fold, (train_idx, valid_idx) in enumerate(self.kf.split(self.X, self.y)):
            X_train, X_valid = self.X.iloc[train_idx], self.X.iloc[valid_idx]
            y_train, y_valid = self.y.iloc[train_idx], self.y.iloc[valid_idx]
            
            model = XGBClassifier(**params)
            model.fit(
                X_train, y_train,
                eval_set=[(X_valid, y_valid)],
                verbose=False
            )
            
            pred = model.predict_proba(X_valid)[:, 1]
            score = roc_auc_score(y_valid, pred)
            cv_scores.append(score)
        
        return np.mean(cv_scores)
    
    def objective_random_forest(self, trial):
        """Optimize Random Forest hyperparameters"""
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 500, 1500),
            'max_depth': trial.suggest_int('max_depth', 5, 15),
            'min_samples_split': trial.suggest_int('min_samples_split', 10, 50),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 5, 25),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', 0.8, 0.9]),
            'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
            'random_state': self.random_state,
            'n_jobs': -1
        }
        
        cv_scores = []
        for fold, (train_idx, valid_idx) in enumerate(self.kf.split(self.X, self.y)):
            X_train, X_valid = self.X.iloc[train_idx], self.X.iloc[valid_idx]
            y_train, y_valid = self.y.iloc[train_idx], self.y.iloc[valid_idx]
            
            model = RandomForestClassifier(**params)
            model.fit(X_train, y_train)
            
            pred = model.predict_proba(X_valid)[:, 1]
            score = roc_auc_score(y_valid, pred)
            cv_scores.append(score)
        
        return np.mean(cv_scores)
    
    def optimize_all_models(self, n_trials=100):
        """Optimize all models and return best parameters"""
        print("Optimizing hyperparameters for all models...")
        
        best_params = {}
        
        # Optimize CatBoost
        print("Optimizing CatBoost...")
        study_cb = optuna.create_study(direction='maximize')
        study_cb.optimize(self.objective_catboost, n_trials=n_trials)
        best_params['catboost'] = study_cb.best_params
        print(f"Best CatBoost AUC: {study_cb.best_value:.5f}")
        
        # Optimize LightGBM
        print("Optimizing LightGBM...")
        study_lgb = optuna.create_study(direction='maximize')
        study_lgb.optimize(self.objective_lightgbm, n_trials=n_trials)
        best_params['lightgbm'] = study_lgb.best_params
        print(f"Best LightGBM AUC: {study_lgb.best_value:.5f}")
        
        # Optimize XGBoost
        print("Optimizing XGBoost...")
        study_xgb = optuna.create_study(direction='maximize')
        study_xgb.optimize(self.objective_xgboost, n_trials=n_trials)
        best_params['xgboost'] = study_xgb.best_params
        print(f"Best XGBoost AUC: {study_xgb.best_value:.5f}")
        
        # Optimize Random Forest
        print("Optimizing Random Forest...")
        study_rf = optuna.create_study(direction='maximize')
        study_rf.optimize(self.objective_random_forest, n_trials=n_trials)
        best_params['random_forest'] = study_rf.best_params
        print(f"Best Random Forest AUC: {study_rf.best_value:.5f}")
        
        return best_params

def load_and_prepare_data(train_path, sample_size=50000):
    """Load and prepare data for hyperparameter optimization"""
    print("Loading and preparing data...")
    
    train = pd.read_csv(train_path)
    
    # Sample data for faster optimization
    if len(train) > sample_size:
        train = train.sample(n=sample_size, random_state=42)
    
    # Basic feature engineering
    train['CREDIT_INCOME_RATIO'] = train['AMT_CREDIT'] / (train['AMT_INCOME_TOTAL'] + 1)
    train['ANNUITY_INCOME_RATIO'] = train['AMT_ANNUITY'] / (train['AMT_INCOME_TOTAL'] + 1)
    train['CREDIT_ANNUITY_RATIO'] = train['AMT_CREDIT'] / (train['AMT_ANNUITY'] + 1)
    train['DAYS_BIRTH_ABS'] = abs(train['DAYS_BIRTH'])
    train['DAYS_EMPLOYED_ABS'] = abs(train['DAYS_EMPLOYED'])
    train['EXT_SOURCE_MEAN'] = train[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].mean(axis=1)
    
    # Handle categorical features
    cat_features = train.select_dtypes(include=['object']).columns.tolist()
    if 'TARGET' in cat_features:
        cat_features.remove('TARGET')
    if 'SK_ID_CURR' in cat_features:
        cat_features.remove('SK_ID_CURR')
    
    for col in cat_features:
        train[col].fillna('MISSING', inplace=True)
        if train[col].nunique() > 2:
            lbl = LabelEncoder()
            train[col] = lbl.fit_transform(train[col].astype(str))
    
    # Handle numerical features
    num_features = [col for col in train.columns 
                   if col not in cat_features + ['TARGET', 'SK_ID_CURR']]
    train[num_features] = SimpleImputer(strategy='median').fit_transform(train[num_features])
    
    X = train.drop(columns=['TARGET', 'SK_ID_CURR'])
    y = train['TARGET']
    
    return X, y

if __name__ == "__main__":
    # Load and prepare data
    train_path = "../input/home-credit-default-risk/application_train.csv"
    
    try:
        X, y = load_and_prepare_data(train_path, sample_size=30000)  # Use smaller sample for speed
        
        # Initialize optimizer
        optimizer = HyperparameterOptimizer(X, y, n_folds=3, random_state=42)  # Use 3 folds for speed
        
        # Optimize hyperparameters
        best_params = optimizer.optimize_all_models(n_trials=50)  # Use fewer trials for speed
        
        # Save results
        import json
        with open('best_hyperparameters.json', 'w') as f:
            json.dump(best_params, f, indent=2)
        
        print("\n✅ Hyperparameter optimization completed!")
        print("Best parameters saved to 'best_hyperparameters.json'")
        
        # Print summary
        print("\nBest Parameters Summary:")
        for model_name, params in best_params.items():
            print(f"\n{model_name.upper()}:")
            for param, value in params.items():
                print(f"  {param}: {value}")
                
    except FileNotFoundError:
        print("Data file not found. Please ensure the data file is in the correct path.")
        print("Expected path: ../input/home-credit-default-risk/application_train.csv")