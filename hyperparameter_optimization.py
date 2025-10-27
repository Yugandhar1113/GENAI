import pandas as pd
import numpy as np
import optuna
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
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
        """Objective function for CatBoost optimization"""
        params = {
            'iterations': trial.suggest_int('iterations', 1000, 3000),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'depth': trial.suggest_int('depth', 4, 10),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
            'random_strength': trial.suggest_float('random_strength', 1e-9, 10, log=True),
            'eval_metric': 'AUC',
            'early_stopping_rounds': 100,
            'random_seed': self.random_state,
            'thread_count': -1,
            'verbose': 0
        }
        
        model = CatBoostClassifier(**params)
        scores = []
        
        for train_idx, valid_idx in self.kf.split(self.X, self.y):
            X_train, X_valid = self.X.iloc[train_idx], self.X.iloc[valid_idx]
            y_train, y_valid = self.y.iloc[train_idx], self.y.iloc[valid_idx]
            
            model.fit(X_train, y_train, eval_set=(X_valid, y_valid), use_best_model=True)
            pred = model.predict_proba(X_valid)[:, 1]
            score = roc_auc_score(y_valid, pred)
            scores.append(score)
        
        return np.mean(scores)
    
    def objective_lightgbm(self, trial):
        """Objective function for LightGBM optimization"""
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 1000, 3000),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'num_leaves': trial.suggest_int('num_leaves', 10, 300),
            'min_child_samples': trial.suggest_int('min_child_samples', 20, 100),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
            'metric': 'auc',
            'random_state': self.random_state,
            'n_jobs': -1,
            'verbose': -1
        }
        
        model = LGBMClassifier(**params)
        scores = []
        
        for train_idx, valid_idx in self.kf.split(self.X, self.y):
            X_train, X_valid = self.X.iloc[train_idx], self.X.iloc[valid_idx]
            y_train, y_valid = self.y.iloc[train_idx], self.y.iloc[valid_idx]
            
            model.fit(
                X_train, y_train,
                eval_set=[(X_valid, y_valid)],
                callbacks=[
                    LGBMClassifier.early_stopping(100),
                    LGBMClassifier.log_evaluation(0)
                ]
            )
            pred = model.predict_proba(X_valid)[:, 1]
            score = roc_auc_score(y_valid, pred)
            scores.append(score)
        
        return np.mean(scores)
    
    def objective_xgboost(self, trial):
        """Objective function for XGBoost optimization"""
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 1000, 3000),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
            'eval_metric': 'auc',
            'random_state': self.random_state,
            'n_jobs': -1,
            'tree_method': 'hist'
        }
        
        model = XGBClassifier(**params)
        scores = []
        
        for train_idx, valid_idx in self.kf.split(self.X, self.y):
            X_train, X_valid = self.X.iloc[train_idx], self.X.iloc[valid_idx]
            y_train, y_valid = self.y.iloc[train_idx], self.y.iloc[valid_idx]
            
            model.fit(
                X_train, y_train,
                eval_set=[(X_valid, y_valid)],
                early_stopping_rounds=100,
                verbose=0
            )
            pred = model.predict_proba(X_valid)[:, 1]
            score = roc_auc_score(y_valid, pred)
            scores.append(score)
        
        return np.mean(scores)
    
    def objective_random_forest(self, trial):
        """Objective function for Random Forest optimization"""
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
            'random_state': self.random_state,
            'n_jobs': -1
        }
        
        model = RandomForestClassifier(**params)
        scores = cross_val_score(model, self.X, self.y, cv=self.kf, scoring='roc_auc', n_jobs=-1)
        return np.mean(scores)
    
    def optimize_model(self, model_name, n_trials=100):
        """Optimize hyperparameters for a specific model"""
        print(f"Optimizing {model_name}...")
        
        if model_name == 'catboost':
            objective = self.objective_catboost
        elif model_name == 'lightgbm':
            objective = self.objective_lightgbm
        elif model_name == 'xgboost':
            objective = self.objective_xgboost
        elif model_name == 'random_forest':
            objective = self.objective_random_forest
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=self.random_state))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        print(f"Best {model_name} score: {study.best_value:.5f}")
        print(f"Best {model_name} params: {study.best_params}")
        
        return study.best_params, study.best_value

def optimize_ensemble_weights(oof_predictions, y, n_trials=1000):
    """Optimize ensemble weights using Optuna"""
    def objective(trial):
        weights = []
        for i in range(len(oof_predictions)):
            weight = trial.suggest_float(f'weight_{i}', 0.0, 1.0)
            weights.append(weight)
        
        # Normalize weights
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        # Calculate weighted prediction
        weighted_pred = np.zeros(len(y))
        for i, (model_name, pred) in enumerate(oof_predictions.items()):
            weighted_pred += weights[i] * pred
        
        return roc_auc_score(y, weighted_pred)
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    # Extract optimized weights
    best_weights = []
    for i in range(len(oof_predictions)):
        best_weights.append(study.best_params[f'weight_{i}'])
    
    best_weights = np.array(best_weights)
    best_weights = best_weights / best_weights.sum()
    
    return best_weights, study.best_value

def load_and_preprocess_data():
    """Load and preprocess the Home Credit dataset"""
    print("Loading and preprocessing data...")
    
    try:
        train = pd.read_csv("../input/home-credit-default-risk/application_train.csv")
        test = pd.read_csv("../input/home-credit-default-risk/application_test.csv")
    except FileNotFoundError:
        print("Data files not found. Creating sample data for demonstration...")
        # Create sample data for demonstration
        np.random.seed(42)
        n_samples = 1000
        n_features = 50
        
        X_sample = np.random.randn(n_samples, n_features)
        y_sample = np.random.randint(0, 2, n_samples)
        
        train = pd.DataFrame(X_sample, columns=[f'feature_{i}' for i in range(n_features)])
        train['TARGET'] = y_sample
        train['SK_ID_CURR'] = range(n_samples)
        
        return train.drop(['TARGET', 'SK_ID_CURR'], axis=1), train['TARGET']
    
    # Basic feature engineering
    def basic_feature_engineering(df):
        # Create basic features
        df['CREDIT_INCOME_PERCENT'] = df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL']
        df['ANNUITY_INCOME_PERCENT'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
        df['CREDIT_TERM'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']
        df['DAYS_EMPLOYED_PERCENT'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
        
        # External sources
        ext_sources = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']
        df['EXT_SOURCES_MEAN'] = df[ext_sources].mean(axis=1)
        df['EXT_SOURCES_STD'] = df[ext_sources].std(axis=1)
        
        return df
    
    train = basic_feature_engineering(train)
    
    # Handle categorical features
    cat_features = train.select_dtypes(include=['object']).columns.tolist()
    for col in cat_features:
        train[col].fillna('MISSING', inplace=True)
        if train[col].nunique() > 2:
            lbl = LabelEncoder()
            train[col] = lbl.fit_transform(train[col].astype(str))
    
    # Handle numerical features
    num_features = [col for col in train.columns if col not in cat_features + ['TARGET', 'SK_ID_CURR']]
    train[num_features] = SimpleImputer(strategy='median').fit_transform(train[num_features])
    
    X = train.drop(['TARGET', 'SK_ID_CURR'], axis=1)
    y = train['TARGET']
    
    return X, y

def main():
    print("🔧 Starting Hyperparameter Optimization for Home Credit Ensemble")
    
    # Load data
    X, y = load_and_preprocess_data()
    print(f"Data shape: {X.shape}")
    
    # Initialize optimizer
    optimizer = HyperparameterOptimizer(X, y, n_folds=5, random_state=42)
    
    # Optimize each model
    models_to_optimize = ['catboost', 'lightgbm', 'xgboost', 'random_forest']
    best_params = {}
    
    for model_name in models_to_optimize:
        try:
            params, score = optimizer.optimize_model(model_name, n_trials=50)  # Reduced for demo
            best_params[model_name] = params
            print(f"\n✅ {model_name} optimization completed")
            print(f"   Best score: {score:.5f}")
            print(f"   Best params: {params}")
        except Exception as e:
            print(f"❌ Error optimizing {model_name}: {e}")
    
    # Save optimized parameters
    import json
    with open('optimized_parameters.json', 'w') as f:
        json.dump(best_params, f, indent=2)
    
    print("\n🎯 Hyperparameter optimization completed!")
    print("📁 Optimized parameters saved to 'optimized_parameters.json'")
    
    # Create optimized ensemble with best parameters
    print("\n🚀 Creating optimized ensemble...")
    
    # Train models with optimized parameters and get OOF predictions
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_predictions = {}
    
    for model_name, params in best_params.items():
        print(f"Training optimized {model_name}...")
        oof_pred = np.zeros(X.shape[0])
        
        for fold, (train_idx, valid_idx) in enumerate(kf.split(X, y)):
            X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
            y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
            
            if model_name == 'catboost':
                model = CatBoostClassifier(**params, random_seed=42, thread_count=-1, verbose=0)
                model.fit(X_train, y_train, eval_set=(X_valid, y_valid), use_best_model=True)
            elif model_name == 'lightgbm':
                model = LGBMClassifier(**params, random_state=42, n_jobs=-1, verbose=-1)
                model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], 
                         callbacks=[LGBMClassifier.early_stopping(100), LGBMClassifier.log_evaluation(0)])
            elif model_name == 'xgboost':
                model = XGBClassifier(**params, random_state=42, n_jobs=-1)
                model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], 
                         early_stopping_rounds=100, verbose=0)
            elif model_name == 'random_forest':
                model = RandomForestClassifier(**params, random_state=42, n_jobs=-1)
                model.fit(X_train, y_train)
            
            pred = model.predict_proba(X_valid)[:, 1]
            oof_pred[valid_idx] = pred
        
        oof_predictions[model_name] = oof_pred
        auc = roc_auc_score(y, oof_pred)
        print(f"  {model_name} OOF AUC: {auc:.5f}")
    
    # Optimize ensemble weights
    print("\n🎯 Optimizing ensemble weights...")
    best_weights, best_ensemble_score = optimize_ensemble_weights(oof_predictions, y, n_trials=500)
    
    print(f"\nOptimized ensemble weights:")
    for i, (model_name, weight) in enumerate(zip(oof_predictions.keys(), best_weights)):
        print(f"  {model_name}: {weight:.4f}")
    print(f"\nBest ensemble AUC: {best_ensemble_score:.5f}")
    
    # Save optimized weights
    weights_dict = {model: float(weight) for model, weight in zip(oof_predictions.keys(), best_weights)}
    with open('optimized_weights.json', 'w') as f:
        json.dump(weights_dict, f, indent=2)
    
    print("\n✅ Optimization process completed successfully!")
    print("📊 Use the optimized parameters and weights in your main ensemble script")

if __name__ == "__main__":
    main()