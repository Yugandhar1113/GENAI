import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.ensemble import StackingClassifier, VotingClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif
import optuna
import warnings
warnings.filterwarnings('ignore')

class AdvancedHomeCreditsEnsemble:
    def __init__(self, n_folds=7, random_state=42):
        self.n_folds = n_folds
        self.random_state = random_state
        self.scaler = RobustScaler()
        self.feature_selector = SelectKBest(f_classif, k='all')
        
    def advanced_feature_engineering(self, df, is_train=True):
        """Advanced feature engineering with domain knowledge"""
        
        # Credit to income ratios
        df['CREDIT_INCOME_RATIO'] = df['AMT_CREDIT'] / (df['AMT_INCOME_TOTAL'] + 1)
        df['ANNUITY_INCOME_RATIO'] = df['AMT_ANNUITY'] / (df['AMT_INCOME_TOTAL'] + 1)
        df['CREDIT_ANNUITY_RATIO'] = df['AMT_CREDIT'] / (df['AMT_ANNUITY'] + 1)
        
        # Age and employment features
        df['DAYS_BIRTH_ABS'] = abs(df['DAYS_BIRTH'])
        df['DAYS_EMPLOYED_ABS'] = abs(df['DAYS_EMPLOYED'])
        df['AGE_YEARS'] = df['DAYS_BIRTH_ABS'] / 365.25
        df['EMPLOYMENT_YEARS'] = df['DAYS_EMPLOYED_ABS'] / 365.25
        
        # Family and social features
        df['INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / (df['CNT_FAM_MEMBERS'] + 1)
        df['CHILDREN_RATIO'] = df['CNT_CHILDREN'] / (df['CNT_FAM_MEMBERS'] + 1)
        
        # External source combinations
        df['EXT_SOURCE_MEAN'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].mean(axis=1)
        df['EXT_SOURCE_STD'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].std(axis=1)
        df['EXT_SOURCE_PROD'] = df['EXT_SOURCE_1'] * df['EXT_SOURCE_2'] * df['EXT_SOURCE_3']
        
        # Polynomial features for external sources
        for i in range(1, 4):
            df[f'EXT_SOURCE_{i}_SQUARED'] = df[f'EXT_SOURCE_{i}'] ** 2
            df[f'EXT_SOURCE_{i}_CUBED'] = df[f'EXT_SOURCE_{i}'] ** 3
            
        # Document flags
        doc_cols = [col for col in df.columns if 'FLAG_DOCUMENT' in col]
        df['DOCUMENT_COUNT'] = df[doc_cols].sum(axis=1)
        
        # Region ratings
        region_cols = ['REGION_RATING_CLIENT', 'REGION_RATING_CLIENT_W_CITY']
        df['REGION_RATING_MEAN'] = df[region_cols].mean(axis=1)
        
        # Organization type risk encoding (based on domain knowledge)
        high_risk_orgs = ['Transport: type 3', 'Construction', 'Security Ministries']
        df['HIGH_RISK_ORG'] = df['ORGANIZATION_TYPE'].isin(high_risk_orgs).astype(int)
        
        return df
    
    def get_base_models_level1(self):
        """First level base models with diverse algorithms"""
        return {
            'catboost_1': CatBoostClassifier(
                iterations=2000, learning_rate=0.02, depth=8,
                l2_leaf_reg=3, border_count=128, eval_metric='AUC',
                early_stopping_rounds=200, random_seed=self.random_state,
                thread_count=-1, verbose=0
            ),
            'lightgbm_1': LGBMClassifier(
                n_estimators=2000, learning_rate=0.02, max_depth=8,
                num_leaves=64, reg_alpha=3, reg_lambda=3,
                feature_fraction=0.8, bagging_fraction=0.8,
                metric='auc', random_state=self.random_state, n_jobs=-1
            ),
            'xgboost_1': XGBClassifier(
                n_estimators=2000, learning_rate=0.02, max_depth=8,
                reg_alpha=3, reg_lambda=3, subsample=0.8,
                colsample_bytree=0.8, eval_metric='auc',
                random_state=self.random_state, n_jobs=-1
            ),
            'catboost_2': CatBoostClassifier(
                iterations=1500, learning_rate=0.03, depth=6,
                l2_leaf_reg=5, bootstrap_type='Bernoulli',
                subsample=0.8, eval_metric='AUC',
                early_stopping_rounds=150, random_seed=self.random_state+1,
                thread_count=-1, verbose=0
            ),
            'lightgbm_2': LGBMClassifier(
                n_estimators=1500, learning_rate=0.03, max_depth=6,
                num_leaves=32, reg_alpha=5, reg_lambda=5,
                feature_fraction=0.7, bagging_fraction=0.7,
                metric='auc', random_state=self.random_state+1, n_jobs=-1
            ),
            'random_forest': RandomForestClassifier(
                n_estimators=800, max_depth=8, max_features='sqrt',
                min_samples_split=20, min_samples_leaf=10,
                random_state=self.random_state, n_jobs=-1
            ),
            'extra_trees': ExtraTreesClassifier(
                n_estimators=800, max_depth=8, max_features='sqrt',
                min_samples_split=20, min_samples_leaf=10,
                random_state=self.random_state, n_jobs=-1
            ),
            'gradient_boost': GradientBoostingClassifier(
                n_estimators=500, learning_rate=0.05, max_depth=6,
                subsample=0.8, random_state=self.random_state
            )
        }
    
    def get_base_models_level2(self):
        """Second level base models with different configurations"""
        return {
            'catboost_l2': CatBoostClassifier(
                iterations=1000, learning_rate=0.05, depth=5,
                l2_leaf_reg=10, eval_metric='AUC',
                early_stopping_rounds=100, random_seed=self.random_state+10,
                thread_count=-1, verbose=0
            ),
            'lightgbm_l2': LGBMClassifier(
                n_estimators=1000, learning_rate=0.05, max_depth=5,
                num_leaves=16, reg_alpha=10, reg_lambda=10,
                metric='auc', random_state=self.random_state+10, n_jobs=-1
            ),
            'xgboost_l2': XGBClassifier(
                n_estimators=1000, learning_rate=0.05, max_depth=5,
                reg_alpha=10, reg_lambda=10, eval_metric='auc',
                random_state=self.random_state+10, n_jobs=-1
            )
        }
    
    def get_meta_models(self):
        """Meta models for final stacking"""
        return {
            'logistic': LogisticRegression(
                C=0.1, max_iter=2000, random_state=self.random_state
            ),
            'ridge': RidgeClassifier(
                alpha=1.0, random_state=self.random_state
            ),
            'mlp': MLPClassifier(
                hidden_layer_sizes=(100, 50), max_iter=500,
                random_state=self.random_state
            ),
            'svm': SVC(
                C=0.1, probability=True, random_state=self.random_state
            ),
            'naive_bayes': GaussianNB()
        }
    
    def train_level1_models(self, X, y, X_test):
        """Train first level models with cross-validation"""
        print("Training Level 1 Models...")
        
        kf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        
        oof_preds_l1 = {}
        test_preds_l1 = {}
        
        models = self.get_base_models_level1()
        
        for model_name, model in models.items():
            print(f"Training {model_name}...")
            
            oof_preds_l1[model_name] = np.zeros(X.shape[0])
            test_preds_l1[model_name] = np.zeros(X_test.shape[0])
            
            for fold, (train_idx, valid_idx) in enumerate(kf.split(X, y)):
                X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
                y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
                
                if 'catboost' in model_name:
                    model.fit(
                        X_train, y_train,
                        eval_set=(X_valid, y_valid),
                        use_best_model=True,
                        verbose=0
                    )
                elif 'lightgbm' in model_name:
                    model.fit(
                        X_train, y_train,
                        eval_set=[(X_valid, y_valid)],
                        callbacks=[LGBMClassifier().early_stopping(100)]
                    )
                elif 'xgboost' in model_name:
                    model.fit(
                        X_train, y_train,
                        eval_set=[(X_valid, y_valid)],
                        verbose=False
                    )
                else:
                    model.fit(X_train, y_train)
                
                oof_preds_l1[model_name][valid_idx] = model.predict_proba(X_valid)[:, 1]
                test_preds_l1[model_name] += model.predict_proba(X_test)[:, 1] / self.n_folds
                
                fold_auc = roc_auc_score(y_valid, oof_preds_l1[model_name][valid_idx])
                print(f"  Fold {fold + 1} AUC: {fold_auc:.5f}")
            
            overall_auc = roc_auc_score(y, oof_preds_l1[model_name])
            print(f"  Overall AUC: {overall_auc:.5f}\n")
        
        return oof_preds_l1, test_preds_l1
    
    def train_level2_models(self, stacked_X, y, stacked_test):
        """Train second level models"""
        print("Training Level 2 Models...")
        
        kf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state+100)
        
        oof_preds_l2 = {}
        test_preds_l2 = {}
        
        models = self.get_base_models_level2()
        
        for model_name, model in models.items():
            print(f"Training {model_name}...")
            
            oof_preds_l2[model_name] = np.zeros(stacked_X.shape[0])
            test_preds_l2[model_name] = np.zeros(stacked_test.shape[0])
            
            for fold, (train_idx, valid_idx) in enumerate(kf.split(stacked_X, y)):
                X_train, X_valid = stacked_X[train_idx], stacked_X[valid_idx]
                y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
                
                model.fit(X_train, y_train)
                
                oof_preds_l2[model_name][valid_idx] = model.predict_proba(X_valid)[:, 1]
                test_preds_l2[model_name] += model.predict_proba(stacked_test)[:, 1] / self.n_folds
            
            overall_auc = roc_auc_score(y, oof_preds_l2[model_name])
            print(f"  Overall AUC: {overall_auc:.5f}\n")
        
        return oof_preds_l2, test_preds_l2
    
    def train_meta_models(self, meta_X, y, meta_test):
        """Train meta models for final predictions"""
        print("Training Meta Models...")
        
        # Scale features for neural networks and SVM
        meta_X_scaled = self.scaler.fit_transform(meta_X)
        meta_test_scaled = self.scaler.transform(meta_test)
        
        meta_preds = {}
        models = self.get_meta_models()
        
        for model_name, model in models.items():
            print(f"Training {model_name}...")
            
            if model_name in ['mlp', 'svm']:
                model.fit(meta_X_scaled, y)
                meta_preds[model_name] = model.predict_proba(meta_test_scaled)[:, 1]
            else:
                model.fit(meta_X, y)
                if hasattr(model, 'predict_proba'):
                    meta_preds[model_name] = model.predict_proba(meta_test)[:, 1]
                else:
                    # For Ridge classifier
                    decision = model.decision_function(meta_test)
                    meta_preds[model_name] = 1 / (1 + np.exp(-decision))  # Sigmoid
        
        return meta_preds
    
    def optimize_blending_weights(self, predictions_dict, y_true):
        """Optimize blending weights using Optuna"""
        def objective(trial):
            weights = []
            for i, model_name in enumerate(predictions_dict.keys()):
                weight = trial.suggest_float(f'weight_{i}', 0.0, 1.0)
                weights.append(weight)
            
            # Normalize weights
            weights = np.array(weights)
            weights = weights / weights.sum()
            
            # Create blended prediction
            blended_pred = np.zeros_like(y_true, dtype=float)
            for i, (model_name, pred) in enumerate(predictions_dict.items()):
                blended_pred += weights[i] * pred
            
            return roc_auc_score(y_true, blended_pred)
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=200, show_progress_bar=False)
        
        best_weights = []
        for i in range(len(predictions_dict)):
            best_weights.append(study.best_params[f'weight_{i}'])
        
        best_weights = np.array(best_weights)
        best_weights = best_weights / best_weights.sum()
        
        return best_weights, study.best_value
    
    def create_voting_ensemble(self, X, y, X_test):
        """Create voting ensemble with optimized weights"""
        print("Creating Voting Ensemble...")
        
        # Select best performing models for voting
        voting_models = [
            ('catboost', CatBoostClassifier(
                iterations=1500, learning_rate=0.03, depth=7,
                eval_metric='AUC', random_seed=self.random_state, verbose=0
            )),
            ('lightgbm', LGBMClassifier(
                n_estimators=1500, learning_rate=0.03, max_depth=7,
                metric='auc', random_state=self.random_state, n_jobs=-1
            )),
            ('xgboost', XGBClassifier(
                n_estimators=1500, learning_rate=0.03, max_depth=7,
                eval_metric='auc', random_state=self.random_state, n_jobs=-1
            )),
            ('rf', RandomForestClassifier(
                n_estimators=500, max_depth=7, random_state=self.random_state, n_jobs=-1
            ))
        ]
        
        voting_clf = VotingClassifier(
            estimators=voting_models,
            voting='soft',
            weights=[0.3, 0.25, 0.25, 0.2]
        )
        
        voting_clf.fit(X, y)
        voting_pred = voting_clf.predict_proba(X_test)[:, 1]
        
        return voting_pred
    
    def fit_predict(self, train_path, test_path):
        """Main training and prediction pipeline"""
        print("Loading and preprocessing data...")
        
        # Load data
        train = pd.read_csv(train_path)
        test = pd.read_csv(test_path)
        test_ids = test['SK_ID_CURR']
        
        # Feature engineering
        train = self.advanced_feature_engineering(train, is_train=True)
        test = self.advanced_feature_engineering(test, is_train=False)
        
        # Combine and preprocess
        train['is_train'] = 1
        test['is_train'] = 0
        test['TARGET'] = np.nan
        data = pd.concat([train, test], axis=0, ignore_index=True)
        
        # Handle categorical features
        cat_features = data.select_dtypes(include=['object']).columns.tolist()
        if 'TARGET' in cat_features:
            cat_features.remove('TARGET')
        if 'SK_ID_CURR' in cat_features:
            cat_features.remove('SK_ID_CURR')
        
        for col in cat_features:
            data[col].fillna('MISSING', inplace=True)
            if data[col].nunique() > 2:
                lbl = LabelEncoder()
                data[col] = lbl.fit_transform(data[col].astype(str))
        
        # Handle numerical features
        num_features = [col for col in data.columns 
                       if col not in cat_features + ['TARGET', 'SK_ID_CURR', 'is_train']]
        data[num_features] = SimpleImputer(strategy='median').fit_transform(data[num_features])
        
        # Split back
        train = data[data['is_train'] == 1].drop(columns=['is_train'])
        test = data[data['is_train'] == 0].drop(columns=['is_train', 'TARGET'])
        
        X = train.drop(columns=['TARGET', 'SK_ID_CURR'])
        y = train['TARGET']
        X_test = test.drop(columns=['SK_ID_CURR'])
        
        print(f"Training set shape: {X.shape}")
        print(f"Test set shape: {X_test.shape}")
        
        # Level 1: Base models
        oof_preds_l1, test_preds_l1 = self.train_level1_models(X, y, X_test)
        
        # Create stacked features for level 2
        stacked_X_l1 = np.column_stack([oof_preds_l1[model] for model in oof_preds_l1])
        stacked_test_l1 = np.column_stack([test_preds_l1[model] for model in test_preds_l1])
        
        # Level 2: Second level models
        oof_preds_l2, test_preds_l2 = self.train_level2_models(stacked_X_l1, y, stacked_test_l1)
        
        # Create meta features (combine level 1 and level 2)
        meta_X = np.column_stack([
            stacked_X_l1,
            np.column_stack([oof_preds_l2[model] for model in oof_preds_l2])
        ])
        meta_test = np.column_stack([
            stacked_test_l1,
            np.column_stack([test_preds_l2[model] for model in test_preds_l2])
        ])
        
        # Meta models
        meta_preds = self.train_meta_models(meta_X, y, meta_test)
        
        # Voting ensemble
        voting_pred = self.create_voting_ensemble(X, y, X_test)
        
        # Combine all predictions for final blending
        all_predictions = {**test_preds_l1, **test_preds_l2, **meta_preds, 'voting': voting_pred}
        all_oof_predictions = {**oof_preds_l1, **oof_preds_l2}
        
        # Add voting OOF predictions
        kf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        voting_oof = np.zeros(X.shape[0])
        
        for fold, (train_idx, valid_idx) in enumerate(kf.split(X, y)):
            X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
            y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
            
            voting_models = [
                ('catboost', CatBoostClassifier(
                    iterations=1500, learning_rate=0.03, depth=7,
                    eval_metric='AUC', random_seed=self.random_state, verbose=0
                )),
                ('lightgbm', LGBMClassifier(
                    n_estimators=1500, learning_rate=0.03, max_depth=7,
                    metric='auc', random_state=self.random_state, n_jobs=-1
                )),
                ('xgboost', XGBClassifier(
                    n_estimators=1500, learning_rate=0.03, max_depth=7,
                    eval_metric='auc', random_state=self.random_state, n_jobs=-1
                )),
                ('rf', RandomForestClassifier(
                    n_estimators=500, max_depth=7, random_state=self.random_state, n_jobs=-1
                ))
            ]
            
            voting_clf = VotingClassifier(estimators=voting_models, voting='soft', weights=[0.3, 0.25, 0.25, 0.2])
            voting_clf.fit(X_train, y_train)
            voting_oof[valid_idx] = voting_clf.predict_proba(X_valid)[:, 1]
        
        all_oof_predictions['voting'] = voting_oof
        
        # Optimize blending weights
        print("Optimizing blending weights...")
        best_weights, best_score = self.optimize_blending_weights(all_oof_predictions, y)
        
        print(f"Best blending score: {best_score:.5f}")
        print("Best weights:")
        for i, (model_name, weight) in enumerate(zip(all_predictions.keys(), best_weights)):
            print(f"  {model_name}: {weight:.4f}")
        
        # Create final blended prediction
        final_pred = np.zeros(X_test.shape[0])
        for i, (model_name, pred) in enumerate(all_predictions.items()):
            final_pred += best_weights[i] * pred
        
        return final_pred, test_ids, best_score

# Usage
if __name__ == "__main__":
    ensemble = AdvancedHomeCreditsEnsemble(n_folds=7, random_state=42)
    
    # Assuming data files are in the correct location
    train_path = "../input/home-credit-default-risk/application_train.csv"
    test_path = "../input/home-credit-default-risk/application_test.csv"
    
    try:
        final_predictions, test_ids, cv_score = ensemble.fit_predict(train_path, test_path)
        
        # Create submission
        submission = pd.DataFrame({
            'SK_ID_CURR': test_ids,
            'TARGET': final_predictions
        })
        
        submission.to_csv("advanced_ensemble_submission.csv", index=False)
        print(f"\n✅ Advanced ensemble submission saved with CV score: {cv_score:.5f}")
        print("Submission file: advanced_ensemble_submission.csv")
        
    except FileNotFoundError:
        print("Data files not found. Please ensure the data files are in the correct path.")
        print("Expected paths:")
        print("  - ../input/home-credit-default-risk/application_train.csv")
        print("  - ../input/home-credit-default-risk/application_test.csv")