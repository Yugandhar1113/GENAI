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
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif
import optuna
import warnings
warnings.filterwarnings('ignore')

class AdvancedEnsembleClassifier:
    def __init__(self, n_folds=7, random_state=42):
        self.n_folds = n_folds
        self.random_state = random_state
        self.base_models = {}
        self.meta_models = {}
        self.oof_predictions = {}
        self.test_predictions = {}
        self.feature_importances = pd.DataFrame()
        
    def get_base_models(self):
        """Define base models with optimized hyperparameters"""
        return {
            'catboost_1': CatBoostClassifier(
                iterations=2000, learning_rate=0.02, depth=8,
                l2_leaf_reg=3, bagging_temperature=0.2,
                eval_metric='AUC', early_stopping_rounds=150,
                random_seed=self.random_state, thread_count=-1, verbose=0
            ),
            'catboost_2': CatBoostClassifier(
                iterations=1800, learning_rate=0.025, depth=6,
                l2_leaf_reg=7, bagging_temperature=0.5,
                eval_metric='AUC', early_stopping_rounds=150,
                random_seed=self.random_state + 1, thread_count=-1, verbose=0
            ),
            'lightgbm_1': LGBMClassifier(
                n_estimators=2500, learning_rate=0.02, max_depth=8,
                reg_alpha=3, reg_lambda=3, min_child_samples=50,
                subsample=0.8, colsample_bytree=0.8,
                metric='auc', random_state=self.random_state, n_jobs=-1, verbose=-1
            ),
            'lightgbm_2': LGBMClassifier(
                n_estimators=2200, learning_rate=0.025, max_depth=6,
                reg_alpha=5, reg_lambda=5, min_child_samples=30,
                subsample=0.85, colsample_bytree=0.85,
                metric='auc', random_state=self.random_state + 2, n_jobs=-1, verbose=-1
            ),
            'xgboost_1': XGBClassifier(
                n_estimators=2000, learning_rate=0.02, max_depth=7,
                reg_alpha=3, reg_lambda=3, subsample=0.8,
                colsample_bytree=0.8, eval_metric='auc',
                random_state=self.random_state, n_jobs=-1, tree_method='hist'
            ),
            'xgboost_2': XGBClassifier(
                n_estimators=1800, learning_rate=0.025, max_depth=6,
                reg_alpha=5, reg_lambda=5, subsample=0.85,
                colsample_bytree=0.85, eval_metric='auc',
                random_state=self.random_state + 3, n_jobs=-1, tree_method='hist'
            ),
            'random_forest': RandomForestClassifier(
                n_estimators=800, max_depth=8, max_features='sqrt',
                min_samples_split=10, min_samples_leaf=5,
                random_state=self.random_state, n_jobs=-1
            ),
            'extra_trees': ExtraTreesClassifier(
                n_estimators=800, max_depth=8, max_features='sqrt',
                min_samples_split=10, min_samples_leaf=5,
                random_state=self.random_state, n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=500, learning_rate=0.05, max_depth=6,
                subsample=0.8, random_state=self.random_state
            )
        }
    
    def get_meta_models(self):
        """Define meta models for stacking"""
        return {
            'logistic_regression': LogisticRegression(C=0.05, max_iter=2000, random_state=self.random_state),
            'ridge': RidgeClassifier(alpha=1.0, random_state=self.random_state),
            'catboost_meta': CatBoostClassifier(
                iterations=800, learning_rate=0.01, depth=4,
                eval_metric='AUC', random_seed=self.random_state, verbose=0
            ),
            'xgboost_meta': XGBClassifier(
                n_estimators=500, learning_rate=0.01, max_depth=4,
                eval_metric='auc', random_state=self.random_state
            ),
            'mlp': MLPClassifier(
                hidden_layer_sizes=(100, 50), learning_rate_init=0.001,
                max_iter=500, random_state=self.random_state
            )
        }
    
    def advanced_feature_engineering(self, df):
        """Advanced feature engineering for Home Credit dataset"""
        print("Performing advanced feature engineering...")
        
        # Create new features
        df['CREDIT_INCOME_PERCENT'] = df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL']
        df['ANNUITY_INCOME_PERCENT'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
        df['CREDIT_TERM'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']
        df['DAYS_EMPLOYED_PERCENT'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
        
        # Age-related features
        df['AGE_YEARS'] = -df['DAYS_BIRTH'] / 365
        df['EMPLOYED_YEARS'] = -df['DAYS_EMPLOYED'] / 365
        df['REGISTRATION_YEARS'] = -df['DAYS_REGISTRATION'] / 365
        df['ID_PUBLISH_YEARS'] = -df['DAYS_ID_PUBLISH'] / 365
        
        # Income per family member
        df['INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS']
        
        # Credit to goods ratio
        df['CREDIT_GOODS_RATIO'] = df['AMT_CREDIT'] / df['AMT_GOODS_PRICE']
        
        # External source combinations
        ext_sources = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']
        df['EXT_SOURCES_MEAN'] = df[ext_sources].mean(axis=1)
        df['EXT_SOURCES_STD'] = df[ext_sources].std(axis=1)
        df['EXT_SOURCES_MAX'] = df[ext_sources].max(axis=1)
        df['EXT_SOURCES_MIN'] = df[ext_sources].min(axis=1)
        
        # Polynomial features for important variables
        df['EXT_SOURCE_1_2'] = df['EXT_SOURCE_1'] * df['EXT_SOURCE_2']
        df['EXT_SOURCE_2_3'] = df['EXT_SOURCE_2'] * df['EXT_SOURCE_3']
        df['EXT_SOURCE_1_3'] = df['EXT_SOURCE_1'] * df['EXT_SOURCE_3']
        
        # Flag features
        df['FLAG_DOCUMENT_SUM'] = df[[col for col in df.columns if 'FLAG_DOCUMENT' in col]].sum(axis=1)
        df['AMT_REQ_CREDIT_BUREAU_SUM'] = df[[col for col in df.columns if 'AMT_REQ_CREDIT_BUREAU' in col]].sum(axis=1)
        
        return df
    
    def train_base_models(self, X, y, X_test, cat_features=None):
        """Train base models using K-fold cross-validation"""
        print(f"Training base models with {self.n_folds}-fold CV...")
        
        kf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        models = self.get_base_models()
        
        # Initialize storage
        self.oof_predictions = {model_name: np.zeros(X.shape[0]) for model_name in models}
        self.test_predictions = {model_name: np.zeros(X_test.shape[0]) for model_name in models}
        
        for fold, (train_idx, valid_idx) in enumerate(kf.split(X, y)):
            print(f"\nFold {fold + 1}/{self.n_folds}")
            X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
            y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
            
            for model_name, model in models.items():
                print(f"  Training {model_name}...")
                
                if 'catboost' in model_name:
                    model.fit(
                        X_train, y_train,
                        eval_set=(X_valid, y_valid),
                        cat_features=cat_features,
                        use_best_model=True,
                        verbose=0
                    )
                elif 'lightgbm' in model_name:
                    model.fit(
                        X_train, y_train,
                        eval_set=[(X_valid, y_valid)],
                        callbacks=[
                            LGBMClassifier.early_stopping(150),
                            LGBMClassifier.log_evaluation(0)
                        ]
                    )
                elif 'xgboost' in model_name:
                    model.fit(
                        X_train, y_train,
                        eval_set=[(X_valid, y_valid)],
                        early_stopping_rounds=150,
                        verbose=0
                    )
                else:
                    model.fit(X_train, y_train)
                
                # Store OOF predictions
                oof_pred = model.predict_proba(X_valid)[:, 1]
                self.oof_predictions[model_name][valid_idx] = oof_pred
                
                # Accumulate test predictions
                test_pred = model.predict_proba(X_test)[:, 1]
                self.test_predictions[model_name] += test_pred / self.n_folds
                
                # Calculate fold AUC
                fold_auc = roc_auc_score(y_valid, oof_pred)
                print(f"    {model_name} Fold {fold + 1} AUC: {fold_auc:.5f}")
        
        # Print base model performance
        print("\nBase Model OOF Performance:")
        for model_name in self.oof_predictions:
            auc = roc_auc_score(y, self.oof_predictions[model_name])
            print(f"{model_name:>20}: {auc:.5f}")
    
    def train_meta_models(self, y):
        """Train meta models for stacking"""
        print("\nTraining meta models...")
        
        # Create stacked features
        stacked_X = np.column_stack([self.oof_predictions[model] for model in self.oof_predictions])
        stacked_test = np.column_stack([self.test_predictions[model] for model in self.test_predictions])
        
        # Scale features for neural network
        scaler = StandardScaler()
        stacked_X_scaled = scaler.fit_transform(stacked_X)
        stacked_test_scaled = scaler.transform(stacked_test)
        
        meta_models = self.get_meta_models()
        meta_predictions = {}
        
        for meta_name, meta_model in meta_models.items():
            print(f"  Training {meta_name}...")
            
            if meta_name == 'mlp':
                meta_model.fit(stacked_X_scaled, y)
                meta_pred = meta_model.predict_proba(stacked_test_scaled)[:, 1]
            else:
                meta_model.fit(stacked_X, y)
                meta_pred = meta_model.predict_proba(stacked_test)[:, 1]
            
            meta_predictions[meta_name] = meta_pred
            
            # Calculate meta model performance on stacked OOF
            if meta_name == 'mlp':
                oof_meta_pred = meta_model.predict_proba(stacked_X_scaled)[:, 1]
            else:
                oof_meta_pred = meta_model.predict_proba(stacked_X)[:, 1]
            
            meta_auc = roc_auc_score(y, oof_meta_pred)
            print(f"    {meta_name} Stacked AUC: {meta_auc:.5f}")
        
        return meta_predictions, stacked_test
    
    def ensemble_predictions(self, meta_predictions, X, y, X_test):
        """Create final ensemble using multiple approaches"""
        print("\nCreating ensemble predictions...")
        
        # 1. Weighted average of meta models
        meta_weights = {
            'catboost_meta': 0.25,
            'xgboost_meta': 0.25,
            'logistic_regression': 0.20,
            'ridge': 0.15,
            'mlp': 0.15
        }
        
        weighted_meta_pred = sum(meta_weights[name] * pred for name, pred in meta_predictions.items())
        
        # 2. Voting classifier approach
        base_models = self.get_base_models()
        
        # Retrain on full data for voting
        for model_name, model in base_models.items():
            print(f"  Retraining {model_name} on full data...")
            model.fit(X, y)
        
        voting_clf = VotingClassifier(
            estimators=[(name, model) for name, model in base_models.items()],
            voting='soft',
            weights=[3, 2.5, 3, 2.5, 2.8, 2.3, 1.5, 1.5, 1.2]  # Optimized weights
        )
        voting_clf.fit(X, y)
        voting_pred = voting_clf.predict_proba(X_test)[:, 1]
        
        # 3. Bayesian model averaging
        base_weights = np.array([3, 2.5, 3, 2.5, 2.8, 2.3, 1.5, 1.5, 1.2])
        base_weights = base_weights / base_weights.sum()
        
        bayesian_pred = sum(w * pred for w, pred in zip(base_weights, self.test_predictions.values()))
        
        # 4. Final ensemble combination
        ensemble_weights = {
            'weighted_meta': 0.5,
            'voting': 0.25,
            'bayesian': 0.25
        }
        
        final_prediction = (
            ensemble_weights['weighted_meta'] * weighted_meta_pred +
            ensemble_weights['voting'] * voting_pred +
            ensemble_weights['bayesian'] * bayesian_pred
        )
        
        return final_prediction, {
            'weighted_meta': weighted_meta_pred,
            'voting': voting_pred,
            'bayesian': bayesian_pred,
            'final': final_prediction
        }

def main():
    print("🚀 Starting Advanced Ensemble Solution for Home Credit Default Risk")
    
    # Load data
    print("Loading data...")
    try:
        train = pd.read_csv("../input/home-credit-default-risk/application_train.csv")
        test = pd.read_csv("../input/home-credit-default-risk/application_test.csv")
    except FileNotFoundError:
        print("Data files not found. Please ensure the dataset is available.")
        return
    
    test_ids = test['SK_ID_CURR']
    
    # Initialize ensemble classifier
    ensemble = AdvancedEnsembleClassifier(n_folds=7, random_state=42)
    
    # Feature engineering
    train = ensemble.advanced_feature_engineering(train)
    test = ensemble.advanced_feature_engineering(test)
    
    # Combine and preprocess data
    train['is_train'] = 1
    test['is_train'] = 0
    test['TARGET'] = np.nan
    data = pd.concat([train, test], axis=0, ignore_index=True)
    
    # Handle categorical features
    cat_features = data.select_dtypes(include=['object']).columns.tolist()
    cat_indices = []
    
    for i, col in enumerate(data.columns):
        if col in cat_features:
            data[col].fillna('MISSING', inplace=True)
            if data[col].nunique() > 2:
                lbl = LabelEncoder()
                data[col] = lbl.fit_transform(data[col].astype(str))
            cat_indices.append(i)
    
    # Handle numerical features
    num_features = [col for col in data.columns if col not in cat_features + ['TARGET', 'SK_ID_CURR', 'is_train']]
    imputer = SimpleImputer(strategy='median')
    data[num_features] = imputer.fit_transform(data[num_features])
    
    # Feature selection
    print("Performing feature selection...")
    train_data = data[data['is_train'] == 1]
    X_full = train_data.drop(columns=['TARGET', 'SK_ID_CURR', 'is_train'])
    y = train_data['TARGET']
    
    # Select top features
    selector = SelectKBest(score_func=f_classif, k=min(500, X_full.shape[1]))
    X_selected = selector.fit_transform(X_full, y)
    selected_features = X_full.columns[selector.get_support()]
    
    # Split back into train and test
    train_final = data[data['is_train'] == 1]
    test_final = data[data['is_train'] == 0]
    
    X = train_final[selected_features]
    y = train_final['TARGET']
    X_test = test_final[selected_features]
    
    print(f"Final feature set: {X.shape[1]} features")
    
    # Train ensemble
    ensemble.train_base_models(X, y, X_test, cat_features=None)
    meta_predictions, stacked_test = ensemble.train_meta_models(y)
    final_prediction, all_predictions = ensemble.ensemble_predictions(meta_predictions, X, y, X_test)
    
    # Generate submissions
    print("\n📊 Generating submission files...")
    
    submissions = {
        'stacked_ensemble': all_predictions['weighted_meta'],
        'voting_ensemble': all_predictions['voting'],
        'bayesian_ensemble': all_predictions['bayesian'],
        'final_ensemble': all_predictions['final']
    }
    
    for name, pred in submissions.items():
        submission = pd.DataFrame({
            'SK_ID_CURR': test_ids,
            'TARGET': pred
        })
        filename = f"{name}_submission.csv"
        submission.to_csv(filename, index=False)
        print(f"✅ {filename} saved")
    
    print("\n🎯 Ensemble training completed successfully!")
    print("📈 Multiple submission files generated for maximum performance")

if __name__ == "__main__":
    main()