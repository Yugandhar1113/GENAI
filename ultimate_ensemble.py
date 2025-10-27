import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
import warnings
warnings.filterwarnings('ignore')

# Set random seeds
np.random.seed(42)
import random
random.seed(42)

class UltimateEnsemble:
    """Ultimate ensemble model with advanced techniques"""
    
    def __init__(self, n_folds=5, feature_selection=True, n_features=200):
        self.n_folds = n_folds
        self.feature_selection = feature_selection
        self.n_features = n_features
        self.base_models = {}
        self.meta_models = {}
        self.selected_features = None
        self.scaler = StandardScaler()
        
    def advanced_feature_engineering(self, df):
        """Advanced feature engineering with domain knowledge"""
        
        print("🔧 Applying advanced feature engineering...")
        
        # Age features
        df['DAYS_BIRTH'] = abs(df['DAYS_BIRTH'])
        df['AGE'] = df['DAYS_BIRTH'] / 365
        df['AGE_GROUP'] = pd.cut(df['AGE'], bins=[0, 25, 35, 45, 55, 100], labels=[0, 1, 2, 3, 4])
        
        # Employment features
        df['DAYS_EMPLOYED'] = abs(df['DAYS_EMPLOYED'])
        df['EMPLOYMENT_YEARS'] = df['DAYS_EMPLOYED'] / 365
        df['EMPLOYMENT_YEARS'] = df['EMPLOYMENT_YEARS'].clip(upper=50)
        
        # Income features
        df['INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS']
        df['INCOME_CREDIT_RATIO'] = df['AMT_INCOME_TOTAL'] / df['AMT_CREDIT']
        df['INCOME_ANNUITY_RATIO'] = df['AMT_INCOME_TOTAL'] / df['AMT_ANNUITY']
        df['CREDIT_ANNUITY_RATIO'] = df['AMT_CREDIT'] / df['AMT_ANNUITY']
        
        # Payment features
        df['PAYMENT_RATE'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']
        
        # Document features
        doc_cols = [col for col in df.columns if 'FLAG_DOC' in col]
        if doc_cols:
            df['DOCS_SUBMITTED'] = df[doc_cols].sum(axis=1)
            df['DOCS_SUBMITTED_RATIO'] = df['DOCS_SUBMITTED'] / len(doc_cols)
        
        # Contact features
        contact_cols = [col for col in df.columns if 'FLAG_' in col and 'DOC' not in col]
        if contact_cols:
            df['CONTACT_FLAGS'] = df[contact_cols].sum(axis=1)
        
        # External source features
        ext_cols = [col for col in df.columns if 'EXT_SOURCE' in col]
        if ext_cols:
            df['EXT_SOURCE_MEAN'] = df[ext_cols].mean(axis=1)
            df['EXT_SOURCE_STD'] = df[ext_cols].std(axis=1)
            df['EXT_SOURCE_SUM'] = df[ext_cols].sum(axis=1)
            df['EXT_SOURCE_MAX'] = df[ext_cols].max(axis=1)
            df['EXT_SOURCE_MIN'] = df[ext_cols].min(axis=1)
            df['EXT_SOURCE_RANGE'] = df['EXT_SOURCE_MAX'] - df['EXT_SOURCE_MIN']
        
        # Bureau features (if available)
        bureau_cols = [col for col in df.columns if 'BUREAU' in col]
        if bureau_cols:
            df['BUREAU_COUNT'] = df[bureau_cols].count(axis=1)
            df['BUREAU_MEAN'] = df[bureau_cols].mean(axis=1)
            df['BUREAU_STD'] = df[bureau_cols].std(axis=1)
        
        # Previous application features (if available)
        prev_cols = [col for col in df.columns if 'PREV' in col]
        if prev_cols:
            df['PREV_COUNT'] = df[prev_cols].count(axis=1)
            df['PREV_MEAN'] = df[prev_cols].mean(axis=1)
            df['PREV_STD'] = df[prev_cols].std(axis=1)
        
        # POS cash features (if available)
        pos_cols = [col for col in df.columns if 'POS' in col]
        if pos_cols:
            df['POS_COUNT'] = df[pos_cols].count(axis=1)
            df['POS_MEAN'] = df[pos_cols].mean(axis=1)
            df['POS_STD'] = df[pos_cols].std(axis=1)
        
        # Installments features (if available)
        install_cols = [col for col in df.columns if 'INSTAL' in col]
        if install_cols:
            df['INSTAL_COUNT'] = df[install_cols].count(axis=1)
            df['INSTAL_MEAN'] = df[install_cols].mean(axis=1)
            df['INSTAL_STD'] = df[install_cols].std(axis=1)
        
        # Credit card features (if available)
        cc_cols = [col for col in df.columns if 'CC_' in col]
        if cc_cols:
            df['CC_COUNT'] = df[cc_cols].count(axis=1)
            df['CC_MEAN'] = df[cc_cols].mean(axis=1)
            df['CC_STD'] = df[cc_cols].std(axis=1)
        
        # Additional engineered features
        df['CREDIT_INCOME_RATIO'] = df['AMT_CREDIT'] / (df['AMT_INCOME_TOTAL'] + 1)
        df['ANNUITY_INCOME_RATIO'] = df['AMT_ANNUITY'] / (df['AMT_INCOME_TOTAL'] + 1)
        df['CREDIT_TERM'] = df['AMT_CREDIT'] / (df['AMT_ANNUITY'] + 1)
        df['DAYS_EMPLOYED_PERCENT'] = df['DAYS_EMPLOYED'] / (df['DAYS_BIRTH'] + 1)
        
        return df
    
    def get_base_models(self):
        """Get optimized base models"""
        return {
            'catboost_1': CatBoostClassifier(
                iterations=2000, learning_rate=0.02, depth=8,
                l2_leaf_reg=3, eval_metric='AUC', early_stopping_rounds=150,
                random_seed=42, thread_count=-1, verbose=0,
                bootstrap_type='Bernoulli', subsample=0.8
            ),
            'catboost_2': CatBoostClassifier(
                iterations=1500, learning_rate=0.03, depth=6,
                l2_leaf_reg=5, eval_metric='AUC', early_stopping_rounds=100,
                random_seed=123, thread_count=-1, verbose=0,
                bootstrap_type='MVS', subsample=0.9
            ),
            'lightgbm_1': LGBMClassifier(
                n_estimators=2000, learning_rate=0.02, max_depth=8,
                reg_alpha=3, reg_lambda=3, metric='auc',
                random_state=42, n_jobs=-1, subsample=0.8, colsample_bytree=0.8,
                min_child_samples=20, min_child_weight=1e-3
            ),
            'lightgbm_2': LGBMClassifier(
                n_estimators=1500, learning_rate=0.03, max_depth=6,
                reg_alpha=5, reg_lambda=5, metric='auc',
                random_state=123, n_jobs=-1, subsample=0.9, colsample_bytree=0.9,
                min_child_samples=30, min_child_weight=1e-2
            ),
            'xgboost_1': XGBClassifier(
                n_estimators=2000, learning_rate=0.02, max_depth=8,
                reg_alpha=3, reg_lambda=3, eval_metric='auc',
                random_state=42, n_jobs=-1, subsample=0.8, colsample_bytree=0.8,
                min_child_weight=1, gamma=0.1
            ),
            'xgboost_2': XGBClassifier(
                n_estimators=1500, learning_rate=0.03, max_depth=6,
                reg_alpha=5, reg_lambda=5, eval_metric='auc',
                random_state=123, n_jobs=-1, subsample=0.9, colsample_bytree=0.9,
                min_child_weight=3, gamma=0.2
            ),
            'random_forest': RandomForestClassifier(
                n_estimators=1000, max_depth=10, max_features='sqrt',
                random_state=42, n_jobs=-1, min_samples_split=10,
                min_samples_leaf=5, bootstrap=True
            ),
            'extra_trees': ExtraTreesClassifier(
                n_estimators=1000, max_depth=10, max_features='sqrt',
                random_state=42, n_jobs=-1, min_samples_split=10,
                min_samples_leaf=5, bootstrap=True
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=1000, learning_rate=0.02, max_depth=6,
                random_state=42, subsample=0.8, min_samples_split=10,
                min_samples_leaf=5
            )
        }
    
    def get_meta_models(self):
        """Get meta models for stacking"""
        return {
            'logistic_regression': LogisticRegression(C=0.1, max_iter=2000, random_state=42),
            'ridge_classifier': RidgeClassifier(alpha=1.0, random_state=42),
            'catboost_meta': CatBoostClassifier(
                iterations=1000, learning_rate=0.01, depth=4,
                eval_metric='AUC', random_seed=42, verbose=0
            ),
            'lightgbm_meta': LGBMClassifier(
                n_estimators=1000, learning_rate=0.01, max_depth=4,
                metric='auc', random_state=42, n_jobs=-1
            ),
            'xgboost_meta': XGBClassifier(
                n_estimators=1000, learning_rate=0.01, max_depth=4,
                eval_metric='auc', random_state=42, n_jobs=-1
            ),
            'mlp_classifier': MLPClassifier(
                hidden_layer_sizes=(100, 50), max_iter=1000,
                random_state=42, early_stopping=True, validation_fraction=0.1
            )
        }
    
    def select_features(self, X, y):
        """Select best features using multiple methods"""
        if not self.feature_selection:
            return X.columns.tolist()
        
        print(f"🎯 Selecting top {self.n_features} features...")
        
        # Method 1: F-test
        f_selector = SelectKBest(score_func=f_classif, k=self.n_features)
        f_selector.fit(X, y)
        f_features = X.columns[f_selector.get_support()].tolist()
        
        # Method 2: Mutual Information
        mi_selector = SelectKBest(score_func=mutual_info_classif, k=self.n_features)
        mi_selector.fit(X, y)
        mi_features = X.columns[mi_selector.get_support()].tolist()
        
        # Method 3: Random Forest importance
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X, y)
        rf_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        rf_features = rf_importance.head(self.n_features)['feature'].tolist()
        
        # Combine features (union of all methods)
        all_features = list(set(f_features + mi_features + rf_features))
        
        # If too many features, take the most common ones
        if len(all_features) > self.n_features:
            feature_counts = {}
            for feature in all_features:
                feature_counts[feature] = sum([
                    feature in f_features,
                    feature in mi_features,
                    feature in rf_features
                ])
            
            # Sort by frequency and take top features
            sorted_features = sorted(feature_counts.items(), key=lambda x: x[1], reverse=True)
            selected_features = [f[0] for f in sorted_features[:self.n_features]]
        else:
            selected_features = all_features
        
        print(f"✅ Selected {len(selected_features)} features")
        return selected_features
    
    def train_models(self, X, y, X_test):
        """Train all models with cross-validation"""
        
        # Feature selection
        if self.feature_selection:
            self.selected_features = self.select_features(X, y)
            X = X[self.selected_features]
            X_test = X_test[self.selected_features]
        
        print(f"📊 Training data shape: {X.shape}")
        print(f"📊 Test data shape: {X_test.shape}")
        
        # Get models
        base_models = self.get_base_models()
        meta_models = self.get_meta_models()
        
        # Prepare storage
        oof_predictions = {name: np.zeros(X.shape[0]) for name in base_models}
        test_predictions = {name: np.zeros(X_test.shape[0]) for name in base_models}
        model_scores = {name: [] for name in base_models}
        
        # Stratified K-Fold
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=42)
        
        print(f"\n🎯 Training {len(base_models)} base models with {self.n_folds}-fold CV...")
        
        # Train base models
        for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y)):
            print(f"\n📁 Fold {fold + 1}/{self.n_folds}")
            X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
            y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
            
            for model_name, model in base_models.items():
                print(f"  🚀 Training {model_name}...")
                
                # Train model
                if 'catboost' in model_name:
                    model.fit(
                        X_train, y_train,
                        eval_set=(X_valid, y_valid),
                        use_best_model=True,
                        verbose=0
                    )
                else:
                    model.fit(X_train, y_train)
                
                # Store predictions
                oof_predictions[model_name][valid_idx] = model.predict_proba(X_valid)[:, 1]
                test_predictions[model_name] += model.predict_proba(X_test)[:, 1] / self.n_folds
                
                # Calculate score
                fold_auc = roc_auc_score(y_valid, oof_predictions[model_name][valid_idx])
                model_scores[model_name].append(fold_auc)
                print(f"    ✅ {model_name} Fold {fold + 1} AUC: {fold_auc:.5f}")
        
        # Evaluate base models
        print("\n📈 Base Model Performance:")
        print("-" * 60)
        for model_name in base_models:
            scores = model_scores[model_name]
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            print(f"{model_name:>15}: {mean_score:.5f} ± {std_score:.5f}")
        
        # Create stacked dataset
        print("\n🔗 Creating stacked dataset...")
        stacked_X = np.column_stack([oof_predictions[model] for model in base_models])
        stacked_test = np.column_stack([test_predictions[model] for model in base_models])
        
        # Scale features for meta-models
        stacked_X_scaled = self.scaler.fit_transform(stacked_X)
        stacked_test_scaled = self.scaler.transform(stacked_test)
        
        # Train meta-models
        print(f"\n🎯 Training {len(meta_models)} meta-models...")
        meta_predictions = {}
        meta_scores = {}
        
        for meta_name, meta_model in meta_models.items():
            print(f"  🚀 Training {meta_name}...")
            
            # Use scaled features for certain models
            if meta_name in ['logistic_regression', 'ridge_classifier', 'mlp_classifier']:
                X_meta = stacked_X_scaled
                test_meta = stacked_test_scaled
            else:
                X_meta = stacked_X
                test_meta = stacked_test
            
            # Train meta-model
            meta_model.fit(X_meta, y)
            
            # Get predictions
            if hasattr(meta_model, 'predict_proba'):
                meta_pred = meta_model.predict_proba(test_meta)[:, 1]
            else:
                meta_pred = meta_model.predict(test_meta)
            
            meta_predictions[meta_name] = meta_pred
            
            # Calculate score
            meta_score = roc_auc_score(y, meta_model.predict(X_meta))
            meta_scores[meta_name] = meta_score
            print(f"    ✅ {meta_name} Meta AUC: {meta_score:.5f}")
        
        # Advanced blending
        print("\n🎨 Implementing advanced blending...")
        
        # Strategy 1: Weighted average based on performance
        weights = np.array([meta_scores[name] for name in meta_predictions])
        weights = weights / weights.sum()
        weighted_blend = sum(weights[i] * pred for i, (name, pred) in enumerate(meta_predictions.items()))
        
        # Strategy 2: Simple average
        simple_blend = np.mean(list(meta_predictions.values()), axis=0)
        
        # Strategy 3: Geometric mean
        geometric_blend = np.exp(np.mean([np.log(pred + 1e-15) for pred in meta_predictions.values()], axis=0))
        
        # Strategy 4: Rank averaging
        rank_blend = np.mean([np.argsort(np.argsort(pred)) for pred in meta_predictions.values()], axis=0)
        rank_blend = rank_blend / len(rank_blend)
        
        # Retrain base models on full data
        print("\n🎯 Retraining base models on full data...")
        full_models = {}
        for model_name, model in base_models.items():
            print(f"  🔄 Retraining {model_name}...")
            if 'catboost' in model_name:
                model.fit(X, y, verbose=0)
            else:
                model.fit(X, y)
            full_models[model_name] = model
        
        # Get full predictions
        full_predictions = {}
        for model_name, model in full_models.items():
            full_predictions[model_name] = model.predict_proba(X_test)[:, 1]
        
        # Combine base model predictions
        base_ensemble = np.mean(list(full_predictions.values()), axis=0)
        
        # Final blending
        final_predictions = {
            'weighted_ensemble': 0.7 * weighted_blend + 0.3 * base_ensemble,
            'simple_ensemble': 0.6 * simple_blend + 0.4 * base_ensemble,
            'geometric_ensemble': 0.5 * geometric_blend + 0.5 * base_ensemble,
            'rank_ensemble': rank_blend,
            'pure_stack': weighted_blend,
            'pure_base': base_ensemble
        }
        
        return final_predictions, model_scores, meta_scores
    
    def fit_predict(self, train_path, test_path, output_dir="."):
        """Complete pipeline: load, preprocess, train, and predict"""
        
        print("🚀 Starting Ultimate Ensemble Pipeline...")
        
        # Load data
        print("📂 Loading data...")
        train = pd.read_csv(train_path)
        test = pd.read_csv(test_path)
        test_ids = test['SK_ID_CURR']
        
        # Feature engineering
        train = self.advanced_feature_engineering(train)
        test = self.advanced_feature_engineering(test)
        
        # Combine and preprocess
        train['is_train'] = 1
        test['is_train'] = 0
        test['TARGET'] = np.nan
        data = pd.concat([train, test], axis=0)
        
        # Handle categorical features
        cat_features = data.select_dtypes(include=['object']).columns.tolist()
        for col in cat_features:
            data[col].fillna('MISSING', inplace=True)
            if data[col].nunique() > 2:
                lbl = LabelEncoder()
                data[col] = lbl.fit_transform(data[col].astype(str))
        
        # Handle numerical features
        num_features = [col for col in data.columns if col not in cat_features + ['TARGET', 'SK_ID_CURR', 'is_train']]
        data[num_features] = SimpleImputer(strategy='median').fit_transform(data[num_features])
        
        # Split back
        train = data[data['is_train'] == 1].drop(columns=['is_train'])
        test = data[data['is_train'] == 0].drop(columns=['is_train', 'TARGET'])
        
        X = train.drop(columns=['TARGET', 'SK_ID_CURR'])
        y = train['TARGET']
        X_test = test.drop(columns=['SK_ID_CURR'])
        
        # Train models and get predictions
        predictions, model_scores, meta_scores = self.train_models(X, y, X_test)
        
        # Generate submissions
        print("\n📁 Generating submission files...")
        for name, pred in predictions.items():
            submission = pd.DataFrame({
                'SK_ID_CURR': test_ids,
                'TARGET': pred
            })
            submission.to_csv(f"{output_dir}/{name}_submission.csv", index=False)
            print(f"  ✅ Saved {name}_submission.csv")
        
        # Model analysis
        print("\n📊 Model Analysis:")
        print("-" * 40)
        
        # Best base model
        base_performance = {name: np.mean(scores) for name, scores in model_scores.items()}
        best_base = max(base_performance, key=base_performance.get)
        print(f"🏆 Best Base Model: {best_base} (AUC: {base_performance[best_base]:.5f})")
        
        # Best meta model
        best_meta = max(meta_scores, key=meta_scores.get)
        print(f"🏆 Best Meta Model: {best_meta} (AUC: {meta_scores[best_meta]:.5f})")
        
        print("\n🎉 Ultimate Ensemble Pipeline completed!")
        print("📁 Multiple submission files generated with different blending strategies.")
        
        return predictions, model_scores, meta_scores

# Example usage
if __name__ == "__main__":
    # Initialize ensemble
    ensemble = UltimateEnsemble(
        n_folds=5,
        feature_selection=True,
        n_features=200
    )
    
    # Run complete pipeline
    predictions, model_scores, meta_scores = ensemble.fit_predict(
        train_path="../input/home-credit-default-risk/application_train.csv",
        test_path="../input/home-credit-default-risk/application_test.csv"
    )