import pandas as pd
import numpy as np
import optuna
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import IsolationForest
import warnings
warnings.filterwarnings('ignore')

class AdvancedTechniques:
    def __init__(self, random_state=42):
        self.random_state = random_state
        
    def adversarial_validation(self, train_df, test_df, features=None):
        """
        Perform adversarial validation to detect distribution shift
        Returns AUC score - higher means more distribution shift
        """
        print("Performing adversarial validation...")
        
        if features is None:
            features = [col for col in train_df.columns 
                       if col not in ['TARGET', 'SK_ID_CURR']]
        
        # Create adversarial dataset
        train_adv = train_df[features].copy()
        test_adv = test_df[features].copy()
        
        train_adv['is_test'] = 0
        test_adv['is_test'] = 1
        
        adv_data = pd.concat([train_adv, test_adv], axis=0, ignore_index=True)
        
        # Handle missing values
        for col in features:
            if adv_data[col].dtype == 'object':
                adv_data[col].fillna('MISSING', inplace=True)
                if adv_data[col].nunique() > 2:
                    lbl = LabelEncoder()
                    adv_data[col] = lbl.fit_transform(adv_data[col].astype(str))
            else:
                adv_data[col].fillna(adv_data[col].median(), inplace=True)
        
        X_adv = adv_data[features]
        y_adv = adv_data['is_test']
        
        # Train adversarial model
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        adv_scores = []
        
        for fold, (train_idx, valid_idx) in enumerate(kf.split(X_adv, y_adv)):
            X_train, X_valid = X_adv.iloc[train_idx], X_adv.iloc[valid_idx]
            y_train, y_valid = y_adv.iloc[train_idx], y_adv.iloc[valid_idx]
            
            model = LGBMClassifier(
                n_estimators=500, learning_rate=0.05, max_depth=6,
                random_state=self.random_state, n_jobs=-1
            )
            model.fit(X_train, y_train)
            
            pred = model.predict_proba(X_valid)[:, 1]
            score = roc_auc_score(y_valid, pred)
            adv_scores.append(score)
        
        avg_score = np.mean(adv_scores)
        print(f"Adversarial validation AUC: {avg_score:.5f}")
        
        if avg_score > 0.55:
            print("⚠️  Warning: Significant distribution shift detected!")
            print("Consider using domain adaptation techniques.")
        else:
            print("✅ Distribution shift is minimal.")
        
        return avg_score
    
    def pseudo_labeling(self, train_df, test_df, model, confidence_threshold=0.95):
        """
        Generate pseudo-labels for confident predictions on test set
        """
        print("Generating pseudo-labels...")
        
        # Get features
        features = [col for col in train_df.columns 
                   if col not in ['TARGET', 'SK_ID_CURR']]
        
        X_train = train_df[features]
        y_train = train_df['TARGET']
        X_test = test_df[features]
        
        # Train model on original training data
        model.fit(X_train, y_train)
        
        # Get predictions on test set
        test_probs = model.predict_proba(X_test)[:, 1]
        
        # Identify confident predictions
        confident_positive = test_probs >= confidence_threshold
        confident_negative = test_probs <= (1 - confidence_threshold)
        confident_mask = confident_positive | confident_negative
        
        print(f"Found {confident_mask.sum()} confident predictions out of {len(test_probs)} test samples")
        print(f"Confident positive: {confident_positive.sum()}")
        print(f"Confident negative: {confident_negative.sum()}")
        
        if confident_mask.sum() > 0:
            # Create pseudo-labeled data
            pseudo_X = X_test[confident_mask]
            pseudo_y = (test_probs[confident_mask] > 0.5).astype(int)
            
            # Combine with original training data
            extended_X = pd.concat([X_train, pseudo_X], axis=0, ignore_index=True)
            extended_y = pd.concat([y_train, pd.Series(pseudo_y)], axis=0, ignore_index=True)
            
            return extended_X, extended_y, confident_mask
        else:
            print("No confident predictions found. Using original data.")
            return X_train, y_train, confident_mask
    
    def multi_objective_optimization(self, X, y, n_trials=100):
        """
        Multi-objective optimization for AUC and log-loss
        """
        print("Performing multi-objective optimization...")
        
        def objective(trial):
            # CatBoost parameters
            params = {
                'iterations': trial.suggest_int('iterations', 1000, 2000),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
                'depth': trial.suggest_int('depth', 4, 8),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
                'eval_metric': 'AUC',
                'early_stopping_rounds': 100,
                'random_seed': self.random_state,
                'verbose': 0
            }
            
            kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.random_state)
            auc_scores = []
            logloss_scores = []
            
            for fold, (train_idx, valid_idx) in enumerate(kf.split(X, y)):
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
                auc_score = roc_auc_score(y_valid, pred)
                logloss_score = log_loss(y_valid, pred)
                
                auc_scores.append(auc_score)
                logloss_scores.append(logloss_score)
            
            avg_auc = np.mean(auc_scores)
            avg_logloss = np.mean(logloss_scores)
            
            # Multi-objective: maximize AUC, minimize log-loss
            return avg_auc - 0.1 * avg_logloss
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        
        print(f"Best multi-objective score: {study.best_value:.5f}")
        return study.best_params
    
    def detect_outliers(self, df, contamination=0.1):
        """
        Detect outliers using Isolation Forest
        """
        print("Detecting outliers...")
        
        # Select numerical features
        num_features = df.select_dtypes(include=[np.number]).columns.tolist()
        if 'TARGET' in num_features:
            num_features.remove('TARGET')
        if 'SK_ID_CURR' in num_features:
            num_features.remove('SK_ID_CURR')
        
        # Handle missing values
        df_clean = df[num_features].fillna(df[num_features].median())
        
        # Detect outliers
        iso_forest = IsolationForest(
            contamination=contamination,
            random_state=self.random_state,
            n_jobs=-1
        )
        outlier_labels = iso_forest.fit_predict(df_clean)
        
        outlier_mask = outlier_labels == -1
        print(f"Detected {outlier_mask.sum()} outliers out of {len(df)} samples")
        
        return outlier_mask
    
    def feature_interaction_engineering(self, df):
        """
        Create interaction features between important variables
        """
        print("Creating interaction features...")
        
        # Important feature pairs for Home Credit
        interaction_pairs = [
            ('AMT_CREDIT', 'AMT_INCOME_TOTAL'),
            ('AMT_ANNUITY', 'AMT_INCOME_TOTAL'),
            ('AMT_CREDIT', 'AMT_ANNUITY'),
            ('EXT_SOURCE_1', 'EXT_SOURCE_2'),
            ('EXT_SOURCE_2', 'EXT_SOURCE_3'),
            ('EXT_SOURCE_1', 'EXT_SOURCE_3'),
            ('DAYS_BIRTH', 'DAYS_EMPLOYED'),
            ('CNT_CHILDREN', 'CNT_FAM_MEMBERS'),
        ]
        
        for feat1, feat2 in interaction_pairs:
            if feat1 in df.columns and feat2 in df.columns:
                # Multiplication
                df[f'{feat1}_x_{feat2}'] = df[feat1] * df[feat2]
                
                # Division (with protection against division by zero)
                df[f'{feat1}_div_{feat2}'] = df[feat1] / (df[feat2] + 1e-8)
                df[f'{feat2}_div_{feat1}'] = df[feat2] / (df[feat1] + 1e-8)
                
                # Addition and subtraction
                df[f'{feat1}_plus_{feat2}'] = df[feat1] + df[feat2]
                df[f'{feat1}_minus_{feat2}'] = df[feat1] - df[feat2]
        
        print(f"Created {len(interaction_pairs) * 5} interaction features")
        return df
    
    def time_based_validation(self, df, time_col='SK_ID_CURR', n_splits=5):
        """
        Create time-based validation splits (simulated for Home Credit)
        """
        print("Creating time-based validation splits...")
        
        # Sort by ID (proxy for time in this case)
        df_sorted = df.sort_values(time_col)
        
        # Create splits
        splits = []
        split_size = len(df_sorted) // n_splits
        
        for i in range(n_splits):
            if i == 0:
                train_end = split_size * (i + 2)
                valid_start = split_size * (i + 2)
                valid_end = split_size * (i + 3)
            else:
                train_end = split_size * (i + 2)
                valid_start = split_size * (i + 2)
                valid_end = min(split_size * (i + 3), len(df_sorted))
            
            if valid_end <= len(df_sorted):
                train_idx = df_sorted.iloc[:train_end].index.tolist()
                valid_idx = df_sorted.iloc[valid_start:valid_end].index.tolist()
                splits.append((train_idx, valid_idx))
        
        print(f"Created {len(splits)} time-based splits")
        return splits
    
    def ensemble_submissions(self, submission_files, weights=None):
        """
        Ensemble multiple submission files
        """
        print("Ensembling submission files...")
        
        submissions = []
        for file in submission_files:
            sub = pd.read_csv(file)
            submissions.append(sub)
        
        if weights is None:
            weights = [1.0 / len(submissions)] * len(submissions)
        
        # Weighted average
        final_sub = submissions[0].copy()
        final_sub['TARGET'] = 0
        
        for i, (sub, weight) in enumerate(zip(submissions, weights)):
            final_sub['TARGET'] += weight * sub['TARGET']
        
        return final_sub
    
    def post_processing_smoothing(self, predictions, alpha=0.01):
        """
        Apply post-processing smoothing to predictions
        """
        print("Applying post-processing smoothing...")
        
        # Rank-based smoothing
        predictions_smoothed = predictions.copy()
        
        # Apply power transformation
        predictions_smoothed = np.power(predictions_smoothed, alpha)
        
        # Normalize to [0, 1]
        predictions_smoothed = (predictions_smoothed - predictions_smoothed.min()) / \
                              (predictions_smoothed.max() - predictions_smoothed.min())
        
        return predictions_smoothed

def run_advanced_pipeline(train_path, test_path):
    """
    Run the complete advanced pipeline
    """
    print("🚀 Starting Advanced Home Credit Pipeline...")
    
    # Initialize
    advanced = AdvancedTechniques(random_state=42)
    
    # Load data
    print("Loading data...")
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    
    # 1. Adversarial Validation
    adv_score = advanced.adversarial_validation(train, test)
    
    # 2. Feature Interaction Engineering
    train = advanced.feature_interaction_engineering(train)
    test = advanced.feature_interaction_engineering(test)
    
    # 3. Outlier Detection
    outlier_mask = advanced.detect_outliers(train)
    print(f"Removing {outlier_mask.sum()} outliers from training data")
    train_clean = train[~outlier_mask].reset_index(drop=True)
    
    # 4. Prepare features
    features = [col for col in train_clean.columns 
               if col not in ['TARGET', 'SK_ID_CURR']]
    
    # Handle categorical features
    cat_features = train_clean.select_dtypes(include=['object']).columns.tolist()
    for col in cat_features:
        if col in features:
            train_clean[col].fillna('MISSING', inplace=True)
            test[col].fillna('MISSING', inplace=True)
            
            lbl = LabelEncoder()
            combined = pd.concat([train_clean[col], test[col]], axis=0)
            lbl.fit(combined.astype(str))
            
            train_clean[col] = lbl.transform(train_clean[col].astype(str))
            test[col] = lbl.transform(test[col].astype(str))
    
    # Handle numerical features
    num_features = [col for col in features if col not in cat_features]
    imputer = SimpleImputer(strategy='median')
    train_clean[num_features] = imputer.fit_transform(train_clean[num_features])
    test[num_features] = imputer.transform(test[num_features])
    
    X = train_clean[features]
    y = train_clean['TARGET']
    X_test = test[features]
    
    # 5. Multi-objective Optimization
    best_params = advanced.multi_objective_optimization(X, y, n_trials=50)
    
    # 6. Pseudo-labeling
    base_model = CatBoostClassifier(**best_params, verbose=0)
    X_extended, y_extended, confident_mask = advanced.pseudo_labeling(
        train_clean, test, base_model, confidence_threshold=0.95
    )
    
    # 7. Final model training with extended data
    print("Training final model with extended data...")
    final_model = CatBoostClassifier(**best_params, verbose=0)
    final_model.fit(X_extended, y_extended)
    
    # 8. Generate predictions
    final_predictions = final_model.predict_proba(X_test)[:, 1]
    
    # 9. Post-processing
    final_predictions = advanced.post_processing_smoothing(final_predictions)
    
    # 10. Create submission
    submission = pd.DataFrame({
        'SK_ID_CURR': test['SK_ID_CURR'],
        'TARGET': final_predictions
    })
    
    submission.to_csv("advanced_techniques_submission.csv", index=False)
    print("✅ Advanced techniques submission saved!")
    
    return submission, adv_score

if __name__ == "__main__":
    train_path = "../input/home-credit-default-risk/application_train.csv"
    test_path = "../input/home-credit-default-risk/application_test.csv"
    
    try:
        submission, adv_score = run_advanced_pipeline(train_path, test_path)
        print(f"\n🎯 Pipeline completed successfully!")
        print(f"Adversarial validation score: {adv_score:.5f}")
        print("Submission saved as 'advanced_techniques_submission.csv'")
        
    except FileNotFoundError:
        print("❌ Data files not found. Please ensure the data files are in the correct path.")
        print("Expected paths:")
        print("  - ../input/home-credit-default-risk/application_train.csv")
        print("  - ../input/home-credit-default-risk/application_test.csv")