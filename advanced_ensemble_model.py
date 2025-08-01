import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
import random
random.seed(42)

print("🚀 Loading and preprocessing data...")

## Data Loading & Preprocessing
train = pd.read_csv("../input/home-credit-default-risk/application_train.csv")
test = pd.read_csv("../input/home-credit-default-risk/application_test.csv")
test_ids = test['SK_ID_CURR']

# Advanced Feature Engineering
def advanced_feature_engineering(df):
    """Advanced feature engineering with domain knowledge"""
    
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
    df['DOCS_SUBMITTED'] = df[doc_cols].sum(axis=1)
    df['DOCS_SUBMITTED_RATIO'] = df['DOCS_SUBMITTED'] / len(doc_cols)
    
    # Contact features
    contact_cols = [col for col in df.columns if 'FLAG_' in col and 'DOC' not in col]
    df['CONTACT_FLAGS'] = df[contact_cols].sum(axis=1)
    
    # External source features
    ext_cols = [col for col in df.columns if 'EXT_SOURCE' in col]
    df['EXT_SOURCE_MEAN'] = df[ext_cols].mean(axis=1)
    df['EXT_SOURCE_STD'] = df[ext_cols].std(axis=1)
    df['EXT_SOURCE_SUM'] = df[ext_cols].sum(axis=1)
    df['EXT_SOURCE_MAX'] = df[ext_cols].max(axis=1)
    df['EXT_SOURCE_MIN'] = df[ext_cols].min(axis=1)
    
    # Bureau features (if available)
    bureau_cols = [col for col in df.columns if 'BUREAU' in col]
    if bureau_cols:
        df['BUREAU_COUNT'] = df[bureau_cols].count(axis=1)
        df['BUREAU_MEAN'] = df[bureau_cols].mean(axis=1)
    
    # Previous application features (if available)
    prev_cols = [col for col in df.columns if 'PREV' in col]
    if prev_cols:
        df['PREV_COUNT'] = df[prev_cols].count(axis=1)
        df['PREV_MEAN'] = df[prev_cols].mean(axis=1)
    
    # POS cash features (if available)
    pos_cols = [col for col in df.columns if 'POS' in col]
    if pos_cols:
        df['POS_COUNT'] = df[pos_cols].count(axis=1)
        df['POS_MEAN'] = df[pos_cols].mean(axis=1)
    
    # Installments features (if available)
    install_cols = [col for col in df.columns if 'INSTAL' in col]
    if install_cols:
        df['INSTAL_COUNT'] = df[install_cols].count(axis=1)
        df['INSTAL_MEAN'] = df[install_cols].mean(axis=1)
    
    # Credit card features (if available)
    cc_cols = [col for col in df.columns if 'CC_' in col]
    if cc_cols:
        df['CC_COUNT'] = df[cc_cols].count(axis=1)
        df['CC_MEAN'] = df[cc_cols].mean(axis=1)
    
    return df

print("🔧 Applying advanced feature engineering...")
train = advanced_feature_engineering(train)
test = advanced_feature_engineering(test)

# Combine and preprocess data
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

# Split back into train and test
train = data[data['is_train'] == 1].drop(columns=['is_train'])
test = data[data['is_train'] == 0].drop(columns=['is_train', 'TARGET'])

X = train.drop(columns=['TARGET', 'SK_ID_CURR'])
y = train['TARGET']
X_test = test.drop(columns=['SK_ID_CURR'])

print(f"📊 Training data shape: {X.shape}")
print(f"📊 Test data shape: {X_test.shape}")

## Advanced Model Definitions
def get_base_models():
    """First level models with optimized hyperparameters"""
    return {
        'catboost_1': CatBoostClassifier(
            iterations=2000, learning_rate=0.02, depth=8,
            l2_leaf_reg=3, eval_metric='AUC', early_stopping_rounds=150,
            cat_features=cat_features, random_seed=42, thread_count=-1,
            verbose=0, bootstrap_type='Bernoulli', subsample=0.8
        ),
        'catboost_2': CatBoostClassifier(
            iterations=1500, learning_rate=0.03, depth=6,
            l2_leaf_reg=5, eval_metric='AUC', early_stopping_rounds=100,
            cat_features=cat_features, random_seed=123, thread_count=-1,
            verbose=0, bootstrap_type='MVS', subsample=0.9
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

def get_meta_models():
    """Second level models for stacking"""
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

## Advanced K-Fold Training with Stratified Sampling
n_folds = 5
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

# Prepare storage for predictions
base_models = get_base_models()
oof_predictions = {model_name: np.zeros(X.shape[0]) for model_name in base_models}
test_predictions = {model_name: np.zeros(X_test.shape[0]) for model_name in base_models}
model_scores = {model_name: [] for model_name in base_models}

print(f"\n🎯 Training {len(base_models)} base models with {n_folds}-fold cross-validation...")

for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y)):
    print(f"\n📁 Fold {fold + 1}/{n_folds}")
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
        test_predictions[model_name] += model.predict_proba(X_test)[:, 1] / n_folds
        
        # Calculate fold score
        fold_auc = roc_auc_score(y_valid, oof_predictions[model_name][valid_idx])
        model_scores[model_name].append(fold_auc)
        print(f"    ✅ {model_name} Fold {fold + 1} AUC: {fold_auc:.5f}")

## Evaluate Base Models
print("\n📈 Base Model Performance Summary:")
print("-" * 60)
for model_name in base_models:
    scores = model_scores[model_name]
    mean_score = np.mean(scores)
    std_score = np.std(scores)
    print(f"{model_name:>15}: {mean_score:.5f} ± {std_score:.5f}")

## Advanced Stacking Implementation
print("\n🔗 Creating stacked dataset...")
stacked_X = np.column_stack([oof_predictions[model] for model in base_models])
stacked_test = np.column_stack([test_predictions[model] for model in base_models])

# Feature scaling for meta-models
scaler = StandardScaler()
stacked_X_scaled = scaler.fit_transform(stacked_X)
stacked_test_scaled = scaler.transform(stacked_test)

## Meta-Model Training
meta_models = get_meta_models()
meta_predictions = {}
meta_scores = {}

print(f"\n🎯 Training {len(meta_models)} meta-models...")

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
    
    # Calculate meta-model score
    meta_score = roc_auc_score(y, meta_model.predict(X_meta))
    meta_scores[meta_name] = meta_score
    print(f"    ✅ {meta_name} Meta AUC: {meta_score:.5f}")

## Advanced Blending Strategies
print("\n🎨 Implementing advanced blending strategies...")

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

## Ensemble with Base Models
print("\n🎯 Creating final ensemble...")

# Retrain base models on full data
full_models = {}
for model_name, model in get_base_models().items():
    print(f"  🔄 Retraining {model_name} on full data...")
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

## Final Blending with Multiple Strategies
print("\n🎨 Final blending with multiple strategies...")

# Combine stacked predictions with base ensemble
final_pred_1 = 0.7 * weighted_blend + 0.3 * base_ensemble
final_pred_2 = 0.6 * simple_blend + 0.4 * base_ensemble
final_pred_3 = 0.5 * geometric_blend + 0.5 * base_ensemble

# Create multiple submission files
submissions = {
    'weighted_ensemble': final_pred_1,
    'simple_ensemble': final_pred_2,
    'geometric_ensemble': final_pred_3,
    'rank_ensemble': rank_blend,
    'pure_stack': weighted_blend,
    'pure_base': base_ensemble
}

# Generate submissions
for name, pred in submissions.items():
    submission = pd.DataFrame({
        'SK_ID_CURR': test_ids,
        'TARGET': pred
    })
    submission.to_csv(f"{name}_submission.csv", index=False)
    print(f"  ✅ Saved {name}_submission.csv")

## Model Analysis and Insights
print("\n📊 Model Analysis:")
print("-" * 40)

# Best performing base models
base_performance = {name: np.mean(scores) for name, scores in model_scores.items()}
best_base = max(base_performance, key=base_performance.get)
print(f"🏆 Best Base Model: {best_base} (AUC: {base_performance[best_base]:.5f})")

# Best performing meta models
best_meta = max(meta_scores, key=meta_scores.get)
print(f"🏆 Best Meta Model: {best_meta} (AUC: {meta_scores[best_meta]:.5f})")

# Correlation analysis
print(f"\n📈 Prediction Correlations:")
stacked_df = pd.DataFrame(stacked_X, columns=base_models.keys())
correlation_matrix = stacked_df.corr()
print(correlation_matrix.round(3))

print("\n🎉 Advanced ensemble model training completed!")
print("📁 Multiple submission files generated with different blending strategies.")
print("🔍 Analyze the correlation matrix to understand model diversity.")
print("🏆 Try different submission files to find the best performing ensemble.")