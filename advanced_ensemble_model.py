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

print("🚀 Advanced Ensemble Model for Home Credit Default Risk")
print("=" * 60)

## Data Loading & Preprocessing
print("📊 Loading data...")
train = pd.read_csv("../input/home-credit-default-risk/application_train.csv")
test = pd.read_csv("../input/home-credit-default-risk/application_test.csv")
test_ids = test['SK_ID_CURR']

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")

# Feature Engineering
def advanced_feature_engineering(df):
    """Advanced feature engineering with domain knowledge"""
    
    # Age features
    df['DAYS_BIRTH'] = abs(df['DAYS_BIRTH'])
    df['AGE'] = df['DAYS_BIRTH'] / 365
    df['AGE_GROUP'] = pd.cut(df['AGE'], bins=[0, 25, 35, 45, 55, 100], labels=[0, 1, 2, 3, 4])
    
    # Employment features
    df['DAYS_EMPLOYED'] = abs(df['DAYS_EMPLOYED'])
    df['EMPLOYMENT_YEARS'] = df['DAYS_EMPLOYED'] / 365
    df['EMPLOYMENT_YEARS'] = df['EMPLOYMENT_YEARS'].clip(0, 50)
    
    # Income features
    df['INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS']
    df['INCOME_CREDIT_RATIO'] = df['AMT_INCOME_TOTAL'] / df['AMT_CREDIT']
    df['INCOME_ANNUITY_RATIO'] = df['AMT_INCOME_TOTAL'] / df['AMT_ANNUITY']
    
    # Credit features
    df['CREDIT_ANNUITY_RATIO'] = df['AMT_CREDIT'] / df['AMT_ANNUITY']
    df['CREDIT_GOODS_RATIO'] = df['AMT_CREDIT'] / df['AMT_GOODS_PRICE']
    
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
    df['EXT_SOURCE_MAX'] = df[ext_cols].max(axis=1)
    df['EXT_SOURCE_MIN'] = df[ext_cols].min(axis=1)
    
    # Bureau features (if available)
    bureau_cols = [col for col in df.columns if 'BUREAU' in col]
    if bureau_cols:
        df['BUREAU_MEAN'] = df[bureau_cols].mean(axis=1)
        df['BUREAU_STD'] = df[bureau_cols].std(axis=1)
    
    # Previous application features (if available)
    prev_cols = [col for col in df.columns if 'PREV' in col]
    if prev_cols:
        df['PREV_MEAN'] = df[prev_cols].mean(axis=1)
        df['PREV_STD'] = df[prev_cols].std(axis=1)
    
    # Polynomial features for important numerical columns
    important_nums = ['AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'EXT_SOURCE_MEAN']
    for col in important_nums:
        if col in df.columns:
            df[f'{col}_SQUARED'] = df[col] ** 2
            df[f'{col}_CUBED'] = df[col] ** 3
    
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
print(f"📝 Processing {len(cat_features)} categorical features...")

for col in cat_features:
    data[col].fillna('MISSING', inplace=True)
    if data[col].nunique() > 2:
        lbl = LabelEncoder()
        data[col] = lbl.fit_transform(data[col].astype(str))

# Handle numerical features
num_features = [col for col in data.columns if col not in cat_features + ['TARGET', 'SK_ID_CURR', 'is_train']]
print(f"🔢 Processing {len(num_features)} numerical features...")

# Advanced imputation
imputer = SimpleImputer(strategy='median')
data[num_features] = imputer.fit_transform(data[num_features])

# Feature scaling for some models
scaler = RobustScaler()
data[num_features] = scaler.fit_transform(data[num_features])

# Split back into train and test
train = data[data['is_train'] == 1].drop(columns=['is_train'])
test = data[data['is_train'] == 0].drop(columns=['is_train', 'TARGET'])

X = train.drop(columns=['TARGET', 'SK_ID_CURR'])
y = train['TARGET']
X_test = test.drop(columns=['SK_ID_CURR'])

print(f"Final feature count: {X.shape[1]}")

## Advanced Model Definitions
def get_base_models():
    """First-level base models with optimized hyperparameters"""
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
            min_child_weight=2, gamma=0.2
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
    """Second-level meta-models for stacking"""
    return {
        'logistic_regression': LogisticRegression(
            C=0.1, max_iter=2000, random_state=42, solver='liblinear'
        ),
        'ridge_classifier': RidgeClassifier(
            alpha=1.0, random_state=42
        ),
        'catboost_meta': CatBoostClassifier(
            iterations=1000, learning_rate=0.01, depth=4,
            eval_metric='AUC', random_seed=42, verbose=0,
            l2_leaf_reg=1
        ),
        'lightgbm_meta': LGBMClassifier(
            n_estimators=1000, learning_rate=0.01, max_depth=4,
            reg_alpha=1, reg_lambda=1, metric='auc',
            random_state=42, n_jobs=-1
        ),
        'xgboost_meta': XGBClassifier(
            n_estimators=1000, learning_rate=0.01, max_depth=4,
            reg_alpha=1, reg_lambda=1, eval_metric='auc',
            random_state=42, n_jobs=-1
        ),
        'neural_network': MLPClassifier(
            hidden_layer_sizes=(100, 50), activation='relu',
            solver='adam', alpha=0.001, learning_rate='adaptive',
            max_iter=1000, random_state=42
        )
    }

## Advanced K-Fold Training with Stratified Sampling
print("\n🎯 Training base models with stratified K-fold...")
n_folds = 5
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

# Prepare storage for predictions
base_models = get_base_models()
oof_predictions = {model_name: np.zeros(X.shape[0]) for model_name in base_models}
test_predictions = {model_name: np.zeros(X_test.shape[0]) for model_name in base_models}
model_scores = {model_name: [] for model_name in base_models}

# Train base models
for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y)):
    print(f"\n📊 Fold {fold + 1}/{n_folds}")
    X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
    y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
    
    for model_name, model in base_models.items():
        print(f"  🚀 Training {model_name}...")
        
        # Clone model for each fold
        model_copy = model.__class__(**model.get_params())
        
        if 'catboost' in model_name:
            model_copy.fit(
                X_train, y_train,
                eval_set=(X_valid, y_valid),
                use_best_model=True,
                verbose=0
            )
        else:
            model_copy.fit(X_train, y_train)
        
        # Store predictions
        oof_pred = model_copy.predict_proba(X_valid)[:, 1]
        oof_predictions[model_name][valid_idx] = oof_pred
        
        test_pred = model_copy.predict_proba(X_test)[:, 1]
        test_predictions[model_name] += test_pred / n_folds
        
        # Calculate fold score
        fold_auc = roc_auc_score(y_valid, oof_pred)
        model_scores[model_name].append(fold_auc)
        print(f"    ✅ {model_name} Fold {fold + 1} AUC: {fold_auc:.5f}")

## Evaluate Base Models
print("\n📈 Base Model Performance Summary:")
print("-" * 50)
for model_name in base_models:
    mean_auc = np.mean(model_scores[model_name])
    std_auc = np.std(model_scores[model_name])
    print(f"{model_name:>15}: {mean_auc:.5f} ± {std_auc:.5f}")

## Advanced Stacking
print("\n🔗 Creating advanced stacking ensemble...")

# Create stacked dataset
stacked_X = np.column_stack([oof_predictions[model] for model in base_models])
stacked_test = np.column_stack([test_predictions[model] for model in base_models])

# Train meta-models
meta_models = get_meta_models()
meta_predictions = {}

for meta_name, meta_model in meta_models.items():
    print(f"  🎯 Training meta-model: {meta_name}")
    meta_model.fit(stacked_X, y)
    meta_pred = meta_model.predict_proba(stacked_test)[:, 1]
    meta_predictions[meta_name] = meta_pred

## Advanced Blending Techniques
print("\n🎨 Applying advanced blending techniques...")

# 1. Weighted Average Blending (based on model performance)
model_weights = {}
for model_name in base_models:
    mean_score = np.mean(model_scores[model_name])
    model_weights[model_name] = mean_score

# Normalize weights
total_weight = sum(model_weights.values())
model_weights = {k: v/total_weight for k, v in model_weights.items()}

weighted_blend = np.zeros(X_test.shape[0])
for model_name, weight in model_weights.items():
    weighted_blend += weight * test_predictions[model_name]

# 2. Meta-model blending
meta_weights = {
    'logistic_regression': 0.25,
    'catboost_meta': 0.25,
    'lightgbm_meta': 0.25,
    'xgboost_meta': 0.15,
    'ridge_classifier': 0.05,
    'neural_network': 0.05
}

meta_blend = np.zeros(X_test.shape[0])
for meta_name, weight in meta_weights.items():
    meta_blend += weight * meta_predictions[meta_name]

# 3. Geometric mean blending
geometric_blend = np.ones(X_test.shape[0])
for model_name in base_models:
    geometric_blend *= test_predictions[model_name]
geometric_blend = np.power(geometric_blend, 1/len(base_models))

# 4. Harmonic mean blending
harmonic_blend = np.zeros(X_test.shape[0])
for model_name in base_models:
    harmonic_blend += 1 / (test_predictions[model_name] + 1e-8)
harmonic_blend = len(base_models) / harmonic_blend

## Final Ensemble
print("\n🏆 Creating final ensemble...")

# Combine all blending techniques
final_prediction = (
    0.35 * weighted_blend +
    0.30 * meta_blend +
    0.20 * geometric_blend +
    0.15 * harmonic_blend
)

# Apply sigmoid calibration if needed
final_prediction = 1 / (1 + np.exp(-final_prediction))

## Generate Multiple Submissions
print("\n💾 Generating submissions...")

# Main submission
submission_main = pd.DataFrame({
    'SK_ID_CURR': test_ids,
    'TARGET': final_prediction
})
submission_main.to_csv("advanced_ensemble_submission.csv", index=False)

# Individual model submissions for analysis
for model_name in base_models:
    submission_model = pd.DataFrame({
        'SK_ID_CURR': test_ids,
        'TARGET': test_predictions[model_name]
    })
    submission_model.to_csv(f"submission_{model_name}.csv", index=False)

# Meta-model submissions
for meta_name in meta_models:
    submission_meta = pd.DataFrame({
        'SK_ID_CURR': test_ids,
        'TARGET': meta_predictions[meta_name]
    })
    submission_meta.to_csv(f"submission_meta_{meta_name}.csv", index=False)

# Blending submissions
submission_weighted = pd.DataFrame({
    'SK_ID_CURR': test_ids,
    'TARGET': weighted_blend
})
submission_weighted.to_csv("submission_weighted_blend.csv", index=False)

submission_meta_blend = pd.DataFrame({
    'SK_ID_CURR': test_ids,
    'TARGET': meta_blend
})
submission_meta_blend.to_csv("submission_meta_blend.csv", index=False)

print("\n✅ All submissions generated successfully!")
print("\n📊 Submission files created:")
print("  - advanced_ensemble_submission.csv (main ensemble)")
print("  - submission_[model_name].csv (individual models)")
print("  - submission_meta_[meta_name].csv (meta-models)")
print("  - submission_weighted_blend.csv (weighted blending)")
print("  - submission_meta_blend.csv (meta-model blending)")

print(f"\n🎯 Expected performance based on cross-validation:")
print(f"  Best base model: {max([np.mean(scores) for scores in model_scores.values()]):.5f}")
print(f"  Average base model: {np.mean([np.mean(scores) for scores in model_scores.values()]):.5f}")

print("\n🚀 Advanced ensemble model training completed!")
print("💡 Tips for further improvement:")
print("  - Try different feature engineering techniques")
print("  - Experiment with different hyperparameters")
print("  - Add more base models (Neural Networks, SVM)")
print("  - Use different cross-validation strategies")
print("  - Implement feature selection")
print("  - Try different blending weights")