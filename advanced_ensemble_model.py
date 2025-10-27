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
    df['EMPLOYMENT_GROUP'] = pd.cut(df['EMPLOYMENT_YEARS'], bins=[0, 1, 3, 5, 10, 50], labels=[0, 1, 2, 3, 4])
    
    # Income features
    df['INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS']
    df['INCOME_CREDIT_RATIO'] = df['AMT_INCOME_TOTAL'] / df['AMT_CREDIT']
    df['INCOME_ANNUITY_RATIO'] = df['AMT_INCOME_TOTAL'] / df['AMT_ANNUITY']
    df['CREDIT_ANNUITY_RATIO'] = df['AMT_CREDIT'] / df['AMT_ANNUITY']
    
    # Loan features
    df['LOAN_TO_INCOME'] = df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL']
    df['ANNUITY_TO_INCOME'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
    df['PAYMENT_RATE'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']
    
    # External source features (create interaction features)
    ext_cols = [col for col in df.columns if 'EXT_SOURCE' in col]
    if len(ext_cols) >= 2:
        df['EXT_SOURCE_MEAN'] = df[ext_cols].mean(axis=1)
        df['EXT_SOURCE_STD'] = df[ext_cols].std(axis=1)
        df['EXT_SOURCE_MAX'] = df[ext_cols].max(axis=1)
        df['EXT_SOURCE_MIN'] = df[ext_cols].min(axis=1)
        df['EXT_SOURCE_SUM'] = df[ext_cols].sum(axis=1)
        df['EXT_SOURCE_PROD'] = df[ext_cols].prod(axis=1)
    
    # Document features
    doc_cols = [col for col in df.columns if 'FLAG_DOC' in col]
    if doc_cols:
        df['DOCUMENT_COUNT'] = df[doc_cols].sum(axis=1)
        df['DOCUMENT_RATIO'] = df['DOCUMENT_COUNT'] / len(doc_cols)
    
    # Contact features
    contact_cols = [col for col in df.columns if 'FLAG_MOBIL' in col or 'FLAG_EMP_PHONE' in col or 'FLAG_WORK_PHONE' in col or 'FLAG_CONT_MOBILE' in col or 'FLAG_PHONE' in col or 'FLAG_EMAIL' in col]
    if contact_cols:
        df['CONTACT_COUNT'] = df[contact_cols].sum(axis=1)
        df['CONTACT_RATIO'] = df['CONTACT_COUNT'] / len(contact_cols)
    
    # Address features
    address_cols = [col for col in df.columns if 'FLAG_REG_REGION' in col or 'FLAG_LIVE_REGION' in col or 'FLAG_REG_CITY' in col or 'FLAG_LIVE_CITY' in col]
    if address_cols:
        df['ADDRESS_COUNT'] = df[address_cols].sum(axis=1)
        df['ADDRESS_RATIO'] = df['ADDRESS_COUNT'] / len(address_cols)
    
    # Application features
    app_cols = [col for col in df.columns if 'FLAG_LAST_APPL_PER_CONTRACT' in col or 'FLAG_LAST_APPL_IN_DAY' in col or 'FLAG_TERM_YEAR' in col]
    if app_cols:
        df['APP_COUNT'] = df[app_cols].sum(axis=1)
        df['APP_RATIO'] = df['APP_COUNT'] / len(app_cols)
    
    # Risk features
    df['RISK_SCORE'] = (
        df['DAYS_BIRTH'] * 0.1 +
        df['DAYS_EMPLOYED'] * 0.2 +
        df['CNT_CHILDREN'] * 0.3 +
        df['CNT_FAM_MEMBERS'] * 0.2 +
        df['AMT_INCOME_TOTAL'] * 0.1 +
        df['AMT_CREDIT'] * 0.1
    )
    
    # Polynomial features for important numerical columns
    important_nums = ['AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'DAYS_BIRTH', 'DAYS_EMPLOYED']
    for col in important_nums:
        if col in df.columns:
            df[f'{col}_SQUARED'] = df[col] ** 2
            df[f'{col}_CUBED'] = df[col] ** 3
            df[f'{col}_LOG'] = np.log1p(np.abs(df[col]))
    
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
    """Get optimized base models"""
    return {
        'catboost_1': CatBoostClassifier(
            iterations=2000, learning_rate=0.02, depth=8,
            l2_leaf_reg=3, eval_metric='AUC', early_stopping_rounds=150,
            cat_features=cat_features, random_seed=42, thread_count=-1,
            verbose=0, bootstrap_type='Bernoulli', subsample=0.8
        ),
        'catboost_2': CatBoostClassifier(
            iterations=2000, learning_rate=0.03, depth=6,
            l2_leaf_reg=5, eval_metric='AUC', early_stopping_rounds=150,
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
            n_estimators=2000, learning_rate=0.03, max_depth=6,
            reg_alpha=5, reg_lambda=5, metric='auc',
            random_state=123, n_jobs=-1, subsample=0.9, colsample_bytree=0.9,
            min_child_samples=30, min_child_weight=1e-2
        ),
        'xgboost_1': XGBClassifier(
            n_estimators=2000, learning_rate=0.02, max_depth=8,
            reg_alpha=3, reg_lambda=3, eval_metric='auc',
            random_state=42, n_jobs=-1, subsample=0.8, colsample_bytree=0.8,
            min_child_weight=1, tree_method='hist'
        ),
        'xgboost_2': XGBClassifier(
            n_estimators=2000, learning_rate=0.03, max_depth=6,
            reg_alpha=5, reg_lambda=5, eval_metric='auc',
            random_state=123, n_jobs=-1, subsample=0.9, colsample_bytree=0.9,
            min_child_weight=3, tree_method='hist'
        ),
        'random_forest': RandomForestClassifier(
            n_estimators=1000, max_depth=10, max_features='sqrt',
            random_state=42, n_jobs=-1, min_samples_split=10, min_samples_leaf=5
        ),
        'extra_trees': ExtraTreesClassifier(
            n_estimators=1000, max_depth=10, max_features='sqrt',
            random_state=42, n_jobs=-1, min_samples_split=10, min_samples_leaf=5
        ),
        'gradient_boosting': GradientBoostingClassifier(
            n_estimators=1000, learning_rate=0.02, max_depth=6,
            random_state=42, subsample=0.8, min_samples_split=10, min_samples_leaf=5
        )
    }

def get_meta_models():
    """Get meta-models for stacking"""
    return {
        'logistic': LogisticRegression(C=0.1, max_iter=2000, random_state=42),
        'ridge': RidgeClassifier(alpha=1.0, random_state=42),
        'catboost_meta': CatBoostClassifier(
            iterations=1000, learning_rate=0.01, depth=4,
            eval_metric='AUC', random_seed=42, verbose=0
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
        'mlp': MLPClassifier(
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

print("🎯 Training base models with stratified K-fold...")

for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y)):
    print(f"\n📁 Fold {fold + 1}/{n_folds}")
    X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
    y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
    
    for model_name, model in base_models.items():
        print(f"  🚀 Training {model_name}...")
        
        # Clone model for this fold
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
        oof_predictions[model_name][valid_idx] = model_copy.predict_proba(X_valid)[:, 1]
        test_predictions[model_name] += model_copy.predict_proba(X_test)[:, 1] / n_folds
        
        # Calculate fold score
        fold_auc = roc_auc_score(y_valid, oof_predictions[model_name][valid_idx])
        model_scores[model_name].append(fold_auc)
        print(f"  ✅ {model_name} Fold {fold + 1} AUC: {fold_auc:.5f}")

# Average test predictions
for model_name in test_predictions:
    test_predictions[model_name] /= n_folds

## Evaluate Base Models
print("\n📈 Base Model Performance:")
print("-" * 60)
for model_name in oof_predictions:
    mean_auc = np.mean(model_scores[model_name])
    std_auc = np.std(model_scores[model_name])
    final_auc = roc_auc_score(y, oof_predictions[model_name])
    print(f"{model_name:>15}: Mean AUC = {mean_auc:.5f} ± {std_auc:.5f}, Final AUC = {final_auc:.5f}")

## Advanced Stacking
print("\n🔗 Creating stacked dataset...")
stacked_X = np.column_stack([oof_predictions[model] for model in oof_predictions])
stacked_test = np.column_stack([test_predictions[model] for model in test_predictions])

# Add original features to stacked dataset (feature stacking)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

# Select top features for stacking
selector = SelectKBest(f_classif, k=min(50, X.shape[1]))
X_selected = selector.fit_transform(X_scaled, y)
X_test_selected = selector.transform(X_test_scaled)

# Combine predictions with selected features
stacked_X_enhanced = np.column_stack([stacked_X, X_selected])
stacked_test_enhanced = np.column_stack([stacked_test, X_test_selected])

## Meta-Model Training
print("\n🎯 Training meta-models...")
meta_models = get_meta_models()
meta_predictions = {}

for meta_name, meta_model in meta_models.items():
    print(f"  🚀 Training {meta_name} meta-model...")
    
    if 'catboost' in meta_name:
        meta_model.fit(stacked_X_enhanced, y, verbose=0)
    else:
        meta_model.fit(stacked_X_enhanced, y)
    
    meta_predictions[meta_name] = meta_model.predict_proba(stacked_test_enhanced)[:, 1]
    
    # Calculate meta-model score
    meta_oof_pred = meta_model.predict_proba(stacked_X_enhanced)[:, 1]
    meta_auc = roc_auc_score(y, meta_oof_pred)
    print(f"  ✅ {meta_name} Meta AUC: {meta_auc:.5f}")

## Advanced Blending Techniques
print("\n🎨 Applying advanced blending techniques...")

# 1. Weighted Average Blending (optimized weights)
weights = {
    'catboost_1': 0.15, 'catboost_2': 0.15,
    'lightgbm_1': 0.12, 'lightgbm_2': 0.12,
    'xgboost_1': 0.12, 'xgboost_2': 0.12,
    'random_forest': 0.08, 'extra_trees': 0.08,
    'gradient_boosting': 0.06
}

weighted_blend = np.zeros(X_test.shape[0])
for model_name, weight in weights.items():
    weighted_blend += weight * test_predictions[model_name]

# 2. Meta-model blending
meta_weights = {
    'logistic': 0.2, 'ridge': 0.15, 'catboost_meta': 0.25,
    'lightgbm_meta': 0.2, 'xgboost_meta': 0.15, 'mlp': 0.05
}

meta_blend = np.zeros(X_test.shape[0])
for meta_name, weight in meta_weights.items():
    meta_blend += weight * meta_predictions[meta_name]

# 3. Rank-based blending
def rank_blend(predictions_dict):
    """Create rank-based blend of predictions"""
    rank_matrix = np.zeros((X_test.shape[0], len(predictions_dict)))
    for i, (model_name, preds) in enumerate(predictions_dict.items()):
        rank_matrix[:, i] = pd.Series(preds).rank(pct=True)
    return rank_matrix.mean(axis=1)

rank_blend_pred = rank_blend(test_predictions)

# 4. Geometric mean blending
def geometric_mean_blend(predictions_dict):
    """Create geometric mean blend of predictions"""
    pred_matrix = np.column_stack(list(predictions_dict.values()))
    # Add small epsilon to avoid log(0)
    pred_matrix = np.clip(pred_matrix, 1e-7, 1-1e-7)
    return np.exp(np.mean(np.log(pred_matrix), axis=1))

geo_blend_pred = geometric_mean_blend(test_predictions)

## Final Ensemble
print("\n🏆 Creating final ensemble...")

# Combine all blending approaches
final_pred = (
    0.35 * weighted_blend +
    0.30 * meta_blend +
    0.20 * rank_blend_pred +
    0.15 * geo_blend_pred
)

# Apply calibration
from sklearn.calibration import CalibratedClassifierCV
calibrator = CalibratedClassifierCV(method='isotonic', cv=3)
calibrator.fit(stacked_X_enhanced, y)
final_pred_calibrated = calibrator.predict_proba(stacked_test_enhanced)[:, 1]

# Final blend with calibration
final_pred = 0.8 * final_pred + 0.2 * final_pred_calibrated

## Generate Submission
submission = pd.DataFrame({
    'SK_ID_CURR': test_ids,
    'TARGET': final_pred
})

# Ensure predictions are within [0, 1]
submission['TARGET'] = np.clip(submission['TARGET'], 0, 1)

submission.to_csv("advanced_ensemble_submission.csv", index=False)

print("\n✅ Advanced ensemble submission saved as advanced_ensemble_submission.csv")

# Print final statistics
print(f"\n📊 Final Statistics:")
print(f"Mean prediction: {final_pred.mean():.5f}")
print(f"Std prediction: {final_pred.std():.5f}")
print(f"Min prediction: {final_pred.min():.5f}")
print(f"Max prediction: {final_pred.max():.5f}")

# Save model performance summary
performance_summary = pd.DataFrame({
    'Model': list(model_scores.keys()),
    'Mean_AUC': [np.mean(scores) for scores in model_scores.values()],
    'Std_AUC': [np.std(scores) for scores in model_scores.values()],
    'Final_AUC': [roc_auc_score(y, oof_predictions[model]) for model in model_scores.keys()]
})
performance_summary = performance_summary.sort_values('Final_AUC', ascending=False)
performance_summary.to_csv("model_performance_summary.csv", index=False)

print("\n🎉 Advanced ensemble model training completed!")
print("📈 Check model_performance_summary.csv for detailed results")