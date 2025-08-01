import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, VotingClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
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
    
    # Credit features
    df['CREDIT_ANNUITY_RATIO'] = df['AMT_CREDIT'] / df['AMT_ANNUITY']
    df['CREDIT_GOODS_RATIO'] = df['AMT_CREDIT'] / df['AMT_GOODS_PRICE']
    
    # Payment features
    df['PAYMENT_RATE'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']
    
    # Categorical interactions
    df['CODE_GENDER_AGE'] = df['CODE_GENDER'].astype(str) + '_' + df['AGE_GROUP'].astype(str)
    df['EDUCATION_INCOME'] = df['NAME_EDUCATION_TYPE'].astype(str) + '_' + pd.qcut(df['AMT_INCOME_TOTAL'], q=5, labels=['VL', 'L', 'M', 'H', 'VH']).astype(str)
    
    # Risk features
    df['RISK_SCORE'] = (
        df['DAYS_BIRTH'] * 0.1 +
        df['DAYS_EMPLOYED'] * 0.2 +
        df['CNT_CHILDREN'] * 0.3 +
        df['CNT_FAM_MEMBERS'] * 0.2
    )
    
    # Polynomial features for important numerical columns
    important_nums = ['AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'DAYS_BIRTH', 'DAYS_EMPLOYED']
    for col in important_nums:
        if col in df.columns:
            df[f'{col}_SQUARED'] = df[col] ** 2
            df[f'{col}_CUBED'] = df[col] ** 3
            df[f'{col}_LOG'] = np.log1p(df[col])
    
    return df

print("🔧 Applying advanced feature engineering...")
train = advanced_feature_engineering(train)
test = advanced_feature_engineering(test)

# Combine and preprocess data
train['is_train'] = 1
test['is_train'] = 0
test['TARGET'] = np.nan
data = pd.concat([train, test], axis=0)

# Handle categorical features with advanced encoding
cat_features = data.select_dtypes(include=['object']).columns.tolist()
for col in cat_features:
    data[col].fillna('MISSING', inplace=True)
    if data[col].nunique() > 2:
        lbl = LabelEncoder()
        data[col] = lbl.fit_transform(data[col].astype(str))

# Handle numerical features with robust imputation
num_features = [col for col in data.columns if col not in cat_features + ['TARGET', 'SK_ID_CURR', 'is_train']]
data[num_features] = SimpleImputer(strategy='median').fit_transform(data[num_features])

# Split back into train and test
train = data[data['is_train'] == 1].drop(columns=['is_train'])
test = data[data['is_train'] == 0].drop(columns=['is_train', 'TARGET'])

X = train.drop(columns=['TARGET', 'SK_ID_CURR'])
y = train['TARGET']
X_test = test.drop(columns=['SK_ID_CURR'])

print(f"Final feature count: {X.shape[1]}")

## Advanced Model Definitions
def get_advanced_models():
    """Get a comprehensive set of models with optimized hyperparameters"""
    
    # CatBoost models with different configurations
    catboost_models = {
        'catboost_v1': CatBoostClassifier(
            iterations=2000, learning_rate=0.02, depth=8,
            l2_leaf_reg=3, eval_metric='AUC', early_stopping_rounds=150,
            cat_features=cat_features, random_seed=42, thread_count=-1,
            verbose=0, bootstrap_type='Bernoulli', subsample=0.8
        ),
        'catboost_v2': CatBoostClassifier(
            iterations=1500, learning_rate=0.03, depth=6,
            l2_leaf_reg=5, eval_metric='AUC', early_stopping_rounds=100,
            cat_features=cat_features, random_seed=123, thread_count=-1,
            verbose=0, bootstrap_type='MVS', subsample=0.9
        ),
        'catboost_v3': CatBoostClassifier(
            iterations=1000, learning_rate=0.05, depth=10,
            l2_leaf_reg=1, eval_metric='AUC', early_stopping_rounds=50,
            cat_features=cat_features, random_seed=456, thread_count=-1,
            verbose=0, bootstrap_type='Bayesian'
        )
    }
    
    # LightGBM models with different configurations
    lightgbm_models = {
        'lightgbm_v1': LGBMClassifier(
            n_estimators=2000, learning_rate=0.02, max_depth=8,
            reg_alpha=3, reg_lambda=3, metric='auc',
            random_state=42, n_jobs=-1, subsample=0.8, colsample_bytree=0.8,
            min_child_samples=20, min_child_weight=1e-3
        ),
        'lightgbm_v2': LGBMClassifier(
            n_estimators=1500, learning_rate=0.03, max_depth=6,
            reg_alpha=5, reg_lambda=5, metric='auc',
            random_state=123, n_jobs=-1, subsample=0.9, colsample_bytree=0.9,
            min_child_samples=30, min_child_weight=1e-2
        ),
        'lightgbm_v3': LGBMClassifier(
            n_estimators=1000, learning_rate=0.05, max_depth=10,
            reg_alpha=1, reg_lambda=1, metric='auc',
            random_state=456, n_jobs=-1, subsample=0.7, colsample_bytree=0.7,
            min_child_samples=10, min_child_weight=1e-4
        )
    }
    
    # XGBoost models with different configurations
    xgboost_models = {
        'xgboost_v1': XGBClassifier(
            n_estimators=2000, learning_rate=0.02, max_depth=8,
            reg_alpha=3, reg_lambda=3, eval_metric='auc',
            random_state=42, n_jobs=-1, tree_method='hist',
            subsample=0.8, colsample_bytree=0.8, min_child_weight=1e-3
        ),
        'xgboost_v2': XGBClassifier(
            n_estimators=1500, learning_rate=0.03, max_depth=6,
            reg_alpha=5, reg_lambda=5, eval_metric='auc',
            random_state=123, n_jobs=-1, tree_method='hist',
            subsample=0.9, colsample_bytree=0.9, min_child_weight=1e-2
        ),
        'xgboost_v3': XGBClassifier(
            n_estimators=1000, learning_rate=0.05, max_depth=10,
            reg_alpha=1, reg_lambda=1, eval_metric='auc',
            random_state=456, n_jobs=-1, tree_method='hist',
            subsample=0.7, colsample_bytree=0.7, min_child_weight=1e-4
        )
    }
    
    # Traditional ML models
    traditional_models = {
        'random_forest': RandomForestClassifier(
            n_estimators=1000, max_depth=10, max_features='sqrt',
            random_state=42, n_jobs=-1, min_samples_split=10,
            min_samples_leaf=5, bootstrap=True, oob_score=True
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
    
    # Neural Network
    neural_models = {
        'mlp': MLPClassifier(
            hidden_layer_sizes=(256, 128, 64), activation='relu',
            solver='adam', alpha=0.001, learning_rate='adaptive',
            max_iter=500, random_state=42, early_stopping=True,
            validation_fraction=0.1, n_iter_no_change=20
        )
    }
    
    # Combine all models
    all_models = {}
    all_models.update(catboost_models)
    all_models.update(lightgbm_models)
    all_models.update(xgboost_models)
    all_models.update(traditional_models)
    all_models.update(neural_models)
    
    return all_models

## Advanced K-Fold Training with Stratified Sampling
print("\n🎯 Training base models with advanced cross-validation...")
n_folds = 5
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

models = get_advanced_models()
oof_predictions = {model_name: np.zeros(X.shape[0]) for model_name in models}
test_predictions = {model_name: np.zeros(X_test.shape[0]) for model_name in models}
model_scores = {}

for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y)):
    print(f"\n📊 Fold {fold + 1}/{n_folds}")
    X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
    y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
    
    for model_name, model in models.items():
        print(f"  🚀 Training {model_name}...")
        
        try:
            if 'catboost' in model_name:
                model.fit(
                    X_train, y_train,
                    eval_set=(X_valid, y_valid),
                    use_best_model=True,
                    verbose=0
                )
            elif 'mlp' in model_name:
                # Scale features for neural network
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_valid_scaled = scaler.transform(X_valid)
                model.fit(X_train_scaled, y_train)
                oof_pred = model.predict_proba(X_valid_scaled)[:, 1]
                test_pred = model.predict_proba(StandardScaler().fit_transform(X_test))[:, 1]
            else:
                model.fit(X_train, y_train)
                oof_pred = model.predict_proba(X_valid)[:, 1]
                test_pred = model.predict_proba(X_test)[:, 1]
            
            if 'mlp' not in model_name:
                oof_predictions[model_name][valid_idx] = oof_pred
                test_predictions[model_name] += test_pred / n_folds
            
            # Calculate fold AUC
            fold_auc = roc_auc_score(y_valid, oof_predictions[model_name][valid_idx])
            print(f"    ✅ {model_name} Fold {fold + 1} AUC: {fold_auc:.5f}")
            
        except Exception as e:
            print(f"    ❌ Error training {model_name}: {str(e)}")
            continue

## Evaluate Base Models
print("\n📈 Base Model Performance:")
for model_name in oof_predictions:
    if np.any(oof_predictions[model_name] != 0):
        auc = roc_auc_score(y, oof_predictions[model_name])
        model_scores[model_name] = auc
        print(f"{model_name:>20}: OOF AUC = {auc:.5f}")

# Sort models by performance
sorted_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
print(f"\n🏆 Top 5 models:")
for i, (model_name, score) in enumerate(sorted_models[:5], 1):
    print(f"{i}. {model_name}: {score:.5f}")

## Advanced Stacking with Multiple Meta-Models
print("\n🔗 Creating advanced stacking ensemble...")

# Use top models for stacking
top_models = [model for model, _ in sorted_models[:8]]
stacked_X = np.column_stack([oof_predictions[model] for model in top_models])
stacked_test = np.column_stack([test_predictions[model] for model in top_models])

# Multiple meta-models
meta_models = {
    'logistic_regression': LogisticRegression(C=0.1, max_iter=1000, random_state=42),
    'ridge_classifier': RidgeClassifier(alpha=1.0, random_state=42),
    'catboost_meta': CatBoostClassifier(
        iterations=1000, learning_rate=0.01, depth=5,
        eval_metric='AUC', random_seed=42, verbose=0
    ),
    'xgboost_meta': XGBClassifier(
        n_estimators=1000, learning_rate=0.01, max_depth=5,
        eval_metric='auc', random_state=42
    ),
    'lightgbm_meta': LGBMClassifier(
        n_estimators=1000, learning_rate=0.01, max_depth=5,
        metric='auc', random_state=42
    )
}

meta_predictions = {}
for meta_name, meta_model in meta_models.items():
    print(f"  🎯 Training {meta_name} meta-model...")
    meta_model.fit(stacked_X, y)
    meta_predictions[meta_name] = meta_model.predict_proba(stacked_test)[:, 1]

## Advanced Blending with Optimization
print("\n🎛️ Optimizing blend weights...")

# Simple blending first
simple_blend = np.mean([meta_predictions[name] for name in meta_predictions], axis=0)

# Weighted blending (can be optimized further)
weighted_blend = (
    0.3 * meta_predictions['catboost_meta'] +
    0.25 * meta_predictions['xgboost_meta'] +
    0.25 * meta_predictions['lightgbm_meta'] +
    0.1 * meta_predictions['logistic_regression'] +
    0.1 * meta_predictions['ridge_classifier']
)

## Voting Classifier with Top Models
print("\n🗳️ Creating voting classifier...")

# Retrain top models on full data
full_models = {}
for model_name in top_models[:5]:  # Use top 5 models
    print(f"  🔄 Retraining {model_name} on full data...")
    model = models[model_name]
    if 'catboost' in model_name:
        model.fit(X, y, verbose=0)
    elif 'mlp' in model_name:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        model.fit(X_scaled, y)
    else:
        model.fit(X, y)
    full_models[model_name] = model

# Create voting classifier
voting_clf = VotingClassifier(
    estimators=[(name, model) for name, model in full_models.items()],
    voting='soft'
)
voting_clf.fit(X, y)
voting_pred = voting_clf.predict_proba(X_test)[:, 1]

## Final Ensemble with Multiple Strategies
print("\n🎯 Creating final ensemble...")

# Strategy 1: Stacked predictions
stacked_final = weighted_blend

# Strategy 2: Voting classifier
voting_final = voting_pred

# Strategy 3: Simple average of top models
top_model_preds = np.column_stack([test_predictions[model] for model in top_models[:5]])
simple_avg = np.mean(top_model_preds, axis=1)

# Strategy 4: Weighted average based on model performance
weights = np.array([model_scores[model] for model in top_models[:5]])
weights = weights / weights.sum()
weighted_avg = np.average(top_model_preds, axis=1, weights=weights)

# Final blending of all strategies
final_prediction = (
    0.4 * stacked_final +
    0.3 * voting_final +
    0.2 * weighted_avg +
    0.1 * simple_avg
)

## Generate Multiple Submissions
print("\n💾 Generating submissions...")

# Main submission
submission_main = pd.DataFrame({
    'SK_ID_CURR': test_ids,
    'TARGET': final_prediction
})
submission_main.to_csv("advanced_ensemble_submission.csv", index=False)

# Alternative submissions for blending
submission_stacked = pd.DataFrame({
    'SK_ID_CURR': test_ids,
    'TARGET': stacked_final
})
submission_stacked.to_csv("stacked_only_submission.csv", index=False)

submission_voting = pd.DataFrame({
    'SK_ID_CURR': test_ids,
    'TARGET': voting_final
})
submission_voting.to_csv("voting_only_submission.csv", index=False)

submission_weighted = pd.DataFrame({
    'SK_ID_CURR': test_ids,
    'TARGET': weighted_avg
})
submission_weighted.to_csv("weighted_avg_submission.csv", index=False)

print("\n✅ All submissions generated successfully!")
print("📁 Files created:")
print("  - advanced_ensemble_submission.csv (main ensemble)")
print("  - stacked_only_submission.csv (stacking only)")
print("  - voting_only_submission.csv (voting only)")
print("  - weighted_avg_submission.csv (weighted average)")

## Model Performance Summary
print("\n📊 Final Model Performance Summary:")
print("=" * 50)
for model_name, score in sorted_models:
    print(f"{model_name:>25}: {score:.5f}")

print(f"\n🎯 Ensemble Strategy:")
print(f"  - Stacked (40%): {roc_auc_score(y, np.mean([oof_predictions[model] for model in top_models[:8]], axis=0)):.5f}")
print(f"  - Voting (30%): Combined with top 5 models")
print(f"  - Weighted Avg (20%): Based on individual model performance")
print(f"  - Simple Avg (10%): Equal weights for top 5 models")

print("\n🚀 Ready for Kaggle submission!")
print("💡 Tip: Try different combinations of the generated files for optimal results!")