import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
import warnings
warnings.filterwarnings('ignore')

def analyze_feature_importance(X, y, top_n=50):
    """Analyze feature importance using multiple methods"""
    
    print("🔍 Analyzing feature importance...")
    
    # Method 1: Random Forest Feature Importance
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    rf_importance = pd.DataFrame({
        'feature': X.columns,
        'rf_importance': rf.feature_importances_
    }).sort_values('rf_importance', ascending=False)
    
    # Method 2: CatBoost Feature Importance
    cb = CatBoostClassifier(iterations=500, random_seed=42, verbose=0)
    cb.fit(X, y)
    cb_importance = pd.DataFrame({
        'feature': X.columns,
        'cb_importance': cb.feature_importances_
    }).sort_values('cb_importance', ascending=False)
    
    # Method 3: LightGBM Feature Importance
    lgb = LGBMClassifier(n_estimators=500, random_state=42, verbose=-1)
    lgb.fit(X, y)
    lgb_importance = pd.DataFrame({
        'feature': X.columns,
        'lgb_importance': lgb.feature_importances_
    }).sort_values('lgb_importance', ascending=False)
    
    # Method 4: F-statistic
    f_scores, _ = f_classif(X, y)
    f_importance = pd.DataFrame({
        'feature': X.columns,
        'f_score': f_scores
    }).sort_values('f_score', ascending=False)
    
    # Method 5: Mutual Information
    mi_scores = mutual_info_classif(X, y, random_state=42)
    mi_importance = pd.DataFrame({
        'feature': X.columns,
        'mi_score': mi_scores
    }).sort_values('mi_score', ascending=False)
    
    # Combine all methods
    importance_df = rf_importance.merge(cb_importance, on='feature')
    importance_df = importance_df.merge(lgb_importance, on='feature')
    importance_df = importance_df.merge(f_importance, on='feature')
    importance_df = importance_df.merge(mi_importance, on='feature')
    
    # Calculate average rank
    importance_df['avg_rank'] = (
        importance_df['rf_importance'].rank(ascending=False) +
        importance_df['cb_importance'].rank(ascending=False) +
        importance_df['lgb_importance'].rank(ascending=False) +
        importance_df['f_score'].rank(ascending=False) +
        importance_df['mi_score'].rank(ascending=False)
    ) / 5
    
    importance_df = importance_df.sort_values('avg_rank')
    
    # Get top features
    top_features = importance_df.head(top_n)['feature'].tolist()
    
    print(f"📊 Top {top_n} features by average rank:")
    print(importance_df.head(top_n)[['feature', 'avg_rank']].to_string(index=False))
    
    return importance_df, top_features

def plot_feature_importance(importance_df, top_n=20):
    """Plot feature importance"""
    
    plt.figure(figsize=(15, 10))
    
    # Top features by average rank
    top_features = importance_df.head(top_n)
    
    plt.subplot(2, 2, 1)
    plt.barh(range(len(top_features)), top_features['rf_importance'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.title('Random Forest Feature Importance')
    plt.xlabel('Importance')
    
    plt.subplot(2, 2, 2)
    plt.barh(range(len(top_features)), top_features['cb_importance'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.title('CatBoost Feature Importance')
    plt.xlabel('Importance')
    
    plt.subplot(2, 2, 3)
    plt.barh(range(len(top_features)), top_features['lgb_importance'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.title('LightGBM Feature Importance')
    plt.xlabel('Importance')
    
    plt.subplot(2, 2, 4)
    plt.barh(range(len(top_features)), top_features['avg_rank'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.title('Average Rank (Lower is Better)')
    plt.xlabel('Average Rank')
    
    plt.tight_layout()
    plt.savefig('feature_importance_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def correlation_analysis(X, top_features=None):
    """Analyze feature correlations"""
    
    if top_features:
        X_analysis = X[top_features]
    else:
        X_analysis = X
    
    print("🔗 Analyzing feature correlations...")
    
    # Calculate correlation matrix
    corr_matrix = X_analysis.corr()
    
    # Find highly correlated features
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) > 0.95:
                high_corr_pairs.append((
                    corr_matrix.columns[i],
                    corr_matrix.columns[j],
                    corr_matrix.iloc[i, j]
                ))
    
    print(f"📊 Found {len(high_corr_pairs)} highly correlated feature pairs (|correlation| > 0.95):")
    for feat1, feat2, corr in high_corr_pairs[:10]:  # Show first 10
        print(f"  {feat1} <-> {feat2}: {corr:.3f}")
    
    # Plot correlation heatmap for top features
    if top_features and len(top_features) <= 50:
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0)
        plt.title('Feature Correlation Heatmap (Top Features)')
        plt.tight_layout()
        plt.savefig('feature_correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    return corr_matrix, high_corr_pairs

def select_features(X, y, method='importance', n_features=100):
    """Select features using different methods"""
    
    print(f"🎯 Selecting {n_features} features using {method} method...")
    
    if method == 'importance':
        # Use feature importance
        importance_df, _ = analyze_feature_importance(X, y, top_n=n_features)
        selected_features = importance_df.head(n_features)['feature'].tolist()
    
    elif method == 'f_test':
        # Use F-test
        selector = SelectKBest(score_func=f_classif, k=n_features)
        selector.fit(X, y)
        selected_features = X.columns[selector.get_support()].tolist()
    
    elif method == 'mutual_info':
        # Use mutual information
        selector = SelectKBest(score_func=mutual_info_classif, k=n_features)
        selector.fit(X, y)
        selected_features = X.columns[selector.get_support()].tolist()
    
    else:
        raise ValueError("Method must be 'importance', 'f_test', or 'mutual_info'")
    
    print(f"✅ Selected {len(selected_features)} features")
    return selected_features

def analyze_target_distribution(y):
    """Analyze target variable distribution"""
    
    print("📊 Target variable analysis:")
    print(f"  Total samples: {len(y)}")
    print(f"  Positive samples: {y.sum()} ({y.mean()*100:.2f}%)")
    print(f"  Negative samples: {(y==0).sum()} ({(y==0).mean()*100:.2f}%)")
    
    plt.figure(figsize=(8, 6))
    y.value_counts().plot(kind='bar')
    plt.title('Target Variable Distribution')
    plt.xlabel('Target')
    plt.ylabel('Count')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig('target_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()

def analyze_numerical_features(X, y):
    """Analyze numerical features"""
    
    numerical_cols = X.select_dtypes(include=[np.number]).columns
    
    print(f"📈 Analyzing {len(numerical_cols)} numerical features...")
    
    # Calculate statistics for each feature
    feature_stats = []
    for col in numerical_cols:
        stats = {
            'feature': col,
            'mean': X[col].mean(),
            'std': X[col].std(),
            'min': X[col].min(),
            'max': X[col].max(),
            'correlation_with_target': X[col].corr(y),
            'missing_pct': X[col].isnull().mean() * 100
        }
        feature_stats.append(stats)
    
    stats_df = pd.DataFrame(feature_stats)
    stats_df = stats_df.sort_values('correlation_with_target', key=abs, ascending=False)
    
    print("📊 Top 10 features by absolute correlation with target:")
    print(stats_df.head(10)[['feature', 'correlation_with_target']].to_string(index=False))
    
    return stats_df

# Example usage
if __name__ == "__main__":
    """
    # Load your data
    # X = your_features
    # y = your_target
    
    # Analyze target distribution
    analyze_target_distribution(y)
    
    # Analyze numerical features
    numerical_stats = analyze_numerical_features(X, y)
    
    # Analyze feature importance
    importance_df, top_features = analyze_feature_importance(X, y, top_n=50)
    
    # Plot feature importance
    plot_feature_importance(importance_df, top_n=20)
    
    # Analyze correlations
    corr_matrix, high_corr_pairs = correlation_analysis(X, top_features[:30])
    
    # Select features
    selected_features = select_features(X, y, method='importance', n_features=100)
    
    print(f"🎉 Feature analysis completed!")
    print(f"📁 Generated plots: feature_importance_analysis.png, feature_correlation_heatmap.png, target_distribution.png")
    """
    pass