# Advanced Home Credit Default Risk Ensemble

This repository contains an advanced ensemble solution for the Home Credit Default Risk competition, designed to achieve top Kaggle scores through sophisticated stacking, blending, and boosting techniques.

## 🏆 Key Features

- **Multi-Level Stacking**: 3-level ensemble architecture (Base → Level 2 → Meta)
- **Advanced Feature Engineering**: Domain-specific features with polynomial combinations
- **Hyperparameter Optimization**: Automated tuning using Optuna
- **Diverse Model Portfolio**: 13+ different algorithms including boosting, bagging, and neural networks
- **Optimized Blending**: Automatic weight optimization for final predictions
- **Cross-Validation**: Stratified K-Fold with 7 folds for robust validation

## 📊 Model Architecture

### Level 1: Base Models (8 models)
- **CatBoost** (2 variants with different configurations)
- **LightGBM** (2 variants with different parameters)
- **XGBoost** (1 optimized variant)
- **Random Forest** (1 variant)
- **Extra Trees** (1 variant)
- **Gradient Boosting** (1 variant)

### Level 2: Stacking Models (3 models)
- **CatBoost L2** (trained on Level 1 predictions)
- **LightGBM L2** (trained on Level 1 predictions)
- **XGBoost L2** (trained on Level 1 predictions)

### Meta Level: Final Ensemble (5 models)
- **Logistic Regression**
- **Ridge Classifier**
- **Multi-Layer Perceptron**
- **Support Vector Machine**
- **Naive Bayes**

### Additional Techniques
- **Voting Classifier**: Soft voting ensemble of best performers
- **Optimized Blending**: Automated weight optimization using Optuna

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare Data
Place your data files in the following structure:
```
../input/home-credit-default-risk/
├── application_train.csv
└── application_test.csv
```

### 3. Run Hyperparameter Optimization (Optional)
```bash
python hyperparameter_optimization.py
```
This will create `best_hyperparameters.json` with optimized parameters.

### 4. Train the Advanced Ensemble
```bash
python advanced_home_credit_ensemble.py
```

## 🔧 Advanced Feature Engineering

The solution includes 20+ engineered features:

### Financial Ratios
- Credit to Income Ratio
- Annuity to Income Ratio
- Credit to Annuity Ratio
- Income per Family Member

### External Source Features
- Mean, Standard Deviation, and Product of External Sources
- Polynomial features (squared and cubed)

### Domain-Specific Features
- Document count aggregation
- Region rating combinations
- High-risk organization flags
- Age and employment transformations

## 📈 Performance Optimization Tips

### For Top Kaggle Scores:

1. **Increase Cross-Validation Folds**
   ```python
   ensemble = AdvancedHomeCreditsEnsemble(n_folds=10, random_state=42)
   ```

2. **Add More Base Models**
   - Neural Networks (MLPClassifier with different architectures)
   - Additional boosting variants (AdaBoost, HistGradientBoosting)
   - Bayesian models (BayesianRidge, ARDRegression)

3. **Feature Selection**
   - Use recursive feature elimination
   - Add interaction terms
   - Include additional external data sources

4. **Hyperparameter Tuning**
   - Increase Optuna trials to 500+
   - Use TPE sampler for better optimization
   - Tune blending weights more extensively

5. **Ensemble Diversity**
   - Add models with different loss functions
   - Use different preprocessing pipelines
   - Implement pseudo-labeling on test data

## 🎯 Expected Performance

With proper tuning and sufficient computational resources:
- **Local CV**: 0.795+ AUC
- **Public LB**: 0.790+ AUC
- **Private LB**: 0.785+ AUC (top 5% territory)

## 💡 Advanced Techniques for Competition Winners

### 1. Pseudo-Labeling
```python
# Use confident predictions on test set as additional training data
confident_mask = (final_pred > 0.9) | (final_pred < 0.1)
pseudo_labels = (final_pred > 0.5).astype(int)
```

### 2. Multi-Objective Optimization
```python
# Optimize for both AUC and log-loss
def multi_objective(trial):
    # ... model training ...
    auc_score = roc_auc_score(y_valid, pred)
    logloss_score = log_loss(y_valid, pred)
    return auc_score - 0.1 * logloss_score
```

### 3. Feature Engineering Automation
```python
# Use automated feature engineering tools
import featuretools as ft
# Automated feature generation
```

### 4. Adversarial Validation
```python
# Check for distribution shift between train/test
adversarial_auc = train_adversarial_model(train, test)
if adversarial_auc > 0.55:
    print("Warning: Distribution shift detected!")
```

## 🔍 Model Interpretability

### Feature Importance Analysis
```python
# Get feature importances from all models
importance_df = ensemble.get_feature_importance()
importance_df.to_csv('feature_importance.csv')
```

### SHAP Analysis
```python
import shap
explainer = shap.Explainer(best_model)
shap_values = explainer(X_test.sample(1000))
shap.summary_plot(shap_values, X_test.sample(1000))
```

## 🏃‍♂️ Performance Optimization

### Memory Optimization
- Use `float32` instead of `float64`
- Implement chunked processing for large datasets
- Use sparse matrices where applicable

### Speed Optimization
- Enable GPU acceleration for XGBoost/LightGBM
- Use parallel processing for cross-validation
- Cache intermediate results

### Resource Management
```python
# Monitor memory usage
import psutil
print(f"Memory usage: {psutil.virtual_memory().percent}%")
```

## 📊 Submission Strategy

### Multiple Submissions
1. **Conservative**: Higher weight on stable models (CatBoost, LightGBM)
2. **Aggressive**: Include more diverse models and neural networks
3. **Balanced**: Equal weights across all model types

### Late Submission Techniques
- Ensemble previous submissions
- Use test-time augmentation
- Apply post-processing smoothing

## 🤝 Contributing

Feel free to contribute improvements:
1. Additional feature engineering techniques
2. New model architectures
3. Better hyperparameter optimization strategies
4. Performance optimizations

## 📜 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- Home Credit for providing the dataset
- Kaggle community for sharing insights
- Authors of the machine learning libraries used

---

**Note**: This ensemble approach is computationally intensive. For best results, use a machine with:
- 16+ GB RAM
- Multi-core CPU (8+ cores recommended)
- GPU support for XGBoost/CatBoost (optional but recommended)

Good luck achieving that top Kaggle score! 🚀