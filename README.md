# Advanced Ensemble Model for Home Credit Default Risk

This repository contains a comprehensive ensemble model approach for the Home Credit Default Risk competition on Kaggle. The implementation uses advanced stacking, blending, and boosting techniques to achieve top performance.

## 🚀 Features

### Advanced Feature Engineering
- **Domain-specific features**: Age groups, employment categories, income ratios
- **Interaction features**: External source combinations, document counts
- **Polynomial features**: Squared, cubed, and logarithmic transformations
- **Risk scoring**: Custom risk score based on multiple factors
- **Contact and address features**: Aggregated contact and address information

### Multiple Base Models
- **CatBoost**: 2 variants with different hyperparameters
- **LightGBM**: 2 variants with different configurations
- **XGBoost**: 2 variants optimized for different aspects
- **Random Forest**: Traditional ensemble method
- **Extra Trees**: More randomized version of Random Forest
- **Gradient Boosting**: Sklearn's gradient boosting implementation

### Advanced Stacking
- **Stratified K-Fold**: Ensures balanced class distribution across folds
- **Feature stacking**: Combines model predictions with selected original features
- **Multiple meta-models**: Logistic Regression, Ridge, CatBoost, LightGBM, XGBoost, Neural Network

### Advanced Blending Techniques
1. **Weighted Average Blending**: Optimized weights based on model performance
2. **Meta-model Blending**: Combines predictions from different meta-models
3. **Rank-based Blending**: Uses percentile ranks instead of raw predictions
4. **Geometric Mean Blending**: Geometric mean of all predictions
5. **Calibration**: Isotonic calibration for better probability estimates

## 📁 File Structure

```
├── advanced_ensemble_model.py    # Main ensemble model script
├── hyperparameter_optimization.py # Optuna-based hyperparameter optimization
├── requirements.txt              # Python dependencies
├── README.md                     # This file
└── data/                         # Data directory (not included)
    ├── application_train.csv
    └── application_test.csv
```

## 🛠️ Installation

1. Clone this repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## 🎯 Usage

### Basic Usage
```bash
python advanced_ensemble_model.py
```

### Hyperparameter Optimization (Optional)
```bash
python hyperparameter_optimization.py
```

## 📊 Model Architecture

### Level 1: Base Models
- **CatBoost Variants**: Different bootstrap types and hyperparameters
- **LightGBM Variants**: Different regularization and sampling strategies
- **XGBoost Variants**: Different tree methods and regularization
- **Traditional Ensembles**: Random Forest, Extra Trees, Gradient Boosting

### Level 2: Meta-Models
- **Linear Models**: Logistic Regression, Ridge Classifier
- **Tree-based Meta**: CatBoost, LightGBM, XGBoost
- **Neural Network**: Multi-layer Perceptron

### Level 3: Final Blending
- **Weighted Average**: 35% weight
- **Meta-model Blend**: 30% weight
- **Rank-based Blend**: 20% weight
- **Geometric Mean**: 15% weight
- **Calibration**: Applied to final predictions

## 🎨 Advanced Techniques Used

### 1. Stratified K-Fold Cross-Validation
- Ensures each fold has the same proportion of target classes
- Reduces variance in cross-validation scores
- More reliable model evaluation

### 2. Feature Selection for Stacking
- Uses SelectKBest with f_classif
- Selects top 50 features for stacking
- Combines model predictions with selected features

### 3. Multiple Blending Approaches
- **Weighted Average**: Traditional approach with optimized weights
- **Rank-based**: Robust to outliers and different prediction scales
- **Geometric Mean**: Handles multiplicative relationships
- **Meta-model**: Learns optimal combination from data

### 4. Probability Calibration
- Uses isotonic calibration
- Improves probability estimates
- Better for ranking and threshold-based decisions

## 📈 Performance Optimization

### Hyperparameter Tuning
- **Optuna**: Bayesian optimization for hyperparameters
- **Cross-validation**: 5-fold stratified CV for reliable estimates
- **Early stopping**: Prevents overfitting in gradient boosting models

### Model Diversity
- **Different algorithms**: CatBoost, LightGBM, XGBoost, Random Forest
- **Different hyperparameters**: Various learning rates, depths, regularization
- **Different random seeds**: Ensures model diversity

### Feature Engineering
- **Domain knowledge**: Credit-specific features
- **Statistical features**: Means, standard deviations, interactions
- **Polynomial features**: Captures non-linear relationships

## 🏆 Expected Performance

This ensemble approach typically achieves:
- **AUC Score**: 0.800+ on the Home Credit Default Risk competition
- **Robust Performance**: Consistent across different validation sets
- **Good Generalization**: Well-calibrated probabilities

## 🔧 Customization

### Adding New Models
1. Add model definition to `get_base_models()` function
2. Ensure model has `fit()` and `predict_proba()` methods
3. Update blending weights if necessary

### Feature Engineering
1. Modify `advanced_feature_engineering()` function
2. Add domain-specific features
3. Test impact on cross-validation performance

### Blending Strategy
1. Adjust weights in the final blending section
2. Add new blending methods
3. Optimize weights using validation set

## 📝 Output Files

- `advanced_ensemble_submission.csv`: Final submission file
- `model_performance_summary.csv`: Detailed performance metrics for each model
- Console output: Real-time training progress and scores

## 🚨 Important Notes

1. **Data Path**: Update the data path in the script to match your setup
2. **Memory Usage**: The ensemble requires significant RAM (8GB+ recommended)
3. **Training Time**: Full training takes 1-2 hours depending on hardware
4. **GPU Support**: XGBoost uses GPU acceleration if available

## 🤝 Contributing

Feel free to submit issues and enhancement requests!

## 📄 License

This project is licensed under the MIT License.

---

**Note**: This implementation is designed for educational purposes and competition use. Always validate results on your own validation sets and adjust hyperparameters based on your specific data characteristics.