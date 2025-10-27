# 🏆 Advanced Ensemble Solution for Home Credit Default Risk

This repository contains a comprehensive ensemble solution designed to achieve top scores on the Kaggle Home Credit Default Risk competition. The solution implements multiple advanced techniques including stacking, blending, voting, and hyperparameter optimization.

## 🎯 Key Features

### 1. **Advanced Ensemble Techniques**
- **Multi-level Stacking**: Uses 9 diverse base models with 5 different meta-learners
- **Voting Classifiers**: Soft voting with optimized weights
- **Bayesian Model Averaging**: Weighted predictions based on model performance
- **Blending**: Multiple ensemble combination strategies

### 2. **Optimized Base Models**
- **CatBoost** (2 variants with different hyperparameters)
- **LightGBM** (2 variants with different configurations)
- **XGBoost** (2 variants with different settings)
- **Random Forest** with optimized parameters
- **Extra Trees** for additional diversity
- **Gradient Boosting** for robust predictions

### 3. **Advanced Feature Engineering**
- Credit-to-income ratios and financial indicators
- Age and employment-related features
- External source combinations and interactions
- Polynomial features for important variables
- Document and credit bureau aggregations

### 4. **Hyperparameter Optimization**
- **Optuna-based optimization** for all models
- **Ensemble weight optimization** using Bayesian optimization
- **Cross-validation** with stratified folds
- **Early stopping** to prevent overfitting

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare Data
Ensure your data is in the following structure:
```
../input/home-credit-default-risk/
├── application_train.csv
└── application_test.csv
```

### 3. Run Hyperparameter Optimization (Optional but Recommended)
```bash
python hyperparameter_optimization.py
```
This will generate `optimized_parameters.json` and `optimized_weights.json`.

### 4. Run Main Ensemble Solution
```bash
python advanced_ensemble_solution.py
```

## 📊 Expected Performance

Based on the advanced techniques implemented:

- **Base Models**: Individual AUC scores typically range from 0.75-0.79
- **Stacked Ensemble**: Expected AUC improvement of 0.01-0.02
- **Final Ensemble**: Target AUC of 0.80+ (top 10% territory)

## 🔧 Customization Options

### Model Configuration
You can easily modify the base models in `AdvancedEnsembleClassifier.get_base_models()`:

```python
def get_base_models(self):
    return {
        'catboost_1': CatBoostClassifier(
            iterations=2000,
            learning_rate=0.02,
            depth=8,
            # ... other parameters
        ),
        # Add your own models here
    }
```

### Feature Engineering
Extend the feature engineering in `advanced_feature_engineering()`:

```python
def advanced_feature_engineering(self, df):
    # Add your custom features here
    df['CUSTOM_FEATURE'] = df['FEATURE_1'] / df['FEATURE_2']
    return df
```

### Ensemble Weights
Modify ensemble combination weights in `ensemble_predictions()`:

```python
ensemble_weights = {
    'weighted_meta': 0.5,    # Stacked predictions
    'voting': 0.25,          # Voting classifier
    'bayesian': 0.25         # Bayesian averaging
}
```

## 📈 Advanced Techniques Explained

### 1. **Multi-Level Stacking**
```
Level 0: Base Models (9 models)
    ↓
Level 1: Meta Models (5 models)
    ↓
Level 2: Final Ensemble
```

### 2. **Cross-Validation Strategy**
- **7-fold Stratified CV** for robust validation
- **Out-of-fold predictions** to prevent overfitting
- **Early stopping** based on validation performance

### 3. **Feature Selection**
- **Statistical feature selection** using f_classif
- **Top 500 features** selected automatically
- **Correlation-based filtering** to remove redundant features

### 4. **Ensemble Diversity**
- **Different algorithms**: Tree-based, linear, neural networks
- **Different hyperparameters**: Various depths, learning rates
- **Different random seeds**: Ensures prediction diversity

## 🎯 Tips for Top Kaggle Performance

### 1. **Data Quality**
- Handle missing values carefully
- Create meaningful feature interactions
- Use domain knowledge for feature engineering

### 2. **Model Diversity**
- Use models with different inductive biases
- Vary hyperparameters significantly
- Include both complex and simple models

### 3. **Validation Strategy**
- Use the same CV strategy as the leaderboard
- Monitor for overfitting on validation set
- Consider time-based splits if applicable

### 4. **Ensemble Optimization**
- Optimize ensemble weights using validation data
- Try different combination methods
- Consider non-linear ensemble methods

## 📁 Output Files

The solution generates multiple submission files:

1. `stacked_ensemble_submission.csv` - Pure stacking approach
2. `voting_ensemble_submission.csv` - Voting classifier predictions
3. `bayesian_ensemble_submission.csv` - Bayesian model averaging
4. `final_ensemble_submission.csv` - **Best combined ensemble** (recommended)

## 🔍 Monitoring and Debugging

### Performance Tracking
The solution provides detailed logging:
- Individual model fold performances
- Out-of-fold (OOF) AUC scores
- Meta-model stacking performance
- Final ensemble combination results

### Validation Scores
Monitor these key metrics:
- Base model OOF scores should be > 0.75
- Meta-model improvements should be visible
- Final ensemble should outperform individual models

## ⚡ Performance Optimization

### Computational Efficiency
- **Parallel processing**: All models use `n_jobs=-1`
- **GPU acceleration**: XGBoost uses `tree_method='gpu_hist'`
- **Memory optimization**: Efficient data handling

### Training Time
- Base model training: ~2-4 hours (depending on hardware)
- Hyperparameter optimization: ~4-8 hours
- Total pipeline: ~6-12 hours for full optimization

## 🏅 Competition Strategy

### For Top Performance:
1. **Run hyperparameter optimization** first
2. **Use the optimized parameters** in main ensemble
3. **Submit multiple ensemble variants** to test performance
4. **Blend with external datasets** if allowed
5. **Post-process predictions** (calibration, clipping)

### Submission Tips:
- Submit the `final_ensemble_submission.csv` as your primary entry
- Use other ensemble variants as backup submissions
- Monitor public leaderboard feedback
- Avoid overfitting to public LB

## 🤝 Contributing

Feel free to contribute improvements:
- Additional base models
- Better feature engineering
- Novel ensemble techniques
- Performance optimizations

## 📝 License

This project is open source and available under the MIT License.

---

**Good luck achieving top Kaggle performance! 🚀**

For questions or improvements, please open an issue or submit a pull request.