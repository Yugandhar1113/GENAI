# 🏆 Advanced Ensemble Model for Home Credit Default Risk

This repository contains a comprehensive ensemble model implementation for the Home Credit Default Risk competition on Kaggle. The model uses advanced stacking, blending, and boosting techniques to achieve top performance.

## 🚀 Features

### Advanced Ensemble Techniques
- **Multi-Level Stacking**: 9 base models + 6 meta-models
- **Advanced Blending**: Weighted, geometric, rank, and simple averaging
- **Stratified K-Fold Cross-Validation**: Ensures robust validation
- **Hyperparameter Optimization**: Using Optuna for automated tuning
- **Feature Engineering**: Domain-specific feature creation

### Base Models
1. **CatBoost** (2 variants with different configurations)
2. **LightGBM** (2 variants with different configurations)
3. **XGBoost** (2 variants with different configurations)
4. **Random Forest**
5. **Extra Trees**
6. **Gradient Boosting**

### Meta Models
1. **Logistic Regression**
2. **Ridge Classifier**
3. **CatBoost Meta**
4. **LightGBM Meta**
5. **XGBoost Meta**
6. **Neural Network (MLP)**

## 📁 File Structure

```
├── advanced_ensemble_model.py    # Main ensemble implementation
├── hyperparameter_optimization.py # Optuna optimization scripts
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## 🛠️ Installation

1. **Install Dependencies**:
```bash
pip install -r requirements.txt
```

2. **Data Setup**:
   - Place your data files in the correct directory structure:
   ```
   ../input/home-credit-default-risk/
   ├── application_train.csv
   └── application_test.csv
   ```

## 🎯 Usage

### Quick Start
```bash
python advanced_ensemble_model.py
```

### Hyperparameter Optimization (Optional)
```bash
python hyperparameter_optimization.py
```

## 📊 Model Performance

The ensemble generates multiple submission files with different blending strategies:

1. **weighted_ensemble_submission.csv** - Performance-weighted blending
2. **simple_ensemble_submission.csv** - Simple averaging
3. **geometric_ensemble_submission.csv** - Geometric mean
4. **rank_ensemble_submission.csv** - Rank averaging
5. **pure_stack_submission.csv** - Pure stacking predictions
6. **pure_base_submission.csv** - Pure base model ensemble

## 🔧 Advanced Features

### Feature Engineering
- Age and employment features
- Income ratios and payment features
- Document submission analysis
- External source aggregations
- Bureau, previous application, POS, installment, and credit card features

### Cross-Validation Strategy
- **Stratified K-Fold**: Maintains class distribution
- **5 Folds**: Optimal balance between computation and validation
- **Out-of-Fold Predictions**: Prevents data leakage

### Blending Strategies
1. **Weighted Average**: Based on meta-model performance
2. **Simple Average**: Equal weights for all meta-models
3. **Geometric Mean**: Handles different prediction scales
4. **Rank Averaging**: Robust to outliers

## 🎨 Customization

### Adding New Models
```python
def get_base_models():
    models = {
        # ... existing models ...
        'your_new_model': YourModelClass(
            param1=value1,
            param2=value2
        )
    }
    return models
```

### Modifying Blending Weights
```python
# In the final blending section
final_pred = 0.7 * weighted_blend + 0.3 * base_ensemble
```

### Feature Engineering
Add new features in the `advanced_feature_engineering()` function:
```python
def advanced_feature_engineering(df):
    # ... existing features ...
    
    # Your new features
    df['NEW_FEATURE'] = df['COL1'] / df['COL2']
    
    return df
```

## 📈 Performance Tips

### For Better Scores
1. **Run Hyperparameter Optimization**: Use the Optuna scripts
2. **Try Different Blending**: Test all submission files
3. **Feature Selection**: Remove low-importance features
4. **Ensemble Diversity**: Add more diverse base models
5. **Cross-Validation**: Increase folds for more robust validation

### Computational Considerations
- **GPU Acceleration**: Enable for XGBoost if available
- **Parallel Processing**: Models use all CPU cores
- **Memory Management**: Large datasets may require chunking
- **Early Stopping**: Prevents overfitting and saves time

## 🔍 Model Analysis

The script provides comprehensive analysis:
- Individual model performance
- Correlation matrix between predictions
- Best performing base and meta models
- Feature importance analysis

## 🏆 Competition Strategy

### Submission Strategy
1. **Primary**: Use `weighted_ensemble_submission.csv`
2. **Backup**: Try `geometric_ensemble_submission.csv`
3. **Diversity**: Submit different blending strategies
4. **Analysis**: Check correlation matrix for model diversity

### Advanced Techniques
- **Pseudo-Labeling**: Use confident predictions as additional training data
- **Feature Selection**: Remove redundant features
- **Model Pruning**: Remove poorly performing models
- **Ensemble Selection**: Choose best subset of models

## 🐛 Troubleshooting

### Common Issues
1. **Memory Error**: Reduce model complexity or use chunking
2. **Slow Training**: Enable GPU acceleration or reduce iterations
3. **Overfitting**: Increase regularization or reduce model complexity
4. **Poor Performance**: Run hyperparameter optimization

### Performance Monitoring
- Monitor cross-validation scores
- Check for overfitting (gap between CV and training scores)
- Analyze feature importance
- Review correlation matrix

## 📚 References

- [Home Credit Default Risk Competition](https://www.kaggle.com/c/home-credit-default-risk)
- [CatBoost Documentation](https://catboost.ai/)
- [LightGBM Documentation](https://lightgbm.readthedocs.io/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [Optuna Documentation](https://optuna.org/)

## 🤝 Contributing

Feel free to contribute improvements:
1. Add new models
2. Optimize hyperparameters
3. Improve feature engineering
4. Enhance blending strategies

## 📄 License

This project is for educational and competition purposes.

---

**Good luck with your Kaggle competition! 🚀**