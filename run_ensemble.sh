#!/bin/bash

echo "🚀 Home Credit Default Risk - Advanced Ensemble Solution"
echo "========================================================"

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.7+ first."
    exit 1
fi

# Install dependencies
echo "📦 Installing dependencies..."
pip install -r requirements.txt

# Check if data directory exists
if [ ! -d "../input/home-credit-default-risk" ]; then
    echo "⚠️  Data directory not found. Please ensure data is available at:"
    echo "   ../input/home-credit-default-risk/application_train.csv"
    echo "   ../input/home-credit-default-risk/application_test.csv"
    echo ""
    echo "🔄 Running with sample data for demonstration..."
fi

# Run hyperparameter optimization (optional)
read -p "🔧 Do you want to run hyperparameter optimization? (y/N): " optimize
if [[ $optimize =~ ^[Yy]$ ]]; then
    echo "🔧 Running hyperparameter optimization..."
    python hyperparameter_optimization.py
    echo "✅ Hyperparameter optimization completed!"
fi

# Run main ensemble solution
echo "🎯 Running main ensemble solution..."
python advanced_ensemble_solution.py

echo ""
echo "🎉 Ensemble solution completed!"
echo "📊 Check the generated submission files:"
echo "   - final_ensemble_submission.csv (recommended)"
echo "   - stacked_ensemble_submission.csv"
echo "   - voting_ensemble_submission.csv"
echo "   - bayesian_ensemble_submission.csv"
echo ""
echo "🏆 Good luck with your Kaggle submission!"