# Titanic Survival Prediction Project

## Overview
This project analyzes the Titanic dataset to predict passenger survival using various machine learning models. The analysis includes extensive data preprocessing, exploratory data analysis (EDA), and model comparison.

## Project Structure
```
titanic/
│   README.md
│   requirements.txt
│   gender_submission_by_random-forest.csv
│
├── data/
│   ├── train.csv
│   └── test.csv
│
└── notebooks/
    └── titanic_ml_model.ipynb
```

## Setup Instructions
1. Create a virtual environment:
```bash
python -m venv venv
# Windows
.\venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Data Analysis

### Data Preprocessing
- Handled missing values in Age, Embarked, and Cabin columns
- Dropped irrelevant features (Cabin, Name, Ticket)
- Encoded categorical variables (Sex, Embarked)
- Standardized numerical features (Age, Fare)

### Feature Engineering
- Created binary features from categorical variables
- Applied standard scaling to numerical features
- Split data into training and validation sets

## Machine Learning Models Implemented
1. Random Forest Classifier
2. Logistic Regression
3. Decision Tree Classifier
4. Gradient Boosting Classifier
5. XGBoost Classifier

### Model Performance Metrics
Each model was evaluated using:
- Accuracy
- ROC-AUC Score
- Precision
- Recall
- F1-Score
- Confusion Matrix

### Hyperparameter Tuning
- Used GridSearchCV for each model
- Optimized parameters include:
  - n_estimators
  - max_depth
  - learning_rate
  - min_samples_split
  - etc.

## Results
Random Forest Classifier emerged as the best performing model with:
- Highest ROC-AUC score
- Best overall accuracy
- Most balanced precision-recall trade-off

## Dependencies
- numpy >= 1.21.0
- pandas >= 1.3.0
- scikit-learn >= 0.24.0
- matplotlib >= 3.4.0
- seaborn >= 0.11.0
- xgboost >= 1.4.0

## Usage
1. Open `titanic_ml_model.ipynb` in Jupyter Notebook or VS Code
2. Run all cells sequentially
3. Final predictions are saved in `gender_submission_by_random-forest.csv`

## Author
Sai Ruthvik Munugoti

## License
This project is licensed under the MIT License
