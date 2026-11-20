# Kaggle Titanic Survival Prediction

A tabular machine learning project that predicts passenger survival and exports a Kaggle submission file. It covers exploratory analysis, feature engineering, model comparison, and batch prediction.

The [executed notebook](notebook/executed-main.ipynb) records **82.38% mean accuracy** for the soft-voting ensemble over five stratified folds. Feature engineering was applied before cross-validation, so treat this as an exploratory estimate. The notebook reports local validation results.

## Prepare the data

Download `train.csv` and `test.csv` from the [Kaggle Titanic competition](https://www.kaggle.com/competitions/titanic/data) and put them in a `data/` directory at the repository root. Also download `gender_submission.csv` to the same directory if you want to rerun the notebook.

## Train and generate a submission

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install numpy pandas scikit-learn joblib
python scripts/train.py
python scripts/predict.py
```

The training script prints cross-validation accuracy and saves the model and feature list under `scripts/`. Prediction generates `data/submission.csv` with the `PassengerId` and `Survived` columns.

To rerun the [notebook](notebook/main.ipynb), add the plotting and notebook dependencies and launch Jupyter from the `notebook/` directory:

```bash
python -m pip install jupyter matplotlib seaborn
cd notebook
jupyter notebook main.ipynb
```

## Modeling approach

- Extract a title from each passenger's name and derive family size and solo travel indicators.
- Use ticket class and title to estimate missing ages. Derive cabin-deck and fare-group features where available.
- Combine random forest, gradient boosting, logistic regression, and a support vector machine using soft voting.
