# Predicting Death from AIDS Clinical Trials Group Study 175

This notebook is for practicing working with messy data and different machine learning models. This dataset has a lot of inbalanced classes, so it's good for practicing how to fix it.

## About the Data

The data for this notebook was taken from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/890/aids+clinical+trials+group+study+175). It contains data on AIDS patients health records.

## Goal

The goal is to predict whether or not the patient died within a certain window of time.

## Thought Process

#### Skewed Data & Outliers

There were 3 features that had skewed data, however I was having issues applying a log transformation to them. For the outliers, there were 138 rows with outliers. When I applied a natural log to skewed features and removed outliers, it made the predictions worse, so I left those in and didn't apply any logarithmic transformations.

#### Feature Class Imbalances & Importance

A lot of the categorical features had class imbalances, but I relied on variable importance to remove less important features.

### Multicollinearity

There were 2 columns that were multicollinear, which for what they were made sense. When I removed them, the cv score decreased and the predictions became worse.

### SMOTE for Target Class Imbalance

I applied SMOTE for synthetic data generation so that the target can have equal classes. Without it, it made the model results worse.

### Cross-Validation Model Selection

I tested logistic regression, xgboost, catboost, and lightgbm. After running many tests, lightgbm seemed to overall be the best, however with some results catboost ended up being better.

### Hyperparameter Tuning

I used 5-fold grid search cross validation to check hyperparameters for better results.

### Metrics Chosen

This was a classification problem, so I chose F1 score as my metric. I also compared it with a confusion matrix and a ROC/AUC curve so I would have different metrics to go off of.

### Results

The f1 score is 0.89, and the AUC is 0.93, which is not bad. However, because this is a medical problem, it is significantly better the higher the results are. I think this can be improved with slight feature engineering changes that I want to try in the future.

## What I Learned

I learned about the Synthetic Minority Oversampling Technique (SMOTE), and I also learned that it's not always the best. I'm not a fan of creating synthetic data, which is exactly what SMOTE does, but in this example SMOTE did help with model predictions a little bit. I also played around with different sci-kit learn scalers, so it helped me learn the differences between them (like standard scaler vs minmax scaler).

## Future Endeavors

I would like to figure out how to apply the logarithmic transformations and have it actually work/be impactful. If not this, it would be nice to try out different normalization techniques so there would be less skew in the data. Same with the outliers, anytime I tried something that should have worked, it just made the results worse. I would overall like to try different small feature engineering techniques so that the results could be even better.

# Kaggle!

Check out this notebook on [Kaggle](https://www.kaggle.com/code/coreymichaud/predicting-death-from-aids-clinical-trial).
