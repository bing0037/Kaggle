import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# Read the data
train = pd.read_csv('data/train_age_revised.csv')

# pull data into target (y) and predictors (X)
train_y = train.Survived
predictor_cols = [
    "Pclass",
    #"Sex",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
]

# Create training predictors data
train_X = train[predictor_cols]

my_model = RandomForestRegressor()
my_model.fit(train_X, train_y)


# Read the test data
test = pd.read_csv('data/test_age_revised.csv')

print(test.isnull().sum())

# Treat the test data in the same way as training data. In this case, pull same columns.
test_X = test[predictor_cols]
# Use the model to make predictions
predicted_survive = my_model.predict(test_X)
# We will look at the predicted prices to ensure we have something sensible.
print(predicted_survive)


my_submission = pd.DataFrame({'Id': test.PassengerId, 'SalePrice': predicted_survive})
# you could use any filename. We choose submission here
my_submission.to_csv('output/submission.csv', index=False)

# Then, submit!
