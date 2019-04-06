import numpy as np
import pandas
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plot
from sklearn.metrics import r2_score

"""
    Linear regression example
    With Kaggle dataset https://www.kaggle.com/budincsevity/szeged-weather
"""

DATASET_PATH = 'data.csv'
RANGE = 0.1

# Read scv
print('Reading data ...')
df = pandas.read_csv(DATASET_PATH, usecols=['Temperature (C)', 'Humidity'])
df.columns = ['hum', 'temp']

# Print the firt data in dataset
head = df.head()
print(head)

# Remove all rows with nan values
df = df.dropna()

# Devide data for features and labels
labels = df.hum.values
features = df.temp.values.reshape((-1, 1))

# Get train and test set
x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=RANGE)

# Define the model
model = LinearRegression()

# Fit the model
model.fit(x_train, y_train)

# Get predictions
train_pred = model.predict(x_train)
test_pred = model.predict(x_test)
