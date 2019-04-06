import numpy 
import pandas
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plot


"""
 Sklearn linear regression example with kaggle dataset 
 Dataset: https://www.kaggle.com/sohier/calcofi
"""

DATASET_PATH = 'bottle.csv'
RANGE = 0.4

# Read scv
print('Reading data ...')
df = pandas.read_csv(DATASET_PATH, usecols=['Salnty', 'T_degC'])

# Print the firt data in dataset
head = df.head()
print(head)

# Remove all rows with nan values
df = df.dropna()

# Devide data for features and labels
features = df.Salnty.values.reshape((-1, 1))
labels = df.T_degC.values

# Get train and test set
x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=RANGE)

# Define the model
model = LinearRegression()

# Fit the model
model.fit(x_train, y_train)

# Get predictions
train_pred = model.predict(x_train)
test_pred = model.predict(x_test)