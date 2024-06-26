# Codeclause-Intership
# Aim -
# Build a simple linear regression model to predict house prices based on features like
# the number of bedrooms and square footage

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

# Load and Explore the Data
df=pd.read_csv('Housing.csv')

# Explore the dataset
df.head()

df.info()

df.describe()

# Select features and target variable
X = df[['bedrooms', 'area']]
y = df['price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the model
model=LinearRegression()

# Train the model
model.fit(X_train, y_train)

y_pred=model.predict(X_test)

# Evaluate the model
m = mean_squared_error(y_test,y_pred)

r =model.score(X_test, y_test)

print(f'Mean Squared Error: {m}')
print(f'R-squared: {r}')

plt.scatter(y_test, y_pred)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs. Predicted Prices')
plt.show()






