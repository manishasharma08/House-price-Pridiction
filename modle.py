# House Price Prediction using Linear Regression

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Step 1: Load dataset
data = pd.read_csv("house_data.csv")

# Step 2: Show first few rows
print("Dataset Preview:")
print(data.head())

# Step 3: Define features (X) and target (y)
X = data[['Area', 'Bedrooms', 'Age']]
y = data['Price']

# Step 4: Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Create and train model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 6: Predict on test data
predictions = model.predict(X_test)

# Step 7: Evaluate model
mse = mean_squared_error(y_test, predictions)
print("\nModel Evaluation:")
print("Mean Squared Error:", mse)

# Step 8: Compare actual vs predicted
comparison = pd.DataFrame({
    'Actual Price': y_test.values,
    'Predicted Price': predictions
})
print("\nActual vs Predicted:")
print(comparison)

# Step 9: Visualization
plt.scatter(y_test, predictions)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted House Prices")
plt.show()

# Step 10: Predict your own house price
print("\nCustom Prediction:")
area = float(input("Enter Area (sq ft): "))
bedrooms = int(input("Enter Number of Bedrooms: "))
age = int(input("Enter Age of House: "))

sample = [[area, bedrooms, age]]
predicted_price = model.predict(sample)

print("Predicted House Price:", predicted_price[0])