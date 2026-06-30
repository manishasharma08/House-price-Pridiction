# HOUSE PRICE PREDICTION USING LINEAR REGRESSION

## 1. Introduction

House price prediction is an important application of Machine Learning in the real estate industry. The main objective of this project is to predict the price of a house based on various features such as area, number of bedrooms, and age of the house. This helps buyers, sellers, and real estate agents make informed decisions.

In this project, we use Linear Regression, a supervised machine learning algorithm, to predict house prices based on historical data.

---

## 2. Objective

The main objectives of this project are:

* To understand the concept of supervised learning.
* To implement Linear Regression for prediction.
* To analyze how different factors affect house prices.
* To build a predictive model for estimating house prices.

---

## 3. Technology Used

* Programming Language: Python
* Libraries Used:

  * Pandas (for data handling)
  * NumPy (for numerical operations)
  * Matplotlib (for visualization)
  * Scikit-learn (for machine learning model)

---

## 4. Dataset Description

The dataset contains the following features:

* Area (in square feet)
* Number of Bedrooms
* Age of the House
* Price (Target Variable)

Sample Data:

| Area | Bedrooms | Age | Price  |
| ---- | -------- | --- | ------ |
| 1000 | 2        | 5   | 300000 |
| 1500 | 3        | 10  | 500000 |

---

## 5. Methodology

### Step 1: Data Collection

The dataset is collected in CSV format.

### Step 2: Data Preprocessing

* Handling missing values (if any)
* Selecting features and target variable

### Step 3: Splitting Dataset

The dataset is divided into:

* Training Data (80%)
* Testing Data (20%)

### Step 4: Model Training

Linear Regression model is trained using training data.

### Step 5: Prediction

The model predicts house prices using test data.

### Step 6: Evaluation

Model performance is evaluated using Mean Squared Error.

---

## 6. Algorithm Used: Linear Regression

Linear Regression is used to establish a relationship between dependent and independent variables.

Mathematical Equation:
Price = b₀ + b₁X₁ + b₂X₂ + b₃X₃

Where:

* X₁ = Area
* X₂ = Bedrooms
* X₃ = Age
* b₀ = Intercept
* b₁, b₂, b₃ = Coefficients

---

## 7. Implementation (Code)

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load dataset
data = pd.read_csv("house_data.csv")

# Features and target
X = data[['Area', 'Bedrooms', 'Age']]
y = data['Price']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Model training
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction
predictions = model.predict(X_test)

# Evaluation
error = mean_squared_error(y_test, predictions)
print("Mean Squared Error:", error)

# Visualization
plt.scatter(y_test, predictions)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Prices")
plt.show()

# Custom prediction
sample = [[1200, 3, 5]]
predicted_price = model.predict(sample)
print("Predicted Price:", predicted_price)
```

---

## 8. Results

The model successfully predicts house prices based on input features. The prediction accuracy depends on the quality and size of the dataset. The graph between actual and predicted values shows a linear relationship.

---

## 9. Advantages

* Simple and easy to implement
* Fast computation
* Good for small datasets
* Easy to interpret results

---

## 10. Limitations

* Works only for linear relationships
* Sensitive to outliers
* Accuracy depends on data quality

---

## 11. Conclusion

This project demonstrates how Linear Regression can be used to predict house prices effectively. It provides a basic understanding of machine learning concepts and their real-world applications.

---

## 12. Future Scope

* Use advanced models (Random Forest, XGBoost)
* Add more features like location, amenities
* Deploy as a web application
* Improve accuracy using large datasets
