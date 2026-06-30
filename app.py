from flask import Flask, render_template, request
import pandas as pd
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

# Load and train model
data = pd.read_csv("house_data.csv")
X = data[['Area', 'Bedrooms', 'Age']]
y = data['Price']

model = LinearRegression()
model.fit(X, y)

@app.route('/')
def home():
    return render_template('house.html')

@app.route('/predict', methods=['POST'])
def predict():
    area = float(request.form['area'])
    bedrooms = int(request.form['bedrooms'])
    age = int(request.form['age'])

    prediction = model.predict([[area, bedrooms, age]])
    return render_template('house.html', result=prediction[0])

if __name__ == "__main__":
    app.run(debug=True)