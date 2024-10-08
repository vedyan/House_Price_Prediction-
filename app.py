from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import os
from sklearn.linear_model import LinearRegression
app = Flask(__name__)
model_path = r'house_price_model.pkl'
with open(model_path, 'rb') as f:
    model = pickle.load(f)
@app.route('/')
def home():
    return render_template("index.html")
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    try:
        # Extract features from the request
        total_sqft = data['Total_sqft']
        bedrooms = data['Bedrooms']
        bathrooms = data['Bathrooms']
        location_rural = data['Location_Rural']
        location_suburban = data['Location_Suburban']
        location_urban = data['Location_Urban']
        features = np.array([total_sqft, bedrooms, bathrooms, location_rural, location_suburban, location_urban]).reshape(1, -1)
        prediction = model.predict(features)
        formatted_prediction = f"${prediction[0]:.2f}"
        return jsonify({'predicted_price': formatted_prediction})
    except Exception as e:
        return jsonify({'error': str(e)})
if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=port)
