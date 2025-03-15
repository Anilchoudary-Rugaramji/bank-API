from flask import Flask, request, jsonify
import joblib
import pandas as pd
import logging
import os

# Configure logging
logging.basicConfig(filename="api.log", level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")

# Dynamically set the model path (model is in the same directory as this script)
model_path = os.path.join(os.path.dirname(
    __file__), "model.pkl")

# Load trained model
model = joblib.load(model_path)

# Initialize Flask app
app = Flask(__name__)


@app.route("/predict", methods=["POST"])
def predict():
    """Fraud detection API"""
    try:
        # Get JSON data
        data = request.get_json()

        # Validate input
        required_features = ["V1", "V2", "V3",
                             "Amount"]  # Adjust based on dataset
        if not all(feature in data for feature in required_features):
            logging.warning("Validation failed: Missing required features")
            return jsonify({"error": "Missing required features"}), 400

        # Convert input to DataFrame
        df = pd.DataFrame([data])

        # Make prediction
        prediction = model.predict(df)[0]
        logging.info(f"Prediction successful: {prediction}")

        return jsonify({"fraud_prediction": int(prediction)}), 200

    except Exception as e:
        logging.error(f"Error: {str(e)}")
        return jsonify({"error": "Internal Server Error"}), 500


if __name__ == "__main__":
    from waitress import serve  # Use waitress for Windows/Linux deployment
    serve(app, host="0.0.0.0", port=5000)
