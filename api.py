from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib

# Initialize the Flask app and enable CORS
app = Flask(__name__)
CORS(app)

# Function to load the saved models
def load_models():
    scaler = joblib.load(r"C:\Users\abhin\Desktop\PREDICTION\PREDICTION\Random Forest Regressor\scaler_rf.pkl")
    best_rf = joblib.load(r"C:\Users\abhin\Desktop\PREDICTION\PREDICTION\Random Forest Regressor\best_rf_model.pkl")
    return scaler, best_rf

# Load models when the server starts
scaler, best_rf = load_models()

# API endpoint for predictions
@app.route('/predict_rf', methods=['POST'])
def predict_api():
    try:
        # Parse JSON input
        data = request.get_json()
        
        # Extract input features
        input_features = [[
            data["CastingTemp"],
            data["CoolingWaterTemp"],
            data["CastingSpeed"],
            data["EntryTempRollingMill"],
            data["EmulsionTemp"],
            data["EmulsionPressure"],
            data["EmulsionConcentration"],
            data["RodQuenchWaterPressure"]
        ]]
        
        # Scale input features
        input_features_scaled = scaler.transform(input_features)
        
        # Make predictions
        predictions = best_rf.predict(input_features_scaled)
        result = {
            "UTS (Ultimate Tensile Strength)": round(predictions[0][0], 2),
            "Elongation": round(predictions[0][1], 2),
            "Conductivity": round(predictions[0][2], 2)
        }
        
        # Return predictions as JSON
        return jsonify({"status": "success", "predictions": result}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

# Start the Flask server
if __name__ == '__main__':
    app.run(debug=True)
