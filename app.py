from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# 1. Load the trained model artifact (The AI Brain)
try:
    model = joblib.load('factory_model.pkl')
except Exception as e:
    print(f"Error loading model: {e}")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 2. Get JSON data from the request
        data = request.get_json()
        
        # 3. Convert input to DataFrame for the model
        input_df = pd.DataFrame([data])
        
        # 4. Get the Initial AI Prediction
        prediction = int(model.predict(input_df)[0])
        probability = float(model.predict_proba(input_df)[0][1])
        
        # 5. INDUSTRIAL SAFETY OVERRIDES (Hybrid Logic)
        # In real factories, some values are dangerous regardless of what the AI thinks.
        # We force a 'Failure' if sensors hit critical danger zones.
        is_critical = False
        if data.get('temperature', 0) > 150 or data.get('vibration', 0) > 9.0:
            prediction = 1
            probability = max(probability, 0.98) # Boost confidence for extreme values
            is_critical = True
        
        # 6. Return Professional JSON Response
        return jsonify({
            'status': 'Success',
            'prediction': 'Failure Detected' if prediction == 1 else 'Normal Operation',
            'failure_probability': round(probability, 4),
            'recommendation': 'CRITICAL: Shut Down & Schedule Maintenance' if is_critical else 
                             ('Schedule Maintenance' if prediction == 1 else 'Continue Monitoring'),
            'detection_type': 'Safety Override' if is_critical else 'AI Model'
        })
    
    except Exception as e:
        return jsonify({'status': 'Error', 'message': str(e)}), 400

if __name__ == '__main__':
    # Start the production-ready API
    print("FactoryGuard AI API is live on http://127.0.0.1:5000")
    app.run(debug=True, port=5000)