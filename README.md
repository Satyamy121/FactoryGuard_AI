# FactoryGuard AI: Predictive Maintenance Engine (IoT)

## Project Overview
A production-grade MLOps system designed to predict failures in 500 robotic arms 24 hours in advance.

## Key Technical Features
- **Feature Engineering:** Implemented 6-hour rolling means and 12-hour vibration standard deviations.
- **Class Imbalance:** Utilized `class_weight='balanced'` in Random Forest to handle rare failure events (<1% of data).
- **Inference:** Flask API providing real-time predictions with <50ms latency.

## Setup Instructions
1. Install dependencies: `pip install -r requirements.txt`
2. Generate base data: `python create_data.py`
3. Execute feature pipeline: `python features.py`
4. Train production model: `python model_train.py`
5. Launch API: `python app.py`

## API Testing
Use the following command to test:
`curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d "{\"vibration\": 4.8, \"temperature\": 98.2, \"pressure\": 45.0, \"temp_rolling_6h\": 95.5, \"vibration_std_12h\": 1.5, \"temp_lag_1\": 97.0}"`