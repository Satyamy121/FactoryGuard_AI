import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

# 1. Load the features you just created
df = pd.read_csv('refined_features.csv')

# 2. Select the features the model should learn from
# We exclude 'failure' (because that's the answer) and non-numeric columns
X = df[['vibration', 'temperature', 'pressure', 'temp_rolling_6h', 'vibration_std_12h', 'temp_lag_1']]
y = df['failure']

# 3. Split data: 80% for training, 20% for testing the model's accuracy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Initialize the model
# We use 'balanced' weights because machine failures are very rare 
model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)

# 5. Train the model
print("Training the FactoryGuard AI engine...")
model.fit(X_train, y_train)

# 6. Save the model as a production artifact [cite: 53]
joblib.dump(model, 'factory_model.pkl')

print("Step 3 Complete: 'factory_model.pkl' created!")