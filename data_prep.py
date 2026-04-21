import pandas as pd
import numpy as np

# 1. Load your dataset (use a CSV with sensor columns like 'temp', 'vibration')
df = pd.read_csv('sensor_data.csv') 

# 2. Create Rolling Mean (Average of last 6 and 12 hours)
# This fulfills the 'Advanced Rolling Window' requirement 
df['temp_rolling_6h'] = df['temp'].rolling(window=6).mean()
df['vibration_rolling_12h'] = df['vibration'].rolling(window=12).mean()

# 3. Create Lag Features (What was the value 1 hour ago?)
# Essential for time-series classification 
df['temp_lag_1'] = df['temp'].shift(1)

# 4. Handle Missing Values
# Rolling windows create 'NaN' at the start; we must drop them
df.dropna(inplace=True)

print("Features created successfully!")
print(df.head())