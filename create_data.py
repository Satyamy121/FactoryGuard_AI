import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Create 1000 rows of fake sensor data
rows = 1000
data = {
    'timestamp': [datetime.now() - timedelta(hours=i) for i in range(rows)],
    'machine_id': np.random.randint(1, 10, rows),
    'vibration': np.random.uniform(0.5, 5.0, rows),
    'temperature': np.random.uniform(50, 100, rows),
    'pressure': np.random.uniform(10, 50, rows),
    # Most are 0 (no failure), some are 1 (failure)
    'failure': np.random.choice([0, 1], size=rows, p=[0.98, 0.02]) 
}

df = pd.DataFrame(data)
df.sort_values('timestamp', inplace=True)
df.to_csv('sensor_data.csv', index=False)
print("Step 1 Complete: sensor_data.csv created!")