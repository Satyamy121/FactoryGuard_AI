import pandas as pd

def prepare_features(file_path):
    df = pd.read_csv(file_path)
    
    # 1. Advanced Rolling Window Statistics 
    # Calculate average temperature of the last 6 readings
    df['temp_rolling_6h'] = df['temperature'].rolling(window=6).mean()
    
    # Calculate standard deviation of vibration (checking for shaky movements)
    df['vibration_std_12h'] = df['vibration'].rolling(window=12).std()
    
    # 2. Lag Features (t-1) 
    # What was the temperature 1 hour ago?
    df['temp_lag_1'] = df['temperature'].shift(1)
    
    # Drop rows with empty values created by rolling windows
    df.dropna(inplace=True)
    
    return df

if __name__ == "__main__":
    processed_df = prepare_features('sensor_data.csv')
    processed_df.to_csv('refined_features.csv', index=False)
    print("Step 2 Complete: Features engineered and saved!")