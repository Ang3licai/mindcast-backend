import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def preprocess_data(wesad_path, survey_path):
    df = pd.read_csv(wesad_path)
    features = ['EDA', 'HR', 'ACC_x', 'ACC_y', 'ACC_z']
    df = df[features]

    scaler = MinMaxScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=features)
    df_scaled['timestamp'] = pd.date_range(start='2025-01-01', periods=len(df_scaled), freq='S')
    df_scaled.set_index('timestamp', inplace=True)

    survey = pd.read_csv(survey_path)
    df_combined = df_scaled.merge(survey, on='timestamp', how='left')

    return df_combined

