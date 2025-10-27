import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

def train_nn_classifier(data_path):
    df = pd.read_csv(data_path)
    features = ['EDA', 'HR', 'ACC_x', 'ACC_y', 'ACC_z']
    df = df[features]

    scaler = MinMaxScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=features)

    # Simulated stress scores (replace with real survey input)
    df_scaled['stress_score'] = pd.cut(df_scaled['EDA'], bins=[0, 0.3, 0.6, 1.0], labels=[0, 1, 2])

    X = df_scaled.drop(columns=['stress_score'])
    y = df_scaled['stress_score'].astype(int)

    model = Sequential([
        Dense(64, activation='relu', input_shape=(X.shape[1],)),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(3, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(X, y, epochs=20, batch_size=32)

    return model
