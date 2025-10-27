import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import LabelEncoder

def create_sequences(data, labels, window=60):
    X_seq, y_seq = [], []
    for i in range(len(data) - window):
        X_seq.append(data[i:i+window])
        y_seq.append(labels[i+window])
    return np.array(X_seq), np.array(y_seq)

def train_lstm_forecast(X, y):
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_seq, y_seq = create_sequences(X.values, y_encoded)

    model = Sequential([
        LSTM(64, input_shape=(X_seq.shape[1], X_seq.shape[2])),
        Dense(3, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_seq, y_seq, epochs=15, batch_size=32)

    return model
