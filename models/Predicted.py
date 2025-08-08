import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


def prepare_data(granule_file, ts_file):

    granules = pd.read_csv(granule_file)
    required_columns = ['slope', 'intercept', 'span', 'start_t', 'end_t']
    for col in required_columns:
        if col not in granules.columns:
            raise ValueError(f"Error: {col}")

    ts_data = pd.read_csv(ts_file)

    if 'time' not in ts_data.columns:

        if 't' in ts_data.columns:
            ts_data = ts_data.rename(columns={'t': 'time'})
        elif 'timestamp' in ts_data.columns:
            ts_data = ts_data.rename(columns={'timestamp': 'time'})
        else:

            ts_data['time'] = ts_data.index
            print("Error: time")



    X = granules[['slope', 'intercept', 'span']].values
    y = granules[['slope', 'intercept', 'span']].values

    return X, y, granules, ts_data

def create_sequences(X, y, window_size=5):
    X_seq, y_seq = [], []
    for i in range(len(X) - window_size):
        X_seq.append(X[i:i + window_size])
        y_seq.append(y[i + window_size])
    return np.array(X_seq), np.array(y_seq)

def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(3)
    ])

    model.compile(optimizer='adam',
                  loss='mse',
                  metrics=['mae'])

    return model

def generate_ts_from_granules(granules):

    ts_data = []

    for i, row in granules.iterrows():
        slope = row['slope']
        intercept = row['intercept']
        start_t = row['start_t']
        end_t = row['end_t']
        span = int(end_t - start_t)

        for t in range(span):
            time_point = start_t + t
            value = slope * t + intercept
            ts_data.append([time_point, value])

    return pd.DataFrame(ts_data, columns=['time', 'value'])

# 5. 主流程
def main():

    GRANULE_FILE = r'D:\pycharm\study\LSTM\our model\results3_1\beta_0.70\merged_qfigs.csv'
    TS_FILE = r"D:\LSTM\noisy\data3_noise_gauss.csv"
    WINDOW_SIZE = 2
    TEST_SIZE = 0.2
    EPOCHS = 150
    BATCH_SIZE = 32

    X, y, granules, ts_data = prepare_data(GRANULE_FILE, TS_FILE)

    X_seq, y_seq = create_sequences(X, y, WINDOW_SIZE)
    print(f" X_seq={X_seq.shape}, y_seq={y_seq.shape}")

    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X_seq.reshape(-1, X_seq.shape[2])).reshape(X_seq.shape)

    scaler_y = StandardScaler()
    y_scaled = scaler_y.fit_transform(y_seq)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_scaled, test_size=TEST_SIZE, shuffle=False
    )

    model = build_lstm_model((WINDOW_SIZE, X_train.shape[2]))

    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.2,
        callbacks=[early_stop],
        verbose=1
    )

    test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
    print(f"\ntest - MSE: {test_loss:.4f}, MAE: {test_mae:.4f}")

    y_test_pred_scaled = model.predict(X_test)
    y_test_pred = scaler_y.inverse_transform(y_test_pred_scaled)
    y_test_orig = scaler_y.inverse_transform(y_test)

    pred_granules_test = granules.iloc[-len(y_test_pred):].copy()
    pred_granules_test[['pred_slope', 'pred_intercept', 'pred_span']] = y_test_pred
    pred_granules_test['pred_span'] = pred_granules_test['pred_span'].round().clip(lower=1)
    pred_granules_test.to_csv('predicted_granules_test_only_data3_gauss0.2.csv', index=False)

    pred_granules_test[['slope', 'intercept', 'span']] = pred_granules_test[['pred_slope', 'pred_intercept', 'pred_span']]
    predicted_ts = generate_ts_from_granules(pred_granules_test)

    original_ts = generate_ts_from_granules(pred_granules_test.assign(
        slope=y_test_orig[:, 0],
        intercept=y_test_orig[:, 1],
        span=y_test_orig[:, 2]
    ))

    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    # 斜率比较
    axes[0].plot(granules['start_t'], granules['slope'], 'bo-', label='True Slope')
    axes[0].plot(pred_granules_test['start_t'], pred_granules_test['pred_slope'], 'rx-', label='Predicted Slope')
    axes[0].set_title('Slope Comparison')
    axes[0].set_ylabel('Slope')
    axes[0].legend()
    axes[0].grid(True)

    # 截距比较
    axes[1].plot(granules['start_t'], granules['intercept'], 'bo-', label='True Intercept')
    axes[1].plot(pred_granules_test['start_t'], pred_granules_test['pred_intercept'], 'rx-', label='Predicted Intercept')
    axes[1].set_title('Intercept Comparison')
    axes[1].set_ylabel('Intercept')
    axes[1].legend()
    axes[1].grid(True)

    # 跨度比较
    axes[2].plot(granules['start_t'], granules['span'], 'bo-', label='True Span')
    axes[2].plot(pred_granules_test['start_t'], pred_granules_test['pred_span'], 'rx-', label='Predicted Span')
    axes[2].set_title('Span Comparison')
    axes[2].set_xlabel('Start Time')
    axes[2].set_ylabel('Span')
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()
    plt.savefig('granule_parameters_comparison.png', dpi=300)
    plt.show()


if __name__ == "__main__":
    main()