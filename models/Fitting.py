import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Times New Roman'
plt.rcParams['mathtext.it'] = 'Times New Roman:italic'
plt.rcParams['mathtext.bf'] = 'Times New Roman:bold'
plt.rcParams['font.size'] = 16

pred_granules = pd.read_csv(r'D:\pycharm\study\LSTM\our model\predicted_granules_test_only_data4_0.2.csv')

df_true = pd.read_csv(r"D:\LSTM\data4.csv", header=None)
true_series = df_true.iloc[:, 1].values if df_true.shape[1] >= 2 else df_true.iloc[:, 0].values

predicted_y = []
true_y = []
time_axis = []

for _, row in pred_granules.iterrows():
    start_t = int(row['start_t'])
    span = int(row['span'])
    slope = row['slope']
    intercept = row['intercept']
    end_t = start_t + span - 1

    if end_t >= len(true_series):
        end_t = len(true_series) - 1

    t_fit = np.arange(start_t, end_t + 1)
    y_fit = slope * t_fit + intercept

    # 累加真实值和预测值
    predicted_y.extend(y_fit)
    true_y.extend(true_series[t_fit])
    time_axis.extend(t_fit)

predicted_y = np.array(predicted_y)
true_y = np.array(true_y)
time_axis = np.array(time_axis)

plt.figure(figsize=(8, 6))
plt.plot(time_axis, true_y, label='True Value', color='darkblue', linewidth=2)
plt.plot(time_axis, predicted_y, label='Predicted Value', color='red', linewidth=2)
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.grid(False)
# plt.ylim(bottom=0)
plt.tight_layout()
plt.savefig('predicted_vs_true_data4_0.2_real.svg', format='svg')
plt.savefig('predicted_vs_true_data4_0.2_real.png', dpi=300)
plt.show()

rmse = np.sqrt(mean_squared_error(true_y, predicted_y))
mae = mean_absolute_error(true_y, predicted_y)
mape = np.mean(np.abs((true_y - predicted_y) / (true_y + 1e-8))) * 100
r2 = r2_score(true_y, predicted_y)

print("\n📊 Evaluation Metrics (on fitted time steps only):")
print(f"RMSE = {rmse:.4f}")
print(f"MAE  = {mae:.4f}")
print(f"MAPE = {mape:.2f}%")
print(f"R²   = {r2:.4f}")

