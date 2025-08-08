import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

csv_path = r"D:\LSTM\dataset1new.xlsx"
output_csv = r"D:\LSTM\data1_noise_impulse.csv"
output_svg = r'D:\LSTM\data1_comparison_impulse.svg'
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Times New Roman'
plt.rcParams['mathtext.it'] = 'Times New Roman:italic'
plt.rcParams['mathtext.bf'] = 'Times New Roman:bold'
plt.rcParams['font.size'] = 16

try:
    df = pd.read_excel(csv_path, header=None)
    t = df.iloc[:, 0].values
    y = df.iloc[:, 1].values
    if len(t) != len(y):
        print(f"Error: time({len(t)})and value({len(y)})")
        t = np.arange(len(y))

except Exception as e:

    exit()

def add_impulse_noise(y, ratio=0.01, amplitude_scale=1.5):
    y_noisy = y.copy()
    n = len(y)
    num_impulses = int(n * ratio)
    indices = np.random.choice(n, num_impulses, replace=False)

    max_val = np.max(y)
    impulses = amplitude_scale * max_val * (2 * np.random.rand(num_impulses) - 1)
    y_noisy[indices] = impulses
    return y_noisy

y_impulse = add_impulse_noise(y, ratio=0.01, amplitude_scale=1.5)

df_impulse = pd.DataFrame({
    'time': t,
    'value_impulse': y_impulse
})
df_impulse.to_csv(output_csv, index=False)

plt.figure(figsize=(8, 6))
plt.plot(t, y, label='Original', linewidth=2)
plt.plot(t, y_impulse, label='Noisy', linestyle='--', color='red', alpha=0.7)
plt.xlabel("Time")
plt.ylabel("Value")
# plt.title("Original vs Impulse Noisy Time Series")
plt.legend()
plt.grid(False)
plt.xlim(left=0)
plt.tight_layout()
plt.savefig(output_svg, format='svg')  
plt.show()
