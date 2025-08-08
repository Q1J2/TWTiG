import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

csv_path = r"D:\LSTM\data3.xlsx"
output_csv = r"D:\LSTM\data3_noise_gauss.csv"
output_svg = r'D:\LSTM\data3_comparison.svg'

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

def add_gaussian_noise(y, std=100):

    y_float = y.astype(float)
    noise = np.random.normal(0, std, size=y_float.shape)
    return y_float + noise

y_noisy = add_gaussian_noise(y, std=100)

df_noisy = pd.DataFrame({
    'time': t,
    'value_noisy': y_noisy
})
df_noisy.to_csv(output_csv, index=False)

plt.savefig(output_svg, format='svg', bbox_inches='tight')
print(f"save to: {output_svg}")
plt.figure(figsize=(8, 6))
plt.plot(t, y, label='Original', linewidth=2)
plt.plot(t, y_noisy, label='Noisy', linestyle='--', linewidth=1.5, alpha=0.6)
plt.xlabel("Time")
plt.ylabel("Value")
plt.legend()
plt.grid(False)
plt.xlim(left=0)
# plt.ylim(bottom=0)
# yticks= plt.yticks()[0]
# yticklabels = ["" if abs(y) < 1e-8 else str(int(y)) if y.is_integer() else str(y) for y in yticks]
# plt.yticks(yticks, yticklabels)
plt.tight_layout()
plt.savefig(output_svg, format='svg')
plt.show()
