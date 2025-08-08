import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import matplotlib.ticker as ticker

from full_qfig_module import (
    l1_trend_filter,
    detect_initial_peaks,
    detect_additional_peaks,
    fit_qfigs_from_peaks,
    compute_adjacent_distances,
    merge_granules,
    QFIG
)
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Times New Roman'
plt.rcParams['mathtext.it'] = 'Times New Roman:italic'
plt.rcParams['mathtext.bf'] = 'Times New Roman:bold'
plt.rcParams['font.size'] = 16

LAM = 1
INIT_THRESHOLD_RATIO = 0.3
ALPHA = 0.92
BETA_LIST = np.arange(0.7, 0.9 + 0.01, 0.02)
CSV_PATH = r"D:\LSTM\noisy\data3_noise_gauss.csv"

df = pd.read_csv(CSV_PATH, header=None)
y = df.iloc[:, 1].values
t = np.arange(len(y))

x_denoised = l1_trend_filter(y, LAM)
if x_denoised is None:
    raise RuntimeError("Error")
residuals = y - x_denoised

init_peaks = detect_initial_peaks(residuals, fixed_ratio=INIT_THRESHOLD_RATIO)
print(f"init: {init_peaks}")

all_peaks = np.sort(np.unique(np.concatenate([
    init_peaks,
    detect_additional_peaks(residuals, init_peaks)
])))
print(f"All: {all_peaks}")

valid_peaks = [p for p in all_peaks if 0 <= p < len(t)]
if len(valid_peaks) < len(all_peaks):
    print(f"Error {len(all_peaks) - len(valid_peaks)} ")
    all_peaks = np.array(valid_peaks)

qfigs = fit_qfigs_from_peaks(x_denoised, all_peaks)
print(f"init: {len(qfigs)}")

first_peak = all_peaks[0]
start_t = 0
end_t = first_peak - 1

if end_t >= start_t:
    t_segment = np.arange(start_t, end_t + 1)
    y_segment = x_denoised[t_segment]

    slope, intercept = np.polyfit(t_segment, y_segment, 1)

    head_qfig = QFIG(start_t=start_t, end_t=end_t, slope=slope, intercept=intercept)

    qfigs.insert(0, head_qfig)

    if len(all_peaks) > 0 and all_peaks[-1] < len(t) - 1:
        last_peak = all_peaks[-1]
        start_t = last_peak
        end_t = len(t) - 1
        if end_t >= start_t:
            t_segment = np.arange(start_t, end_t + 1)
            y_segment = x_denoised[t_segment]
            slope, intercept = np.polyfit(t_segment, y_segment, 1)
            tail_qfig = QFIG(slope, intercept, start_t, end_t)
            qfigs.append(tail_qfig)

for i, q in enumerate(qfigs[:5]):
    print(f"{i}: start_t={q.start_t}, end_t={q.end_t}, size={q.size}")

distances = compute_adjacent_distances(qfigs)
print(f" {distances}")


best_beta = None
best_mse = float('inf')
best_mae = float('inf')
best_r2 = -float('inf')
best_pred = None

for BETA in BETA_LIST:
    print(f"\n=====  BETA = {BETA:.2f} =====")

    valid_qfigs = []
    for q in qfigs:

        if (0 <= q.start_t < len(t) and
                0 <= q.end_t < len(t) and
                q.start_t <= q.end_t and
                (q.end_t - q.start_t + 1) >= 2):
            valid_qfigs.append(q)
        else:
            print(f"Error： [start_t={q.start_t}, end_t={q.end_t}]，len={len(t)}")

    if not valid_qfigs:
        print("Error")
        continue

    if best_pred is not None:

        print(f"\n BETA  {best_beta:.2f},MSE: {best_mse:.4f}, MAE: {best_mae:.4f}, R²: {best_r2:.4f}")

        plt.figure(figsize=(12, 6))
        plt.plot(t, y, label='Actual data', alpha=0.6)
        plt.plot(t, best_pred, label=f'Best Fitting (BETA={best_beta:.2f})', linewidth=2, color='orange')
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(False)
        plt.tight_layout()
        plt.savefig("results3_gauss_1/best_beta_fitting.svg")
        plt.close()
    else:
        print("❌ ")

    POS, BND, NEG, merged_qfigs, pos_count, bnd_count, neg_count = merge_granules(
        valid_qfigs, t, x_denoised, alpha=ALPHA, beta=BETA
    )

    valid_merged_qfigs = []
    for q in merged_qfigs:
        if (0 <= q.start_t < len(t) and
                0 <= q.end_t < len(t) and
                q.start_t <= q.end_t and
                (q.end_t - q.start_t + 1) >= 2):
            valid_merged_qfigs.append(q)
        else:
            print(f"Error: [start_t={q.start_t}, end_t={q.end_t}]")

    if not valid_merged_qfigs:
        print(f"Error:  (BETA={BETA:.2f})")
        continue

    plt.figure(figsize=(12, 6))
    plt.plot(t, y, label='Actual data', alpha=0.4)
    plt.plot(t, x_denoised, label=r'$L_1$ trend', linewidth=2)

    for q in valid_merged_qfigs:
        t_fit = np.arange(q.start_t, q.end_t + 1)
        y_fit = q.slope * t_fit + q.intercept
        plt.plot(t_fit, y_fit, linewidth=2)

    plt.title(f"QFIG Segmentation Result (BETA={BETA:.2f})")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(False)
    plt.xlim(left=0)
    plt.tight_layout()

    output_dir = f"results3_gauss_1/beta_{BETA:.2f}"
    os.makedirs(output_dir, exist_ok=True)

    plt.savefig(os.path.join(output_dir, "qfig_plot.svg"))
    plt.close()

    qfig_data = []
    for q in valid_merged_qfigs:
        qfig_data.append({
            'start_t': q.start_t,
            'end_t': q.end_t,
            'slope': q.slope,
            'intercept': q.intercept,
            'span': q.size
        })

    if qfig_data:
        pd.DataFrame(qfig_data).to_csv(os.path.join(output_dir, "merged_qfigs.csv"), index=False)
    else:
        print(f"Error: (BETA={BETA:.2f})")

    y_pred = np.zeros_like(y, dtype=float)
    for q in valid_merged_qfigs:
        t_fit = np.arange(q.start_t, q.end_t + 1)
        y_fit = q.slope * t_fit + q.intercept
        y_pred[t_fit] = y_fit

    import matplotlib.ticker as ticker

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(t, y, label='Actual data', alpha=0.6)
    ax.plot(t, y_pred, label='Fitting data', linewidth=2, color='orange')

    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5))

    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.legend(loc='upper right')
    ax.grid(False)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    ticks = ax.get_yticks()
    ticks_no_zero = [tick for tick in ticks if tick != 0]
    ax.set_yticks(ticks_no_zero)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "prediction_vs_truth.svg"))
    plt.close()

    mse = mean_squared_error(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    print(f" BETA={BETA:.2f}, MSE={mse:.4f}, MAE={mae:.4f}, R²={r2:.4f}")
    if mse < best_mse:
        best_mse = mse
        best_mae = mae
        best_r2 = r2
        best_beta = BETA
        best_pred = y_pred.copy()

    pd.DataFrame({'distance': distances}).to_csv(os.path.join(output_dir, "qfig_distances.csv"), index=False)


