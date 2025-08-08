
import numpy as np
import matplotlib.pyplot as plt
import time
import pywt
import ruptures as rpt
import cvxpy as cp
from scipy.signal import argrelextrema
import pandas as pd
from sklearn.linear_model import LinearRegression
from scipy.signal import find_peaks
import os

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Times New Roman'
plt.rcParams['mathtext.it'] = 'Times New Roman:italic'
plt.rcParams['mathtext.bf'] = 'Times New Roman:bold'
plt.rcParams['font.size'] = 16

def generate_smooth_time_series(n_segments=8, segment_length=150, noise_std=0.2, noise_type='gaussian',
                                mode='step', seed=0, trend_strength=0.3, pulse_prob=0.01):
    np.random.seed(seed)
    true_cps = []
    segments = []
    patterns = ['linear', 'quadratic', 'sinusoidal', 'exponential']

    for i in range(n_segments):
        pattern = np.random.choice(patterns) if i > 0 else 'linear'

        x = np.linspace(0, 1, segment_length)
        if pattern == 'linear':
            base = np.linspace(0, 1, segment_length)
        elif pattern == 'quadratic':
            base = x ** 2
        elif pattern == 'sinusoidal':
            base = np.sin(2 * np.pi * 3 * x)
        elif pattern == 'exponential':
            base = np.exp(2 * x) - 1

        base = base * (1 + i * trend_strength)

        if noise_type == 'gaussian':
            noise = np.random.normal(0, noise_std, segment_length)
        elif noise_type == 'impulse':
            noise = np.random.normal(0, noise_std * 0.5, segment_length)
            impulse_idx = np.random.choice(segment_length, int(segment_length * pulse_prob), replace=False)
            noise[impulse_idx] += np.random.normal(0, noise_std * 5, len(impulse_idx))
        elif noise_type == 'autocorrelated':
            noise = np.zeros(segment_length)
            noise[0] = np.random.normal(0, noise_std)
            for j in range(1, segment_length):
                noise[j] = 0.7 * noise[j - 1] + np.random.normal(0, noise_std * 0.6)

        segment = base + noise

        if i > 0 and mode == 'ramp' and np.random.rand() > 0.5:
            transition_len = min(20, segment_length // 4)
            prev_segment = segments[-1][-transition_len:]
            segment[:transition_len] = np.linspace(prev_segment[-1], segment[transition_len], transition_len)

        segments.append(segment)
        if i > 0:
            true_cps.append(i * segment_length)

    ts = np.concatenate(segments)

    global_trend = np.linspace(0, 0.5, len(ts))
    ts += global_trend

    return ts, true_cps


def detect_true_cps(ts, order=3, distance=30):
    maxima = argrelextrema(ts, np.greater, order=order)[0]
    minima = argrelextrema(ts, np.less, order=order)[0]
    cps = sorted(np.concatenate([maxima, minima]))
    filtered = []
    last_cp = -distance
    for cp in cps:
        if cp - last_cp >= distance:
            filtered.append(cp)
            last_cp = cp
    return filtered

def Global_variance_change_detection(ts, window_size=50, threshold=1.5):

    change_points = []
    global_var = np.var(ts)
    for i in range(window_size, len(ts) - window_size):
        local_var = np.var(ts[i - window_size:i + window_size])
        if local_var / global_var > threshold:
            change_points.append(i)
    return None, change_points


def l1_trend_filter(ts, lam=20):
    n = len(ts)
    x = cp.Variable(n)
    obj = cp.Minimize(0.5 * cp.sum_squares(x - ts) + lam * cp.norm1(cp.diff(x, 2)))
    prob = cp.Problem(obj)
    prob.solve(solver=cp.ECOS)
    trend = x.value
    residual = ts - trend
    change_points = np.where(np.abs(np.diff(residual)) > np.std(residual) * 2)[0].tolist()
    return trend, change_points


def pelt_method(ts, penalty=10):
    model = rpt.Pelt(model="l2").fit(ts)
    change_points = model.predict(pen=penalty)
    return None, change_points[:-1]


def wavelet_segmentation(ts, wavelet='db4', level=3):
    coeffs = pywt.wavedec(ts, wavelet, level=level)
    detail = coeffs[1]
    peaks = np.where(np.abs(detail) > np.std(detail))[0].tolist()
    return None, peaks

def l1_trend_filtering(y, lam):
    n = len(y)
    x = cp.Variable(n)

    D = np.zeros((n - 2, n))
    for i in range(n - 2):
        D[i, i] = 1
        D[i, i + 1] = -2
        D[i, i + 2] = 1

    objective = cp.Minimize(0.5 * cp.sum_squares(y - x) + lam * cp.norm1(D @ x))
    problem = cp.Problem(objective)

    try:
        problem.solve(solver=cp.ECOS, verbose=False)
    except Exception as e:
        print(f"λ = {lam} Error：{e}")
        return None

    return x.value if x.value is not None else None


def detect_initial_peaks(residuals, fixed_ratio=0.3):

    threshold = fixed_ratio * np.max(np.abs(residuals))
    peaks, _ = find_peaks(np.abs(residuals), height=threshold)
    return peaks


def detect_additional_peaks(residuals, seg_indices):

    additional_peaks = []
    for i in range(len(seg_indices) - 1):
        start = seg_indices[i]
        end = seg_indices[i + 1]
        segment = residuals[start:end]
        local_threshold = np.std(segment)
        for j in range(len(segment)):
            if abs(segment[j]) > 2* local_threshold:
                idx = start + j
                if idx not in additional_peaks:
                    additional_peaks.append(idx)
    return np.array(additional_peaks)


def improved_l1_trend_filter(ts, lam=20):
    trend = l1_trend_filtering(ts, lam)
    if trend is None:
        return None, []

    residuals = ts - trend
    init_peaks = detect_initial_peaks(residuals)
    seg_indices = [0] + sorted(init_peaks.tolist()) + [len(ts)]
    additional_peaks = detect_additional_peaks(residuals, seg_indices)

    final_peaks = np.unique(np.concatenate([init_peaks, additional_peaks]))
    final_peaks.sort()

    return trend, final_peaks.tolist()

def matching_f1_score(true_cps, pred_cps, tolerance=10):
    true_cps = np.array(true_cps)
    pred_cps = np.array(pred_cps)
    matched_true = set()
    matched_pred = set()
    for i, t in enumerate(true_cps):
        for j, p in enumerate(pred_cps):
            if abs(t - p) <= tolerance and j not in matched_pred:
                matched_true.add(i)
                matched_pred.add(j)
                break
    TP = len(matched_true)
    FP = len(pred_cps) - TP
    FN = len(true_cps) - TP
    precision = TP / (TP + FP + 1e-8)
    recall = TP / (TP + FN + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return precision, recall, f1


def compute_mae_rmse(true_cps, pred_cps):
    if not true_cps or not pred_cps:
        return np.nan, np.nan
    offsets = []
    for t in true_cps:
        closest_pred = min(pred_cps, key=lambda p: abs(p - t))
        offsets.append(abs(t - closest_pred))
    mae = np.mean(offsets)
    rmse = np.sqrt(np.mean(np.square(offsets)))
    return mae, rmse


def evaluate_performance(true_cps, pred_cps, tolerance=20):
    precision, recall, f1 = matching_f1_score(true_cps, pred_cps, tolerance)
    mae, rmse = compute_mae_rmse(true_cps, pred_cps)
    return precision, recall, f1, mae, rmse

def run_multiple_experiments(n_experiments=5):
    methods = [
        ("Global Variance Change Detection", Global_variance_change_detection, {"window_size": 50, "threshold":1.5}),
        ('$L_1$-trend Filtering', l1_trend_filter, {'lam': 30}),
        ('Improved $L_1$-trend Filter', improved_l1_trend_filter, {'lam': 30}),
        ('PELT', pelt_method, {'penalty': 10}),
        ('Wavelet', wavelet_segmentation, {'wavelet': 'db4', 'level': 3}),
    ]

    results = {name: {'precision': [], 'recall': [], 'f1': [], 'mae': [], 'rmse': [], 'time': []}
               for name, _, _ in methods}
    first_run_results = None

    print(f"Running {n_experiments} experiments...")
    for exp_idx in range(n_experiments):
        seed = exp_idx
        ts, true_cps = generate_smooth_time_series(
            n_segments=8,
            segment_length=150,
            noise_std=0.2,
            seed=seed,
            noise_type='gaussian',
            mode='step'
        )

        if exp_idx == 0:
            first_run_results = {
                'ts': ts,
                'true_cps': true_cps,
                'method_results': {}
            }

        for name, method, kwargs in methods:
            start = time.time()
            try:
                _, pred_cps = method(ts, **kwargs)
            except Exception as e:
                print(f" {name} Error: {e}")
                pred_cps = []
            elapsed = time.time() - start
            precision, recall, f1, mae, rmse = evaluate_performance(true_cps, pred_cps)

            results[name]['precision'].append(precision)
            results[name]['recall'].append(recall)
            results[name]['f1'].append(f1)
            results[name]['mae'].append(mae)
            results[name]['rmse'].append(rmse)
            results[name]['time'].append(elapsed)

            if exp_idx == 0:
                first_run_results['method_results'][name] = pred_cps

    avg_results = {}
    for name in results:
        avg_results[name] = {
            'precision': np.nanmean(results[name]['precision']),
            'recall': np.nanmean(results[name]['recall']),
            'f1': np.nanmean(results[name]['f1']),
            'mae': np.nanmean(results[name]['mae']),
            'rmse': np.nanmean(results[name]['rmse']),
            'time': np.nanmean(results[name]['time'])
        }

    print("\nAverage Performance over {} experiments:".format(n_experiments))
    print(f"{'Method':<25} {'Precision':<10} {'Recall':<10} {'F1':<10} {'MAE':<10} {'RMSE':<10} {'Time(s)':<10}")
    for name, data in avg_results.items():
        print(f"{name:<25} {data['precision']:<10.3f} {data['recall']:<10.3f} {data['f1']:<10.3f} "
              f"{data['mae']:<10.2f} {data['rmse']:<10.2f} {data['time']:<10.3f}")

    if first_run_results:
        visualize_results(first_run_results)

    return avg_results


def visualize_results(results):
    ts = results['ts']
    true_cps = results['true_cps']
    method_results = results['method_results']

    plt.figure(figsize=(15, 10))
    plt.subplot(len(method_results) + 1, 1, 1)
    plt.plot(ts, label='Time Series')
    for cp in true_cps:
        plt.axvline(cp, color='red', linestyle='--', alpha=0.6, label='True CP' if cp == true_cps[0] else "")
    plt.title("Synthetic Time Series with True Change Points", fontsize=14)
    plt.legend(fontsize=10)

    for i, (method_name, pred_cps) in enumerate(method_results.items(), 2):
        plt.subplot(len(method_results) + 1, 1, i)
        plt.plot(ts, label='Time Series')
        for cp in true_cps:
            plt.axvline(cp, color='red', linestyle='--', alpha=0.3)
        for cp in pred_cps:
            plt.axvline(cp, color='blue', linestyle='-', alpha=0.6, label='Detected CP' if cp == pred_cps[0] else "")
        plt.title(f"{method_name} ", fontsize=14)
        plt.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig("change_point_detection_results.svg", format='svg')
    plt.savefig("change_point_detection_results.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    results = run_multiple_experiments(n_experiments=5)
    df = pd.DataFrame(results).T
    df.to_csv("change_point_detection_results.csv")
    print("\nResults saved to 'change_point_detection_results.csv'")