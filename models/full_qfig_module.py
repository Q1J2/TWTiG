import numpy as np
import math
from scipy.signal import find_peaks
from scipy.stats import linregress
import cvxpy as cp

class QFIG:
    def __init__(self, slope, intercept, start_t, end_t):
        self.slope = slope
        self.intercept = intercept
        self.start_t = int(start_t)
        self.end_t = int(end_t)
        self.size = int(end_t - start_t + 1)
        self.N = self.size

    def midpoint(self):

        return (self.start_t + self.end_t) // 2

    def predict(self, t):

        return self.slope * t + self.intercept

    def center_line(self, t):

        return self.predict(t)
def l1_trend_filter(y, lam):
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
            if abs(segment[j]) > local_threshold:
                idx = start + j
                if idx not in additional_peaks:
                    additional_peaks.append(idx)
    return np.array(additional_peaks)


def fit_granule(t, y):
    if len(t) < 2 or len(y) < 2:
        print(f"Warning: t or y is empty, skipping this granule.")
        return None, None
    try:
        slope, intercept, *_ = linregress(t, y)
    except:
        slope, intercept = None, None

    if slope is None or intercept is None:
        print(f"Warning: slope or intercept is None, skipping this granule.")
        return None, None
    return slope, intercept


def fit_qfigs_from_peaks(x_denoised, peaks):
    granules = []
    for i in range(len(peaks) - 1):
        start = peaks[i]
        end = peaks[i + 1]
        t_seg = np.arange(start, end + 1)
        y_seg = x_denoised[start:end + 1]
        if len(t_seg) < 2:
            continue
        slope, intercept = np.polyfit(t_seg, y_seg, 1)

        granules.append(QFIG(slope, intercept, start, end))
    return granules


def l1_hausdorff(q1: QFIG, q2: QFIG) -> float:

    t_min = min(q1.start_t, q2.start_t)
    t_max = max(q1.end_t, q2.end_t)
    total_length = t_max - t_min

    t_rel = np.linspace(0, total_length, num=1000)

    y1 = q1.predict(t_rel + (q1.start_t - t_min))

    y2 = q2.predict(t_rel + (q2.start_t - t_min))

    split_point = min(q1.end_t - t_min, q2.end_t - t_min)

    split_idx = np.argmin(np.abs(t_rel - split_point))

    if split_idx == 0:
        split_idx = 1
    elif split_idx == len(t_rel) - 1:
        split_idx = len(t_rel) - 2

    t_part1 = t_rel[:split_idx]
    y1_part1 = y1[:split_idx]
    y2_part1 = y2[:split_idx]

    t_part2 = t_rel[split_idx:]
    y1_part2 = y1[split_idx:]
    y2_part2 = y2[split_idx:]

    diff1 = np.abs(y1_part1 - y2_part1)
    area1 = np.trapz(diff1, t_part1)

    diff2 = np.abs(y1_part2 - y2_part2)
    area2 = np.trapz(diff2, t_part2)

    return area1 + area2


def d_prime(g1: QFIG, g2: QFIG) -> float:
    return l1_hausdorff(g1, g2)


def v_p(P):
    return 1 / (1 + math.exp(-max(d_prime(P[0], P[1]), d_prime(P[1], P[2]))))


def v_n(P):
    return 1 / (1 + math.exp(-min(d_prime(P[0], P[1]), d_prime(P[1], P[2]))))


def compute_adjacent_distances(qfigs):
    distances = []
    for i in range(len(qfigs) - 1):
        d = l1_hausdorff(qfigs[i], qfigs[i + 1])
        distances.append(d)
    return distances


def merge_granules(granules, t, y, alpha=0.7, beta=0.5):
    POS, BND, NEG = [], [], []
    pos_count, bnd_count, neg_count = 0, 0, 0

    def calculate_priority(granule_trio):
        return sum(granule.size for granule in granule_trio) + 1

    def find_min_priority_trio(granules):
        min_pri = float('inf')
        min_index = -1
        for i in range(len(granules) - 2):
            trio = granules[i:i + 3]
            pri = calculate_priority(trio)
            if pri < min_pri:
                min_pri = pri
                min_index = i
        return min_index

    while len(granules) >= 3:
        i = find_min_priority_trio(granules)
        if i == -1:
            break

        P = granules[i:i + 3]
        vp = v_p(P)
        vn = v_n(P)

        if vp <= beta:
            start_t = P[0].start_t
            end_t = P[2].end_t
            idx = (t >= start_t) & (t <= end_t)
            if np.sum(idx) < 2:
                break
            k, b = fit_granule(t[idx], y[idx])
            if k is None or b is None:
                break
            merged_granule = QFIG(k, b, start_t, end_t)
            POS.append(merged_granule)
            granules = granules[:i] + [merged_granule] + granules[i + 3:]
            continue

        elif vn >= alpha:
            NEG.append(P)
            for gr in P:
                gr.priority_blocked = True
            break

        else:
            BND.append(P)
            mid = P[1]
            mid_point = int((mid.start_t + mid.end_t) / 2)
            mid_point = max(mid.start_t + 1, min(mid.end_t - 1, mid_point))

            left_idx = (t >= mid.start_t) & (t <= mid_point)
            right_idx = (t > mid_point) & (t <= mid.end_t)
            if np.sum(left_idx) < 2 or np.sum(right_idx) < 2:
                break

            k_l, b_l = fit_granule(t[left_idx], y[left_idx])
            k_r, b_r = fit_granule(t[right_idx], y[right_idx])
            if None in (k_l, b_l, k_r, b_r):
                break
            g_l = QFIG(k_l, b_l, mid.start_t, mid_point)
            g_r = QFIG(k_r, b_r, mid_point + 1, mid.end_t)

            left_merge_start = P[0].start_t
            left_merge_end = g_l.end_t
            left_idx = (t >= left_merge_start) & (t <= left_merge_end)
            merged_l = None
            if np.sum(left_idx) >= 2:
                k_merge_l, b_merge_l = fit_granule(t[left_idx], y[left_idx])
                if k_merge_l is not None:
                    merged_l = QFIG(k_merge_l, b_merge_l, left_merge_start, left_merge_end)

            right_merge_start = g_r.start_t
            right_merge_end = P[2].end_t
            right_idx = (t >= right_merge_start) & (t <= right_merge_end)
            merged_r = None
            if np.sum(right_idx) >= 2:
                k_merge_r, b_merge_r = fit_granule(t[right_idx], y[right_idx])
                if k_merge_r is not None:
                    merged_r = QFIG(k_merge_r, b_merge_r, right_merge_start, right_merge_end)

            if merged_l and merged_r:
                granules = granules[:i] + [merged_l, merged_r] + granules[i + 3:]
                continue
            else:
                break
    return POS, BND, NEG, granules, pos_count, bnd_count, neg_count
