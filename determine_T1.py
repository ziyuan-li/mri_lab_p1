import os
import numpy as np
import matplotlib.pyplot as plt
import pydicom
import cv2
from scipy.optimize import curve_fit


# Configuration
indices  = [2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14, 15, 16]
base_dir = r"C:\Li\Projects\tum_courses\mri_lab_p1\data\Group 2 Gadulinium T1\1.2.826.0.1.3680043.8.1276.259246159.268.1000"

ORDER_LABELS = [2, 5, 7, 6, 4, 1, 3]
ORDER_NAMES  = ["ROI1 (green)", "ROI2 (orange)", "ROI3 (cyan)", "ROI4 (purple)", "ROI5 (blue)", "ROI6 (red)", "ROI7 (gold)"]
ORDER_COLORS = ["green", "orange", "cyan", "purple", "blue", "red", "gold"]

DO_PIXEL_FIT_MASKED   = True
DO_PIXEL_FIT_ALLPIXEL = True

PALETTE = np.array([
    [0,   0,   0],
    [230, 25,  75],   # red
    [60,  180, 75],   # green
    [255, 225, 25],   # gold
    [0,   130, 200],  # blue
    [245, 130, 48],   # orange
    [145, 30,  180],  # purple
    [70,  240, 240],  # cyan
], dtype=np.uint8)


# Function
def invrec_model(TI, A, B, T1):
    TI = np.asarray(TI, float)
    T1 = np.maximum(T1, 1e-6)
    return np.abs(A + B * np.exp(-TI / T1))

def r2_score(y, yhat):
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    m = np.isfinite(y) & np.isfinite(yhat)
    if m.sum() < 3: return np.nan
    ss_res = np.sum((y[m] - yhat[m])**2)
    ss_tot = np.sum((y[m] - y[m].mean())**2)
    return np.nan if ss_tot == 0 else 1.0 - ss_res / ss_tot

def estimate_t10_from_null(TI, S):
    TI, S = np.asarray(TI, float), np.asarray(S, float)
    if np.all(~np.isfinite(S)): return 800.0
    k = np.nanargmin(S)
    idx = np.unique(np.clip([k-1, k, k+1], 0, S.size-1))
    TIc, Sc = TI[idx], S[idx]
    try:
        A = np.vstack([TIc**2, TIc, np.ones_like(TIc)]).T
        a, b, _ = np.linalg.lstsq(A, Sc, rcond=None)[0]
        TImin = -b/(2*a) if a != 0 else TI[k]
    except Exception:
        TImin = TI[k]
    TImin = float(np.clip(TImin, TI.min(), TI.max()))
    return float(np.clip(-TImin / np.log(0.5), 1.0, 20000.0))

def fit_ir(TI, S):
    TI, S = np.asarray(TI, float), np.asarray(S, float)
    m = np.isfinite(TI) & np.isfinite(S)
    TI, S = TI[m], S[m]
    if TI.size < 4 or np.nanvar(S) == 0:
        return (np.nan, np.nan, np.nan, np.nan)

    A0  = float(np.nanmean(S[-max(3, S.size//3):]))
    B0  = -2.0 * A0
    T10 = estimate_t10_from_null(TI, S)

    Smax = float(np.nanmax(S))
    Sbd  = max(1.0, 5.0 * Smax)
    bounds = ([-Sbd, -Sbd, 1.0], [Sbd, Sbd, 20000.0])

    try:
        popt, _ = curve_fit(invrec_model, TI, S, p0=[A0, B0, T10],
                            bounds=bounds, method='trf', maxfev=20000)
        A, B, T1 = map(float, popt)
        return (A, B, T1, r2_score(S, invrec_model(TI, A, B, T1)))
    except Exception:
        return (np.nan, np.nan, np.nan, np.nan)

def to_u8(img):
    img = img.astype(np.float32)
    mn, mx = img.min(), img.max()
    if mx <= mn: return np.zeros_like(img, dtype=np.uint8)
    return ((img - mn) / (mx - mn) * 255).astype(np.uint8)


# Read DICOM
TI_list, imgs, sum_image = [], [], None
for i in indices:
    ddir = os.path.join(base_dir, f"1.2.826.0.1.3680043.8.1276.259246159.268.1000.{i}", "Dicoms")
    ds = pydicom.dcmread(os.path.join(ddir, "0001.dcm"))
    TI_list.append(ds.get("InversionTime", np.nan))
    arr = ds.pixel_array.astype(np.float32)

    plt.figure()
    plt.imshow(arr, cmap="gray")
    plt.axis("off")
    plt.savefig("./fig/{}.png".format(i))
    plt.show()

    imgs.append(arr)
    if sum_image is None: sum_image = np.zeros_like(arr, dtype=np.float32)
    sum_image += arr

TI   = np.asarray(TI_list, float)
imgs = np.stack(imgs, axis=0)  # (N_TI, H, W)


# Segmentation
img8 = to_u8(sum_image)
_, bw  = cv2.threshold(img8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
bw    = cv2.morphologyEx(bw, cv2.MORPH_OPEN,  np.ones((3, 3), np.uint8))
bw    = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
_, labels = cv2.connectedComponents(bw)

labels_eroded = np.zeros_like(labels, dtype=np.int32)
ker = np.ones((3, 3), np.uint8)
for lab in range(1, labels.max() + 1):
    er = cv2.erode((labels == lab).astype(np.uint8), ker, iterations=1)
    labels_eroded[er.astype(bool)] = lab

palette_use = PALETTE[:max(1, min(labels_eroded.max() + 1, len(PALETTE)))]
color_img   = palette_use[labels_eroded]

plt.figure(figsize=(4, 4))
plt.imshow(sum_image, cmap="gray")
plt.axis("off")
plt.savefig("./fig/sum.png")
plt.show()

plt.figure(figsize=(4, 4))
plt.imshow(color_img)
plt.axis("off")
plt.savefig("./fig/label.png")
plt.show()

plt.figure(figsize=(4, 4))
plt.imshow(sum_image, cmap="gray")
plt.imshow(color_img, alpha=0.4)
plt.axis("off")
plt.savefig("./fig/sum_label.png")
plt.show()


# Fit
N = imgs.shape[0]
means = np.zeros((N, len(ORDER_LABELS)), float)
for j, lab in enumerate(ORDER_LABELS):
    m = (labels_eroded == lab)
    for t in range(N):
        vals = imgs[t][m]
        means[t, j] = np.nan if vals.size == 0 else float(vals.mean())

plt.figure(figsize=(9, 5))
for j, name in enumerate(ORDER_NAMES):
    plt.plot(TI, means[:, j], 'o-', label=name, color=ORDER_COLORS[j])
plt.xlabel("Inversion Time [ms]")
plt.ylabel("Avg. Intensity in ROI")
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("./fig/int_vs_TI.png")
plt.show()

plt.figure(figsize=(9, 5))
plt.plot(TI, means[:, 5], marker='o', linestyle='None', label="ROI6 (red)", color="Red")
plt.xlabel("Inversion Time [ms]")
plt.ylabel("Avg. Intensity in ROI")
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("./fig/int_vs_TI_example.png")
plt.show()

fit_results = []
plt.figure(figsize=(9, 5))
TI_dense = np.linspace(np.nanmin(TI), np.nanmax(TI), 400)

for j, (name, color) in enumerate(zip(ORDER_NAMES, ORDER_COLORS)):
    S = means[:, j]
    A, B, T1, R2 = fit_ir(TI, S)
    fit_results.append((name, A, B, T1, R2))
    plt.plot(TI, S, 'o', color=color, alpha=0.75, label=f"{name} data")
    if np.isfinite(T1):
        plt.plot(TI_dense, invrec_model(TI_dense, A, B, T1), '-', color=color,
                 label=f"{name} fit (T1≈{T1:.0f} ms)")

plt.xlabel("Inversion Time [ms]"); plt.ylabel("Avg. Intensity in ROI")
plt.grid(True, alpha=0.3); plt.legend(ncol=2, fontsize=9)
plt.tight_layout()
plt.savefig("./fig/roi_fits.png")
plt.show()

print("{:^15}  {:^10}  {:^10}  {:^10}  {:^10}".format("ROI", "T1(ms)", "A", "B", "R^2"))
for name, A, B, T1, R2 in sorted(fit_results, key=lambda x: (np.inf if not np.isfinite(x[3]) else x[3])):
    print("{:^15}  {:^10}  {:^10}  {:^10}  {:^10}".format(name, f"{T1:.2f}", f"{A:.2f}", f"{B:.2f}", f"{R2:.2f}"))

# Pixel-by-pixel T1 (foreground mask region)
if DO_PIXEL_FIT_MASKED:
    H, W = imgs.shape[1:]
    mask_bg  = (bw > 0)
    mean_t   = np.nanmean(imgs, axis=0)
    mask_fg  = mask_bg & (mean_t > np.percentile(mean_t[mask_bg], 5))

    T1_map = np.full((H, W), np.nan, np.float32)
    R2_map = np.full((H, W), np.nan, np.float32)

    for y in range(H):
        row_stack = imgs[:, y, :].astype(float)
        for x in range(W):
            if not mask_fg[y, x]: continue
            _, _, T1, R2 = fit_ir(TI, row_stack[:, x])
            if np.isfinite(T1): T1_map[y, x] = T1
            if np.isfinite(R2): R2_map[y, x] = R2

    T1_map[(T1_map < 1.0) | (T1_map > 20000.0)] = np.nan
    vmin, vmax = np.nanpercentile(T1_map, [5, 95])
    plt.figure(figsize=(6, 5))
    plt.imshow(T1_map, cmap='turbo', vmin=vmin, vmax=vmax)
    plt.colorbar(label='T1 [ms]')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig("./fig/t1_map1.png")
    plt.show()

# Pixel-by-pixel T1 (all pixels)
if DO_PIXEL_FIT_ALLPIXEL:
    H, W = imgs.shape[1], imgs.shape[2]
    T1_map_all = np.full((H, W), np.nan, dtype=np.float32)
    R2_map_all = np.full((H, W), np.nan, dtype=np.float32)

    for y in range(H):
        row_stack = imgs[:, y, :].astype(float)  # 形状: (N_TI, W)
        for x in range(W):
            _, _, T1hat, r2 = fit_ir(TI, row_stack[:, x])
            if np.isfinite(T1hat): T1_map_all[y, x] = T1hat
            if np.isfinite(r2):    R2_map_all[y, x] = r2

    T1_vis = T1_map_all.copy()
    T1_vis[(T1_vis < 1.0) | (T1_vis > 20000.0)] = np.nan
    vmin, vmax = np.nanpercentile(T1_vis, [5, 95])

    plt.figure(figsize=(6,5))
    plt.imshow(T1_vis, cmap='turbo', vmin=vmin, vmax=vmax)
    plt.colorbar(label='T1 [ms]')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig("./fig/t1_map2.png")
    plt.show()
