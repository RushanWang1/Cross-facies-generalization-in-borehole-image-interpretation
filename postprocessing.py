import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize, remove_small_objects, closing, disk
from skimage.measure import label
from scipy.optimize import least_squares
from collections import deque
import cv2
from PIL import Image
import imageio
from scipy import ndimage
import mplstereonet

from utils import az_plunge_to_vec, rodrigues_rotation, plane_strike_dip_to_normal, normal_to_strike_dip, correct_apparent_to_true
from plotstereonetfromcsv import plot_stereonet_from_csv, _TAB10
# -------------------------
# Skeletonization utilities
# -------------------------
def preprocess_mask(mask, min_area=0, closing_radius=2):
    """Binarize, remove tiny objects, and close small gaps."""
    m = mask.astype(bool)
    if min_area > 0:
        m = remove_small_objects(m, min_size=min_area)
    if closing_radius > 0:
        m = closing(m, disk(closing_radius))
    return m

def compute_skeleton(mask):
    """Return boolean skeleton (single-pixel width)."""
    return skeletonize(mask)

def find_neighbors8(pt, shape):
    """Return 8-neighbors coordinates for a pixel pt=(r,c)."""
    r, c = pt
    H, W = shape
    for dr in (-1,0,1):
        for dc in (-1,0,1):
            if dr==0 and dc==0:
                continue
            rr, cc = r+dr, c+dc
            if 0 <= rr < H and 0 <= cc < W:
                yield (rr, cc)

def endpoints_of_skel(skel):
    """Return endpoint pixels of skeleton."""
    coords = np.argwhere(skel)
    s = set(map(tuple, coords))
    endpoints = []
    for (r,c) in coords:
        deg = 0
        for nb in find_neighbors8((r,c), skel.shape):
            if nb in s:
                deg += 1
        if deg == 1:
            endpoints.append((r,c))
    return endpoints

def prune_skeleton(skel, max_spur_length=15):
    """Remove short spurs from skeleton."""
    sk = skel.copy().astype(bool)
    H,W = sk.shape
    s = set(map(tuple, np.argwhere(sk)))
    changed = True
    while changed:
        changed = False
        endpoints = endpoints_of_skel(sk)
        for ep in endpoints:
            path = [ep]
            cur = ep
            prev = None
            for step in range(max_spur_length):
                nbs = [nb for nb in find_neighbors8(cur, (H,W)) if sk[nb]]
                if prev is not None:
                    nbs = [nb for nb in nbs if nb != prev]
                if len(nbs) == 0:
                    break
                if len(nbs) > 1:
                    break
                prev = cur
                cur = nbs[0]
                path.append(cur)
                deg = sum(1 for nb in find_neighbors8(cur, (H,W)) if sk[nb])
                if deg > 2:
                    break
            if len(path) <= max_spur_length:
                end_pt = path[-1]
                deg_end = sum(1 for nb in find_neighbors8(end_pt, (H,W)) if sk[nb])
                if deg_end <= 2:
                    for p in path:
                        sk[p] = False
                        changed = True
    return sk

def skeleton_to_longest_path(skel_component):
    """Extract ordered longest path from one skeleton component."""
    coords = np.argwhere(skel_component)
    if len(coords)==0:
        return np.array([]), np.array([])
    H,W = skel_component.shape
    s = set(map(tuple, coords))

    def bfs_from(start):
        q = deque([start])
        parents = {start: None}
        last = start
        while q:
            u = q.popleft()
            last = u
            for nb in find_neighbors8(u, (H,W)):
                if nb in s and nb not in parents:
                    parents[nb] = u
                    q.append(nb)
        return last, parents

    endpoints = endpoints_of_skel(skel_component)
    if len(endpoints) == 0:
        start = tuple(coords[0])
        far, _ = bfs_from(start)
        far2, parents2 = bfs_from(far)
        path = []
        cur = far2
        while cur is not None:
            path.append(cur)
            cur = parents2[cur]
        path = path[::-1]
    else:
        best_path = []
        for ep in endpoints:
            far, parents = bfs_from(ep)
            path = []
            cur = far
            while cur is not None:
                path.append(cur)
                cur = parents[cur]
            if len(path) > len(best_path):
                best_path = path
        path = best_path[::-1]

    path = np.array(path)
    ys = path[:,0].astype(float)
    xs = path[:,1].astype(float)
    return xs, ys

def extract_centerlines(mask, min_component_area=30, spur_length=15):
    """Return list of (xs, ys) centerlines from mask."""
    m = preprocess_mask(mask, min_area=min_component_area, closing_radius=2)
    print("After preprocessing, mask has {} pixels.".format(np.sum(m)))
    sk = compute_skeleton(m)
    sk = prune_skeleton(sk, max_spur_length=spur_length)
    lab = label(sk, connectivity=2)
    centerlines = []
    for region_label in np.unique(lab):
        if region_label == 0:
            continue
        comp = (lab == region_label)
        xs, ys = skeleton_to_longest_path(comp)
        if len(xs) > 5:
            centerlines.append((xs, ys))
    return centerlines, sk

# -------------------------
# Sinusoid fitting utilities
# -------------------------
def sinusoid_model(params, x):
    A, k, phi, c = params
    return A * np.sin(k * x + phi) + c

def residuals(params, x, y):
    return y - sinusoid_model(params, x)

def fit_sinusoid_least_squares(x, y, initial_params):
    res = least_squares(residuals, initial_params, loss='soft_l1', f_scale=1.0,
                        args=(x, y), max_nfev=2000)
    return res.x, res.cost

def fit_quality(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return 1 - ss_res/ss_tot  # R^2


def ransac_sinusoid(xs, ys, W,
                    n_iter=200,
                    sample_size=6,
                    inlier_tol=3.0,
                    min_inliers=50):
    best = None
    n_points = len(xs)
    if n_points < sample_size:
        return None

    for it in range(n_iter):
        idx = np.random.choice(n_points, size=sample_size, replace=False)
        x_s, y_s = xs[idx], ys[idx]
        A0 = (np.nanmax(ys) - np.nanmin(ys)) / 4.0 + 1e-3
        c0 = np.mean(y_s)
        phi0 = 0.0
        k0 = 2*np.pi / W   # assume ~1 cycle across width
        init = [A0, k0, phi0, c0]
        try:
            params, _ = fit_sinusoid_least_squares(x_s, y_s, init)
        except Exception:
            continue
        y_pred_all = sinusoid_model(params, xs)
        resid = np.abs(ys - y_pred_all)
        inlier_mask = resid < inlier_tol
        n_inliers = np.sum(inlier_mask)
        if n_inliers >= min_inliers:
            score = n_inliers - np.mean(resid[inlier_mask])
            if best is None or score > best['score']:
                best = {'params': params.copy(),
                        'inlier_mask': inlier_mask.copy(),
                        'n_inliers': n_inliers,
                        'score': score}
    if best is None:
        return None
    x_in = xs[best['inlier_mask']]
    y_in = ys[best['inlier_mask']]
    final_params, final_cost = fit_sinusoid_least_squares(x_in, y_in, best['params'])
    best['params'] = final_params
    best['cost'] = final_cost
    return best

import numpy as np
from sklearn.cluster import DBSCAN
from scipy.optimize import least_squares

def fit_multiple_sine_safe_multipercluster(boolean_array,
                           wavelength=1080,
                           dbscan_eps=3,
                           dbscan_min_samples=30,
                           min_points_for_fit=100,
                           phase_tol=0.6,
                           A_tol_ratio=0.4,
                           D_tol=30,
                           rmse_thresh=10.0,
                           verbose=True):
    """
    Fit multiple sine curves to a boolean skeleton image.
    Handles intersecting curves by splitting DBSCAN clusters into sub-traces.
    """

    # --- Extract coordinates ---
    ys, xs = np.where(boolean_array)
    if len(xs) == 0:
        return []

    points = np.stack([xs, ys], axis=1)

    # --- Cluster points with DBSCAN ---
    db = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples).fit(points)
    labels = db.labels_
    unique_labels = [lab for lab in np.unique(labels) if lab != -1]

    # Fixed B from wavelength
    B = 2 * np.pi / wavelength

    def model(x, A, C, D):
        return A * np.sin(B * x + C) + D

    def residuals(params, x, y):
        return model(x, *params) - y

    # --- Helper: split cluster into separate traces if multiple y-values per x ---
    from collections import defaultdict

    def split_cluster(xs_k, ys_k):
        """
        Split cluster into multiple traces based on unique x values.
        This avoids collapsing intersecting sinusoids into one trace.
        """
        x_to_ys = defaultdict(list)
        for x, y in zip(xs_k, ys_k):
            x_to_ys[x].append(y)

        uniq_x = sorted(x_to_ys.keys())
        curves = []
        max_branches = max(len(ylist) for ylist in x_to_ys.values())

        for branch in range(max_branches):
            xs_branch, ys_branch = [], []
            for x in uniq_x:
                ylist = sorted(x_to_ys[x])
                if branch < len(ylist):
                    xs_branch.append(x)
                    ys_branch.append(ylist[branch])
            if len(xs_branch) > 10:
                curves.append((np.array(xs_branch), np.array(ys_branch)))

        return curves

    cluster_fits = []

    # --- Fit one sine per sub-trace ---
    for k in unique_labels:
        mask = labels == k
        pts = points[mask]
        xs_k = pts[:, 0].astype(int)
        ys_k = pts[:, 1].astype(float)

        # Split into separate traces
        curves = split_cluster(xs_k, ys_k)

        for xs_curve, ys_curve in curves:
            uniq_x, inv = np.unique(xs_curve, return_inverse=True)
            med_y = np.array([np.median(ys_curve[inv == i]) for i in range(len(uniq_x))])
            if len(uniq_x) < min_points_for_fit:
                continue

            # initial guess
            A0 = (med_y.max() - med_y.min()) / 2.0 if med_y.max() != med_y.min() else 5.0
            C0 = 0.0
            D0 = np.median(med_y)

            try:
                res = least_squares(residuals, x0=[A0, C0, D0],
                                    args=(uniq_x, med_y),
                                    loss='soft_l1', max_nfev=2000)
            except Exception as e:
                if verbose:
                    print(f"Cluster {k} fit failed: {e}")
                continue

            A_hat, C_hat, D_hat = res.x
            pred = model(uniq_x, A_hat, C_hat, D_hat)
            rmse = np.sqrt(np.mean((pred - med_y) ** 2))
            if rmse > rmse_thresh or abs(A_hat) < 5:
                continue

            # Normalize phase
            C_hat = ((C_hat + np.pi) % (2 * np.pi)) - np.pi

            cluster_fits.append({
                'label': k,
                'params': (A_hat, C_hat, D_hat),
                'n_points': len(uniq_x),
                'x_range': (uniq_x.min(), uniq_x.max()),
                'x_unique': uniq_x,
                'y_unique': med_y
            })

    if verbose:
        print(f"Fitted {len(cluster_fits)} initial cluster sinusoids.")

    # --- Controlled merging ---
    merged = []
    used = set()

    def param_distance(p1, p2):
        A1, C1, D1 = p1
        A2, C2, D2 = p2
        dC = abs(((C1 - C2 + np.pi) % (2 * np.pi)) - np.pi)
        dA = abs(A1 - A2) / max(abs(A1), abs(A2), 1e-6)
        dD = abs(D1 - D2)
        return dC, dA, dD

    for i, ci in enumerate(cluster_fits):
        if i in used:
            continue
        group = [i]
        used.add(i)
        for j in range(i + 1, len(cluster_fits)):
            if j in used:
                continue
            dC, dA, dD = param_distance(ci['params'], cluster_fits[j]['params'])
            if (dC <= phase_tol) and (dA <= A_tol_ratio) and (dD <= D_tol):
                group.append(j)
                used.add(j)

        # combine group
        total_pts = sum(cluster_fits[g]['n_points'] for g in group)
        A_comb = sum(cluster_fits[g]['params'][0] * cluster_fits[g]['n_points'] for g in group) / total_pts
        sin_sum = sum(np.sin(cluster_fits[g]['params'][1]) * cluster_fits[g]['n_points'] for g in group)
        cos_sum = sum(np.cos(cluster_fits[g]['params'][1]) * cluster_fits[g]['n_points'] for g in group)
        C_comb = np.arctan2(sin_sum, cos_sum)
        D_comb = sum(cluster_fits[g]['params'][2] * cluster_fits[g]['n_points'] for g in group) / total_pts

        # recompute RMSE
        xs_all = np.concatenate([cluster_fits[g]['x_unique'] for g in group])
        ys_all = np.concatenate([cluster_fits[g]['y_unique'] for g in group])
        preds_all = A_comb * np.sin(B * xs_all + C_comb) + D_comb
        rmse_all = np.sqrt(np.mean((preds_all - ys_all) ** 2))

        merged.append({
            'params': (A_comb, B, C_comb, D_comb),
            'rmse': rmse_all,
            'n_points': total_pts,
            'x_range': (np.min(xs_all), np.max(xs_all))
        })

    if verbose:
        print(f"Merged into {len(merged)} global traces.")

    # --- Reconstruct curves over full image width ---
    width = boolean_array.shape[1]
    R = width / (2 * np.pi)  # borehole radius in pixels
    results = []
    for m in merged:
        A, B_val, C, D = m['params'] 
        dip_rad = np.arctan(np.abs(A) / R)
        dip_deg = np.degrees(dip_rad)

        # --- Dip direction (apparent) from sinusoid phase robustly ---
        # Use trough position (sin = -1): (B*x + C) = -pi/2 + 2*pi*k
        # Solve for x, then wrap to [0, width)
        x_trough = (-np.pi/2 - C) / B_val
        x_trough = x_trough % width

        # Convert x (pixels) -> azimuth (degrees).
        # Set az0_deg to match your image convention:
        # - if x=0 corresponds to North and increases clockwise: az0_deg = 0
        az0_deg = 0.0
        dip_dir_deg = (az0_deg + x_trough * 360.0 / width) % 360.0

        # Optional: if your image x increases counterclockwise, use this instead:
        # dip_dir_deg = (az0_deg - x_trough * 360.0 / width) % 360.0
        if A < 0:
            dip_dir_deg = (dip_dir_deg + 180.0) % 360.0

        azimuth_deg = dip_dir_deg

        # Add to dict
        m['dip_deg'] = dip_deg
        m['azimuth_deg'] = azimuth_deg

        x_full = np.arange(width)
        y_full = A * np.sin(B_val * x_full + C) + D
        results.append({
            'params': m['params'],
            'rmse': m['rmse'],
            'n_points': m['n_points'],
            'x_range': m['x_range'],
            'x_full': x_full,
            'y_full': y_full,
            'dip_deg': m['dip_deg'],
            'azimuth_deg': m['azimuth_deg']
        })

    return results

import numpy as np
import matplotlib.pyplot as plt
import mplstereonet

# ---- Conversion helpers ----
def sine_params_to_plane(A, C, wavelength):
    """
    Convert sine parameters (A, C) from borehole image fit into (dip, dip_dir).
    A : amplitude (pixels)
    C : phase (radians)
    wavelength : circumference of borehole image (pixels)

    Returns:
        dip_deg, dip_dir_deg
    """
    R = wavelength / (2.0 * np.pi)   # borehole radius (pixels)
    A_abs = float(np.abs(A))

    dip_rad = np.arctan2(A_abs, R)
    dip_deg = np.degrees(dip_rad)

    # azimuth relation: alpha = (3π/2 - C) mod 2π
    dip_dir_deg = (270.0 - np.degrees(C)) % 360.0
    return dip_deg, dip_dir_deg


def fitted_traces_to_planes(fit_results, min_support=20,
                            dip_key='dip_deg', dipdir_key='azimuth_deg'):
    """
    Extract (dip, dipdir) from the output of fit_multiple_sine_safe_multipercluster().
    Assumes each element already contains dip_deg and azimuth_deg (dip direction).
    """
    planes = []
    for fr in fit_results:
        support = fr.get('n_points', 0)
        if support < min_support:
            continue

        dip = float(fr[dip_key])
        dipdir = float(fr[dipdir_key])  # this is dip direction (0..360 from North)

        planes.append((dip, dipdir))
    return planes

def fit_curves_per_class_from_label_png(
    pred_map_path,
    class_ids=None,                 # e.g. [1,2,3,4,5,6,7] or None = auto
    min_pixels=200,                 # skip tiny masks
    **fit_kwargs                    # forwarded to your fit_multiple_sine_safe_multipercluster
):
    img = Image.open(pred_map_path).convert("L")
    labels = np.array(img).astype(np.int32)

    # If your PNG is stored as 0..255 (e.g. 255 for class 1), uncomment this scaling:
    # labels = (labels / 255.0 * 8).round().astype(np.int32)

    uniq = np.unique(labels)
    print("Unique labels:", uniq)

    if class_ids is None:
        class_ids = [c for c in uniq.tolist() if c != 0]  # skip background

    all_results = []   # will contain dicts with added "class"
    results_by_class = {}

    for cls in class_ids:
        mask = (labels == cls)
        n_pix = int(mask.sum())
        print(f"Class {cls}: pixels={n_pix}")
        if n_pix < min_pixels:
            continue

        skeleton = skeletonize(mask.astype(np.float32))

        results = fit_multiple_sine_safe_multipercluster(
            skeleton,
            wavelength=labels.shape[1],  # important: full image width
            **fit_kwargs
        )

        # attach class to each fitted curve dict
        for r in results:
            r["class"] = int(cls)

        results_by_class[int(cls)] = results
        all_results.extend(results)

    return all_results, results_by_class

# ---- Plotting with mplstereonet ----
def plot_planes_stereonet(planes, title=None, savepath=None, show=True):
    """
    Plot planes (dip, dip_dir) on stereonet as poles.
    """
    fig, ax = mplstereonet.subplots(figsize=(6,6))
    for dip, dipdir in planes:
        # add pole (normal to plane)
        ax.pole(dip, dipdir, 'ko')  
        # optionally: also draw the plane great circle
        ax.plane(dip, dipdir, color='blue', linewidth=0.7, alpha=0.5)

    ax.grid(True)
    if title:
        ax.set_title(title)
    if savepath:
        fig.savefig(savepath, dpi=200, bbox_inches='tight')
    if show:
        plt.show()
    plt.close(fig)

import csv

def export_fitted_curves_to_csv(all_results, csv_path):
    rows = []
    for i, r in enumerate(all_results):
        A, B, C, D = r["params"]  # params are (A, B, C, D) in your returned results
        rows.append({
            "Class": int(r.get("class", -1)),
            "curve_id": i,
            "A": float(A),
            "B": float(B),
            "C": float(C),
            "D": float(D),
            "rmse": float(r.get("rmse", np.nan)),
            "n_points": int(r.get("n_points", 0)),
            "x_min": float(r["x_range"][0]),
            "x_max": float(r["x_range"][1]),
            "Dip": float(r.get("dip_deg", np.nan)),
            "DipDir": float(r.get("azimuth_deg", np.nan)),
            # "Azimuth": float(r.get("azimuth_deg", np.nan)),
        })
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"Saved fitted curves to {csv_path}")


def export_planes_to_csv(planes, filepath, format="strike_dip"):
    """
    Export list of (dip, dip_dir) planes to CSV.
    
    planes: list of (dip, dip_dir) tuples
    filepath: output filename (e.g., 'planes.csv')
    format: "dipdir_dip" (default) or "strike_dip"
    """
    with open(filepath, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        if format == "dipdir_dip":
            writer.writerow(["DipDir", "Dip"])  # header
            for dip, dipdir in planes:
                writer.writerow([f"{dipdir:.2f}", f"{dip:.2f}"])
                
        elif format == "strike_dip":
            writer.writerow(["Strike", "Dip"])
            for dip, dipdir in planes:
                # Right-hand rule: strike = dipdir - 90
                strike = (dipdir - 90.0) % 360.0
                writer.writerow([f"{strike:.2f}", f"{dip:.2f}"])
                
        else:
            raise ValueError("format must be 'dipdir_dip' or 'strike_dip'")

import pandas as pd

def plot_stereonet_from_csv_1(csv_file_path, save_fig_path='stereonet_output.png', pred=True, App_to_True=False, bh_tilt=0.0, bh_azimuth=0.0, classify=False):
    # 1. Load the data
    try:
        df = pd.read_csv(csv_file_path)
        df_dropped = df.drop(index=0)
        if pred:
            dipdir = df_dropped['DipDir'].astype(float)
            strikes = (dipdir + 90.0) % 360.0
        else:
            Azimuth = df_dropped['Azimuth'].astype(float)
            strikes = (Azimuth - 90.0) % 360.0
        dips = df_dropped['Dip'].astype(float)
        if App_to_True:
            strikes, dips = correct_apparent_to_true(
                strikes.to_numpy() if hasattr(strikes, "to_numpy") else np.asarray(strikes),
                dips.to_numpy() if hasattr(dips, "to_numpy") else np.asarray(dips),
                borehole_tilt_deg=bh_tilt,
                borehole_azimuth_deg=bh_azimuth
            )
        has_class = "Class" in df_dropped.columns
        if has_class:
            classes = df_dropped["Class"].astype(int)
    except FileNotFoundError:
        print(f"Error: The file {csv_file_path} was not found.")
        return

    # 2. Create the Figure and Stereonet Axes
    fig = plt.figure(figsize=(7.5, 6))
    ax = fig.add_subplot(111, projection='stereonet')

    # 3. Plot the Planes (Great Circles)
    CLASS_NAME_MAP = {
        1: "Tectonic Fault",
        2: "Bedding",
        3: "Fault Zone Boundary",
        4: "EDZ",
        5: "Unclassified Fracture",
        6: "Fault Gouge",
        7: "Longitudinal Induced Fracture",
        8: "Desiccation Crack"
    }

    if has_class & classify:
        marker_list = ['v', '*', 'o', 's', 'X', '^', 'P', 'D', '<', '>', 'h', 'p', '+', 'x']
        cmap = plt.get_cmap("tab10")
        unique_classes = np.sort(classes.unique())
        for i, cls in enumerate(unique_classes):
            mask = (classes == cls).to_numpy()
            marker = marker_list[cls % len(marker_list)]
            ax.pole(
                strikes[mask], dips[mask],
                c=cmap(cls), marker=marker, markersize=10, linestyle='none',
                label=CLASS_NAME_MAP.get(cls, f"Class {cls}")
            )
        legend = ax.legend(
            loc='upper left',
            bbox_to_anchor=(1.0, 1.0),
            borderaxespad=0.0,
            fontsize=14,
            frameon=False
        )
    else:
        ax.pole(strikes, dips, c='k', marker='o', markersize=5, linestyle='none', label='Poles')
        ax.density_contourf(strikes, dips, measurement='poles', cmap='Greens')
        legend = None

    # 4. Add Decoration
    ax.grid(kind='polar', color='gray', linestyle='--', alpha=0.5)

    # 5. Save — bbox_extra_artists ensures the legend is included in the canvas
    extra = [legend] if legend is not None else []
    plt.savefig(
        save_fig_path,
        dpi=300,
        bbox_inches='tight',
        bbox_extra_artists=extra,
        pad_inches=0.2
    )
    plt.show()



if __name__ == "__main__":
    borehole = "BSE_P26"
    width = 1440
    pred = True
    ##### "trainwithfeatmap" or "trainwithlogits" or "binarypred" or "trainboth"
    mode = "trainboth"
    classify = False
    # if not pred:
    # Generate stereonet for ground truth
    gt_file = f"borehole_generalization/data/{borehole}_Structures_ML_app.csv"
    plot_stereonet_from_csv(gt_file, save_fig_path=f"borehole_generalization/stereonet/{borehole}_stereonet_gt_classify_withlegend.png", pred = False, app_to_true=False, bh_tilt=79.77, bh_azimuth=270.30, classify = True) 
    plot_stereonet_from_csv(gt_file, save_fig_path=f"borehole_generalization/stereonet/{borehole}_stereonet_gt.png", pred = False, app_to_true=False, bh_tilt=79.77, bh_azimuth=270.30, classify = False) 
    # mask_path_gt = f"data/Labeled Data Myriam/FS_B/BFS_B15_bottom_structures_dilated.png"
    # mask_path = f"data/general-boreholes-output/paper/{borehole}_720_onlyPEskipfusion_pred_binary_inversionloss.png"
    # else:
    # Generate stereonet for prediction
    pred_map = f"data/general-boreholes-output/final_pred_originalsetting/{borehole}.png" 
    if classify:
        print("Classified fitting mode enabled.")
        all_results, results_by_class = fit_curves_per_class_from_label_png(
            pred_map_path=pred_map,
            class_ids=[1,2,3,4,5,6,7],   # or None to auto-detect all non-zero
            min_pixels=200,
            dbscan_eps=30,
            dbscan_min_samples=5,
            min_points_for_fit=200,
            phase_tol=0.6,
            A_tol_ratio=0.4,
            D_tol=10,
            rmse_thresh=20.0,
            verbose=True
        )
        
        curves_csv = f"borehole_generalization/paper/{borehole}_{width}_fitted_curves_allclasses_{mode}.csv"
        export_fitted_curves_to_csv(all_results, curves_csv)
        plot_stereonet_from_csv(curves_csv, save_fig_path=f"borehole_generalization/stereonet/{borehole}_stereonet_output_classify_{mode}.png", classify = classify)
            # ---- create curves image (all classes together) ----
        img = Image.open(pred_map) 
        img = img.convert("L")
        img = np.array(img)
        mask = img > 0.5
        mask_arr = mask*1.0
        H,W = img.shape
        print("Mask number of pixels:", np.sum(mask))
        skeleton = skeletonize(mask_arr)
        # img = Image.open(pred_map).convert("L")
        # img = np.array(img)
        # H, W = img.shape 
        results = fit_multiple_sine_safe_multipercluster(skeleton,
                                wavelength=W,
                                dbscan_eps=30,
                                dbscan_min_samples=5,
                                min_points_for_fit=200,
                                phase_tol=0.6,    # radians tolerance for merging
                                A_tol_ratio=0.4,  # amplitude relative tolerance for merging
                                D_tol=10,         # vertical offset tolerance (pixels)
                                rmse_thresh=20.0,  # discard fits with RMSE > this
                                verbose=True)

        curve_img = np.zeros((H, W), dtype=np.uint8)
        for r in results:
            # use fitted curve coordinates
            x_full = r["x_full"].astype(int)
            y_full = np.round(r["y_full"]).astype(int)

            # keep only valid pixels
            mask = (x_full >= 0) & (x_full < W) & (y_full >= 0) & (y_full < H)
            x_full = x_full[mask]
            y_full = y_full[mask]

            # get class labels along the curve
            labels = img[y_full, x_full]

            # ignore background
            labels = labels[labels > 0]

            if len(labels) == 0:
                curve_class = 0
            else:
                curve_class = np.bincount(labels).argmax()

            # store class in result
            r["class_id"] = int(curve_class)
            # draw thicker curve
            for dx in [-2,-1,0,1,2]:
                for dy in [-2,-1,0,1,2]:

                    xx = x_full + dx
                    yy = y_full + dy

                    m = (xx >= 0) & (xx < W) & (yy >= 0) & (yy < H)

                    curve_img[yy[m], xx[m]] = r["class_id"]
            # curve_img[y_full, x_full] = r["class_id"]
        curve_img = ndimage.binary_dilation(curve_img > 0) * curve_img
        imageio.imwrite(f"borehole_generalization/paper/{borehole}_two_step_pred_map_fitcurves_classified_{mode}.png", curve_img)  # save classified curves image

    else:
        print("Classified fitting mode disabled.")
        img = Image.open(pred_map) 
        img = img.convert("L")
        img = np.array(img)
        mask = img > 0.5
        mask_arr = mask*1.0
        H,W = img.shape
        print("Mask number of pixels:", np.sum(mask))
        skeleton = skeletonize(mask_arr)
        results = fit_multiple_sine_safe_multipercluster(skeleton,
                                wavelength=W,
                                dbscan_eps=30,
                                dbscan_min_samples=5,
                                min_points_for_fit=200,
                                phase_tol=0.6,    # radians tolerance for merging
                                A_tol_ratio=0.4,  # amplitude relative tolerance for merging
                                D_tol=10,         # vertical offset tolerance (pixels)
                                rmse_thresh=20.0,  # discard fits with RMSE > this
                                verbose=True)
        for r in results[:5]:
            plt.plot(r['x_full'], r['y_full'])
        h, w = skeleton.shape
        sine_img = np.zeros((h, w), dtype=np.uint8)  # background=0
        print("Image shape:", sine_img.shape)

        for r in results:
            x_full = r['x_full'].astype(int)
            y_full = np.round(r['y_full']).astype(int)

            # keep only points inside image bounds
            mask = (x_full >= 0) & (x_full < w) & (y_full >= 0) & (y_full < h)
            x_full, y_full = x_full[mask], y_full[mask]

            sine_img[y_full, x_full] = 1   # mark curve pixels
        dilate_sineimg = ndimage.binary_dilation(sine_img).astype(sine_img.dtype)
        imageio.imwrite(f"borehole_generalization/paper/{borehole}_two_step_pred_map_fitcurves_{mode}.png", dilate_sineimg)  # multiply so curves are visible
        planes = fitted_traces_to_planes(results, min_support=30)
        dip_dir_file = f"borehole_generalization/data/planes_dipdir_test_{borehole}_pred_map_{mode}.csv"
        # Export in DipDir/Dip format
        export_planes_to_csv(planes, dip_dir_file, format="dipdir_dip") 
        plot_stereonet_from_csv(dip_dir_file, save_fig_path=f"borehole_generalization/stereonet/{borehole}_stereonet_output_{mode}.png")
    

