import os
import sys
sys.path.append(os.path.abspath("../4a/code"))
from part_a import computeH, apply_homography, warpImageBilinear, blend_img, load_scale_img
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import gaussian_filter, map_coordinates, distance_transform_edt
from skimage.feature import corner_harris, peak_local_max
from harris import get_harris_corners

# B.1

def detect_harris_corners(img_gray, edge_discard=20):
    h_mat, coords = get_harris_corners(img_gray, edge_discard=edge_discard)
    xs, ys = coords[1], coords[0]
    return h_mat, np.stack([xs, ys], axis=1).astype(np.float64), h_mat[ys, xs]

def anms(pts, strengths,  num_pts=500, c_robust=0.9, max_candidates=4000):
    n = len(pts)
    if n <= num_pts:
        return pts, strengths, np.inf * np.ones(n)

    order = np.argsort(strengths)[::-1]
    if n > max_candidates and max_candidates is not None:
        order = order[:max_candidates]

    pts_sorted, strengths_sorted = pts[order], strengths[order]
    m = len(pts_sorted)
    dists = np.sum((pts_sorted[:, None, :] - pts_sorted[None, :, :]) ** 2, axis=-1).astype(np.float64)
    dists[(strengths_sorted[:, None] >= c_robust * strengths_sorted[None, :])] = np.inf
    
    radii = np.sqrt(np.min(dists, axis=1))
    idx = np.argsort(radii)[::-1][:num_pts]

    return pts_sorted[idx], strengths_sorted[idx], radii[idx]

# B.2

def extract_feature_descriptors(img_gray, pts, patch_radius=20, sample_size=8, sigma=1.5):
    h, w = img_gray.shape
    blurred = gaussian_filter(img_gray, sigma=sigma)
    valid_pts, descriptors = [], []

    for point in pts:
        x, y = int(round(point[0])), int(round(point[1]))
        if (x - patch_radius >= 0 and x + patch_radius <= w and
            y - patch_radius >= 0 and y + patch_radius <= h):
            desc = blurred[
                y - patch_radius : y + patch_radius,
                x - patch_radius : x + patch_radius
            ][2::5, 2::5][:sample_size, :sample_size].astype(np.float64)

            mu, std = np.mean(desc), np.std(desc)
            if std > 1e-6:
                desc_norm = (desc - mu) / (std + 1e-8)
                descriptors.append(desc_norm.flatten())
                valid_pts.append(point)
    
    return np.array(valid_pts, dtype=np.float64), np.array(descriptors, dtype=np.float64)
            
# B.3
    
def match_features(desc1, desc2, ratio_threshold=0.7):
    if len(desc1) == 0 or len(desc2) == 0:
        return np.zeros((0, 2), dtype=int), np.zeros(0)
    
    dists = np.sum((desc1[:, None, :] - desc2[None, :, :]) ** 2, axis=-1)

    sorted_idx = np.argsort(dists, axis=1)
    d1 = np.sqrt(dists[np.arange(len(desc1)), sorted_idx[:, 0]])
    d2 = np.sqrt(dists[np.arange(len(desc1)), sorted_idx[:, 1]])
    ratios = d1 / (d2 + 1e-8)
    valids = ratios < ratio_threshold
    return np.stack([np.where(valids)[0], sorted_idx[valids, 0]], axis=1), ratios[valids]

# B.4
def ransac_4pt(pts1, pts2, num_iter=3000, inlier_thresh=3.0):
    n = len(pts1)
    if n < 4:
        raise ValueError(f"at least 4 pairs needed, got {n}")
    
    best_inliers = np.zeros(n, dtype=bool)
    max_inliers_cnt = 0
    best_H = None

    for _ in range(num_iter):
        idx = np.random.choice(n, 4, replace=False)
        p1, p2 = pts1[idx], pts2[idx]

        try:
            H_cand = computeH(p2, p1)
            if np.isnan(H_cand).any() or np.isinf(H_cand).any():
                continue

            pred_pts1 = apply_homography(H_cand, pts2)
            errors = np.linalg.norm(pred_pts1 - pts1, axis=1)

            inliers = errors < inlier_thresh
            inliers_cnt = np.sum(inliers)

            if inliers_cnt > max_inliers_cnt:
                max_inliers_cnt = inliers_cnt
                best_inliers = inliers
                best_H = H_cand

        except np.linalg.LinAlgError:
            continue
    
    if max_inliers_cnt >= 4:
        best_H = computeH(pts2[best_inliers], pts1[best_inliers])
    
    return best_H, best_inliers


# visualization

def save_fig(fig, path):
    plt.tight_layout()
    if path:
        plt.savefig(path, bbox_inches='tight', dpi=150)
        print(f"saved {path}")
    plt.close(fig)

def visualize_pair(img1, img2, t1="", t2 ="", pts1=None, pts2=None, save_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    for ax, img, t, pts, c in zip(axes, [img1, img2], [t1, t2], [pts1, pts2], ['r.', 'yo']):
        ax.imshow(img)
        if pts is not None:
            ax.plot(pts[:, 0], pts[:, 1], c, markersize=3)
        ax.set_title(t)
        ax.axis("off")
    save_fig(fig, save_path)

def visualize_matches(img1, img2, pts1, pts2, inliers=None, title="", save_path=None):
    fig, ax = plt.subplots(figsize=(16, 7))
    h = max(img1.shape[0], img2.shape[0])
    p1 = np.pad(img1, ((0, h - img1.shape[0]), (0, 0), (0, 0)))
    p2 = np.pad(img2, ((0, h - img2.shape[0]), (0, 0), (0, 0)))
    ax.imshow(np.hstack([p1, p2]))
    ax.axis('off')
    ax.set_title(title, fontweight='bold')

    shift = np.array([img1.shape[1], 0])
    inliers = np.ones(len(pts1), dtype=bool) if inliers is None else inliers

    for p1, p2, is_inlier in zip(pts1, pts2, inliers):
        c, alpha = ('lime', 0.85) if is_inlier else ('red', 0.35)
        ax.plot([p1[0], p2[0] + shift[0]], [p1[1], p2[1]], color=c, linewidth=.8, alpha=alpha)
    save_fig(fig, save_path)

def visualize_descriptors(descriptors, valid_pts, num_display=16, save_path=None):
    dim = int(np.ceil(np.sqrt(num_display)))
    fig, axes = plt.subplots(dim, dim, figsize=(8, 8))
    for i in range(dim ** 2):
        ax = axes[i // dim, i % dim]
        if i < len(descriptors) and i < num_display:
            ax.imshow(descriptors[i].reshape(8, 8), cmap="gray", interpolation='nearest')
            pt = valid_pts[i]
            ax.set_title(f"({int(pt[0])}, {int(pt[1])})", fontsize=8)
        ax.axis('off')
    plt.suptitle("sample 8x8 normalized feature descriptors", fontsize=12, fontweight='bold')
    save_fig(fig, save_path)

# pipeline execution

def extract_features(img_gray, num_anms=1000):
    _, h_pts, s = detect_harris_corners(img_gray, edge_discard=20)
    anms_pts, _, _ = anms(h_pts, s, num_pts=num_anms)
    valid_pts, desc = extract_feature_descriptors(img_gray, anms_pts)
    return h_pts, anms_pts, valid_pts, desc

def autostitch_pair(name, left, right, manual_mosaic_path=None, scale=0.25, 
                    num_anms=1000, ratio_threshold=0.7, ransac_thresh=3.0, 
                    ransac_iter=4000, out_dir="images"):
    os.makedirs(out_dir, exist_ok=True)
    img_l = load_scale_img(left, scale=scale)
    img_r = load_scale_img(right, scale=scale)
    gray_l = np.array(Image.fromarray(img_l).convert('L'), dtype=np.float32) / 255.0
    gray_r = np.array(Image.fromarray(img_r).convert('L'), dtype=np.float32) / 255.0
    
    hl_pts, anms_l, pts_l, desc_l = extract_features(gray_l, num_anms)
    hr_pts, anms_r, pts_r, desc_r = extract_features(gray_r, num_anms)
    visualize_pair(img_l, img_l, "Harris", "ANMS", hl_pts, anms_l, f"{out_dir}/{name}_anms_left.png")
    visualize_pair(img_r, img_r, "Harris", "ANMS", hr_pts, anms_r, f"{out_dir}/{name}_anms_right.png")

    visualize_descriptors(desc_l, pts_l, num_display=16, save_path=f"{out_dir}/{name}_descriptors_sample.png")

    matches, _ = match_features(desc_l, desc_r, ratio_threshold=ratio_threshold)
    m_l, m_r = pts_l[matches[:, 0]], pts_r[matches[:, 1]]
    visualize_matches(img_l, img_r, m_l, m_r, title=f"{name} initial matches", save_path=f"{out_dir}/{name}_matches.png")
    
    H, inliers = ransac_4pt(m_l, m_r, num_iter=ransac_iter, inlier_thresh=ransac_thresh)
    visualize_matches(img_l, img_r, m_l, m_r, inliers, title=f"{name} RANSAC inliers", 
                      save_path=f"{out_dir}/{name}_ransac_inliers.png")
    
    auto_blend = blend_img(img_l, img_r, H, blend_mode='distance')
    Image.fromarray(auto_blend).save(f"{out_dir}/{name}_pano_auto.png")

    if manual_mosaic_path and os.path.exists(manual_mosaic_path):
        manual_mosaic = np.array(Image.open(manual_mosaic_path).convert('RGB'), dtype=np.uint8)
        visualize_pair(manual_mosaic, auto_blend, "Part A: Manual", "Part B: Automatic", 
                       save_path=f"{out_dir}/{name}_manual_vs_auto.png")
    
    return {"name": name, "matches": len(matches), "inliers": np.sum(inliers)}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale", type=float, default=0.25)
    parser.add_argument("--out_dir", type=str, default="images")
    args = parser.parse_args()

    for name in ["LBNL", "lewis", "stadium"]:
        res = autostitch_pair(
            name,
            f"../4a/images/{name}_left.png",
            f"../4a/images/{name}_right.png",
            f"../4a/images/{name}_mosaic_blended.png",
            scale=args.scale, out_dir=args.out_dir
        )
        print(f"[{name}] Matches: {res['matches']}, Inliers: {res['inliers']}")

if __name__ == "__main__":
    main()
    
    
    
