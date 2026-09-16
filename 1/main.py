import os
import sys
import glob
import time
import argparse
import numpy as np
import cv2

# Helpers

def load_image(filepath):
    """
    Load an image (JPG or TIF) as a floating-point 2D grayscale array in [0, 1].
    Handles 8-bit and 16-bit depths.
    """
    img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)

    if img is None:
        raise FileNotFoundError(f"Could not open image: {filepath}")

    if img.ndim == 3:
        img = img[:, :, 0]
    
    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.0
    elif img.dtype == np.uint16:
        img = img.astype(np.float32) / 65535.0
    else:
        img = img.astype(np.float32)
        min_val, max_val = img.min(), img.max()
        if max_val > min_val:
            img = (img - min_val) / (max_val - min_val)
    
    return img

def split_channels(img):
    """
    Divide glass plate image into 3 equal parts, ordered from top to bottom: Blue, Green, Red.
    """
    h = int(np.floor(img.shape[0] / 3.0))
    return img[:h], img[h:2*h], img[2*h:3*h]
    
def shift_image(img, d):
    """
    Shift image by (dx, dy) pixels
    Using np.roll for fast shift, border regions cropped out during calculation
    """
    dx, dy = d
    return np.roll(np.roll(img, dy, axis=0), dx, axis=1)

def compute_features(img, feature_type='raw'):
    """
    Extract features:
    - 'raw': normalized raw pixel intensities
    - 'gradient': sobel gradient magnitude
    """
    if feature_type == 'raw':
        return img
    elif feature_type == 'gradient':
        grad_x, grad_y = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
        return np.sqrt(grad_x ** 2 + grad_y ** 2)
    else:
        raise ValueError(f"Unknown feature type: {feature_type}")

# Alignment Metrics

def score_alignment(ref, target, metric='ncc'):
    """
    Compute similarity/distance between 2 patches
    - 'ncc': Normalized Cross-Correlation (higher = better)
    - 'l2' / 'ssd': Sum of Squared Differences (lower = better)
    """
    if metric == 'ncc':
        ref_norm, target_norm = np.linalg.norm(ref - np.mean(ref)), np.linalg.norm(target - np.mean(target))
        if ref_norm == 0 or target_norm == 0: return 0.0
        return np.sum((ref - np.mean(ref)) * (target - np.mean(target))) / (ref_norm * target_norm)
    elif metric == 'l2' or metric == 'ssd':
        return np.sum(np.square(ref - target))
    else:
        raise ValueError(f"Unknown metric: {metric}")

# Single-Scale Alignment

def align_single_scale(target, ref, search_window=(-15, 15), metric='ncc', feature='raw', border_crop=0.15, base_shift=(0, 0)):
    """
    Exhaustive search to align target onto ref over displacement grid.

    Parameters:
        target: 2d numpy array
        ref: 2d numpy array
        metric: 'ncc' or 'l2' or 'ssd'
        feature: 'raw' or 'gradient'
        border_crop: fraction of outer border to ignore (0.15 default)
        base_shift: initial guess for shift (dx, dy)
    
    Returns:
        best_shift: (dx, dy) total displacement vector
    """
    ref_feat, target_feat = compute_features(ref, feature_type=feature), compute_features(target, feature_type=feature)

    h, w = ref_feat.shape
    crop_h, crop_w = int(h * border_crop), int(w * border_crop)
    ref_crop = ref_feat[crop_h:h - crop_h, crop_w:w - crop_w]

    best_score = -np.inf if metric == 'ncc' else np.inf
    best_shift = (base_shift[0], base_shift[1])

    min_off, max_off = search_window
    bx, by = base_shift

    for dx in range(bx + min_off, bx + max_off + 1):
        for dy in range(by + min_off, by + max_off + 1):
            target_crop = shift_image(target_feat, (dx, dy))[crop_h: h - crop_h, crop_w: w - crop_w]

            score = score_alignment(ref_crop, target_crop, metric=metric)

            if metric == 'ncc' and score > best_score:
                best_score, best_shift = score, (dx, dy)
            elif metric != 'ncc' and score < best_score:
                best_score, best_shift = score, (dx, dy)

    return best_shift

# Multi-Scale Pyramid Alignment

def align_pyramid(target, ref, metric='ncc', feature='raw', coarsest_dim=128, search_range=2, border_crop=0.15):
    """
    Gaussian image pyramid alignement.
    Starts at coarsest scale, recursively doubles displacement estimate, refines within search range.

    Parameters:
        target: 2d numpy array
        ref: 2d numpy array
        metric: 'ncc' or 'l2' or 'ssd'
        feature: 'raw' or 'gradient'
        coarsest_dim: coarsest scale dimension (128 default)
        search_range: expand search by factor of 2 (2 default)
        border_crop: fraction of outer border to ignore (0.15 default)

    Returns:
        best_shift: (dx, dy) total displacement vector
    """
    if target.shape[0] <= coarsest_dim or target.shape[1] <= coarsest_dim:
        return align_single_scale(target, ref, search_window=(-15, 15), metric=metric, feature=feature, border_crop=border_crop)

    coarse_dx, coarse_dy = align_pyramid(
        cv2.pyrDown(target), cv2.pyrDown(ref), metric=metric, feature=feature,
        coarsest_dim=coarsest_dim, search_range=search_range, border_crop=border_crop
    )

    return align_single_scale(target, ref, search_window=(-search_range, search_range), metric=metric, feature=feature, border_crop=border_crop, base_shift=(2 * coarse_dx, 2 * coarse_dy))

# Main

def process_img(filepath, output_dir, metric='ncc', use_features=None):
    """
    Processes single Prokudin-Gorskii glass plate image:
    1. Loads & splits image into B/G/R
    2. Determines single-scale or pyramid alignment based on resolution
    3. Computes alignment offsets & stacks into RGB image
    4. Saves output image, returns offsets
    """
    filename = os.path.basename(filepath)
    name, ext = os.path.splitext(filename)

    t0 = time.time()
    print(f"\n=======================")
    print(f"Processing: {filename}")

    img = load_image(filepath)
    b, g, r = split_channels(img)
    h, w = b.shape
    print(f"Channel dimensions: {h} x {w}")

    if use_features is None:
        if 'emir' in name.lower():
            feat = 'gradient'
            print("Using gradient features for emir.tif to handle color/brightness disparity")
        else:
            feat = 'raw'
    else:
        feat = use_features
    
    if h > 500:
        print(f"Using Pyramid Alignment (metric: {metric}, feature: {feat})")
        shift_g, shift_r = align_pyramid(g, b, metric=metric, feature=feat), align_pyramid(r, b, metric=metric, feature=feat)
    else:
        print(f"Using Single-Scale Alignment (metric: {metric}, feature: {feat})")
        win = (-35, 35) if 'locomotive' in name.lower() else (-15, 15)
        shift_g, shift_r = align_single_scale(g, b, search_window=win, metric=metric, feature=feat), align_single_scale(r, b, search_window=win, metric=metric, feature=feat)
    
    print(f"Calculated displacement for G: dx = {shift_g[0]}, dy = {shift_g[1]}")
    print(f"Calculated displacement for R: dx = {shift_r[0]}, dy = {shift_r[1]}")
    
    aligned_g, aligned_r = shift_image(g, shift_g), shift_image(r, shift_r)
    rgb_aligned = np.dstack((aligned_r, aligned_g, b))

    os.makedirs(output_dir, exist_ok=True)
    out_path_std = os.path.join(output_dir, name + "_aligned.jpg")
    cv2.imwrite(out_path_std, cv2.cvtColor((np.clip(rgb_aligned, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
    print(f"Standard result saved to {out_path_std}")
    elapsed = time.time() - t0
    print(f"Total processing time: {elapsed} seconds")

    return {'file': filename, 'shift_g': shift_g, 'shift_r': shift_r, 'time': elapsed}

def main():
    parser = argparse.ArgumentParser(description='Align Prokudin-Gorskii images')
    parser.add_argument('--input', type=str, default=None, help="Path to single img or dir of imgs")
    parser.add_argument('--output', type=str, default='./outputs', help="Dir to save colorized outputs")
    parser.add_argument('--metric', type=str, default='ncc', choices=['ncc', 'l2', 'ssd'], help="Similarity metric")
    parser.add_argument('--feat', type=str, default=None, choices=['raw', 'gradient'], help="Feature type for alignment. Defaults to raw, gradient for emir")
    args = parser.parse_args()

    if args.input is None:
        data_dir = './CS180_fa2026_proj1_data'
        if os.path.exists(data_dir):
            files = sorted(glob.glob(os.path.join(data_dir, '*.jpg')) + glob.glob(os.path.join(data_dir, '*.tif')))
        else:
            files = sorted(glob.glob('*.jpg') + glob.glob('*.tif'))
    elif os.path.isdir(args.input):
        files = sorted(glob.glob(os.path.join(args.input, '*.jpg')) + glob.glob(os.path.join(args.input, '*.tif')))
    else:
        files = [args.input]
    
    if not files:
        print("No images found to process.")
        sys.exit(1)

    print(f"Found {len(files)} image(s). \n")
    print("=" * 10)

    res = []
    for f in files:
        res.append(
            process_img(
                f,
                output_dir=args.output,
                metric=args.metric,
                use_features=args.feat
            )
        )

    print("\n" + "=" * 10)
    print("Summary:")
    print("\n" + "=" * 10)
    print(f"{'Image':<25} | {'Green (dx, dy)':<18} | {'Red (dx, dy)':<18} | {'Time (s)':<8}")
    print("=" * 10)
    for r in res:
        g_str , r_str= f"({r['shift_g'][0]}, {r['shift_g'][1]})", f"({r['shift_r'][0]}, {r['shift_r'][1]})"
        print(f"{r['file']:<25} | {g_str:<18} | {r_str:<18} | {r['time']}")
    print("=" * 10)

if __name__ == '__main__':
    main()