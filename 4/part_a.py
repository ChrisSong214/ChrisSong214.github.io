import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import argparse
from scipy.ndimage import distance_transform_edt

def computeH(img1_pts, img2_pts):
    img1_pts = np.array(img1_pts, dtype=np.float64)
    img2_pts = np.array(img2_pts, dtype=np.float64)

    n = img1_pts.shape[0]
    if n < 4:
        raise ValueError("not enough point correspondences, at least 4 required")

    A, b = np.zeros((2 * n, 8)), np.zeros(2 * n)
    
    for i in range(n):
        x, y = img1_pts[i]
        u, v = img2_pts[i]

        A[2*i] = [x, y, 1, 0, 0, 0, -x*u, -y*u]
        A[2*i + 1] = [0, 0, 0, x, y, 1, -x*v, -y*v]

        b[2*i] = u
        b[2*i + 1] = v
    
    h, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)

    return np.array([[h[0], h[1], h[2]], [h[3], h[4], h[5]], [h[6], h[7], 1.0]])

def apply_homography(H, pts):
    pts = np.asarray(pts)
    n = pts.shape[0]
    homog = np.hstack([pts, np.ones((n, 1))])
    transformed = (H @ homog.T).T
    return transformed[:, :2] / transformed[:, 2:3]

def get_warped_canvas_bounds(im_shape, H):
    h, w = im_shape[:2]
    corners = np.array([
        [0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]
    ])
    w_corners = apply_homography(H, corners)

    min_x = np.floor(np.min(w_corners[:, 0])).astype(int)
    max_x = np.ceil(np.max(w_corners[:, 0])).astype(int)
    min_y = np.floor(np.min(w_corners[:, 1])).astype(int)
    max_y = np.ceil(np.max(w_corners[:, 1])).astype(int)
    return min_x, max_x, min_y, max_y

def warpImageNearestNeighbor(img, H, out_bounds=None):
    if img.ndim == 2:
        img = img[:, :, np.newaxis]
    h, w, c = img.shape

    if out_bounds is None:
        min_x, max_x, min_y, max_y = get_warped_canvas_bounds(img.shape, H)
    else:
        min_x, max_x, min_y, max_y = out_bounds

    out_w, out_h = max_x - min_x + 1, max_y - min_y + 1

    x_dest, y_dest = np.meshgrid(np.arange(min_x, max_x + 1), np.arange(min_y, max_y + 1))
    dest_pts = np.vstack([x_dest.ravel(), y_dest.ravel()]).T

    H_inv = np.linalg.inv(H)
    src_pts = apply_homography(H_inv, dest_pts)

    src_x, src_y = np.round(src_pts[:, 0]).astype(int), np.round(src_pts[:, 1]).astype(int)

    mask = np.zeros((out_h, out_w), dtype=bool)
    warped_im = np.zeros((out_h, out_w, c), dtype=img.dtype)
    valid = (src_x >= 0) & (src_x < w) & (src_y >= 0) & (src_y < h)
    
    warped_im[y_dest.ravel()[valid] - min_y, x_dest.ravel()[valid] - min_x] = img[src_y[valid], src_x[valid]]
    mask[y_dest.ravel()[valid] - min_y, x_dest.ravel()[valid] - min_x] = True

    if warped_im.shape[2] == 1:
        warped_im = warped_im[:, :, 0]

    return warped_im, mask, (min_x, min_y)

def warpImageBilinear(img, H, out_bounds=None):
    if img.ndim == 2:
        img = img[:, :, np.newaxis]
    
    img_flt = img.astype(np.float64)
    h, w, c = img.shape

    if out_bounds is None:
        min_x, max_x, min_y, max_y = get_warped_canvas_bounds(img.shape, H)
    else:
        min_x, max_x, min_y, max_y = out_bounds

    out_w, out_h = max_x - min_x + 1, max_y - min_y + 1

    x_dest, y_dest = np.meshgrid(np.arange(min_x, max_x + 1), np.arange(min_y, max_y + 1))
    dest_pts = np.vstack([x_dest.ravel(), y_dest.ravel()]).T

    H_inv = np.linalg.inv(H)
    src_pts = apply_homography(H_inv, dest_pts)

    sx, sy = src_pts[:, 0], src_pts[:, 1]
    x0, y0 = np.floor(sx).astype(int), np.floor(sy).astype(int)
    x1, y1 = x0 + 1, y0 + 1

    valid = (x0 >= 0) & (x1 < w) & (y0 >= 0) & (y1 < h)

    warped_img = np.zeros((out_h, out_w, c), dtype=np.float64)
    mask = np.zeros((out_h, out_w), dtype=bool)

    if np.any(valid):
        v_x0, v_x1 = x0[valid], x1[valid]
        v_y0, v_y1 = y0[valid], y1[valid]

        wx1, wy1 = (sx[valid] - v_x0)[:, np.newaxis], (sy[valid] - v_y0)[:, np.newaxis]
        wx0, wy0 = 1.0 - wx1, 1.0 - wy1

        dest_x = x_dest.ravel()[valid] - min_x
        dest_y = y_dest.ravel()[valid] - min_y

        val00 = img_flt[v_y0, v_x0, :]
        val01 = img_flt[v_y1, v_x0, :]
        val10 = img_flt[v_y0, v_x1, :]
        val11 = img_flt[v_y1, v_x1, :]

        interp = (val00 * wx0 + val10 * wx1) * wy0 + (val01 * wx0 + val11 * wx1) * wy1

        warped_img[dest_y, dest_x, :] = np.clip(interp, 0, 255).astype(img.dtype)
        mask[dest_y, dest_x] = True

    if np.issubdtype(img.dtype, np.integer):
        warped_img = np.clip(np.round(warped_img), 0, 255).astype(img.dtype)
    
    if warped_img.shape[2] == 1:
        warped_img = warped_img[:, :, 0]

    return warped_img, mask, (min_x, min_y)

def rectify_img(img, src_pts, dst_size=(400, 300), method='bilinear'):
    w, h = dst_size
    dst_pts = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float64)

    H = computeH(src_pts, dst_pts)

    if method == 'nn':
        warped, _, _ = warpImageNearestNeighbor(img, H, out_bounds=(0, w - 1, 0, h - 1))
    elif method == 'bilinear':
        warped, _, _ = warpImageBilinear(img, H, out_bounds=(0, w - 1, 0, h - 1))
    else:
        raise ValueError('unknown method')
    
    return warped, H

def blend_img(img1, img2, H2to1, blend_mode='distance'):
    img1_h, img1_w = img1.shape[:2]
    min_x2, max_x2, min_y2, max_y2 = get_warped_canvas_bounds(img2.shape, H2to1)

    canvas_min_x = min(0, min_x2)
    canvas_max_x = max(img1_w - 1, max_x2)
    canvas_min_y = min(0, min_y2)
    canvas_max_y = max(img1_h - 1, max_y2)
    
    canvas_w = canvas_max_x - canvas_min_x + 1
    canvas_h = canvas_max_y - canvas_min_y + 1

    if canvas_w > 5 * img1_w or canvas_h > 5 * img1_h:
        raise ValueError(
            f"\n[Error] Canvas size ({canvas_w} x {canvas_h}) is abnormally large!\n"
            f"This occurs when correspondence points are nearly collinear or clustered in a small area, causing the homography to degenerate.\n"
            f"Please click at least 8-12 points spread widely across the entire overlapping area (top, bottom, left, right)."
        )

    warped2, mask2, offset2 = warpImageBilinear(img2, H2to1, out_bounds=(canvas_min_x, canvas_max_x, canvas_min_y, canvas_max_y))
    warped1, mask1 = np.zeros((canvas_h, canvas_w, 3), dtype=np.float64), np.zeros((canvas_h, canvas_w), dtype=bool)

    off_x, off_y = -canvas_min_x, -canvas_min_y
    warped1[off_y:off_y+img1_h, off_x:off_x+img1_w, :] = img1.astype(np.float64)
    mask1[off_y:off_y+img1_h, off_x:off_x+img1_w] = True
    
    if blend_mode == "distance":
        dist1, dist2 = distance_transform_edt(mask1), distance_transform_edt(mask2)
        total_dist = dist1 + dist2
        total_dist[total_dist == 0] = 1.0

        blended = warped1 * (dist1 / total_dist)[:,:,np.newaxis] + warped2 * (dist2 / total_dist)[:, :, np.newaxis]
    else:
        blended = np.zeros_like(warped1)
        blended[mask1 & ~mask2] = warped1[mask1 & ~mask2]
        blended[mask2 & ~mask1] = warped2[mask2 & ~mask1]
        blended[mask1 & mask2] = (warped1[mask1 & mask2] * 0.5 + warped2[mask1 & mask2]* 0.5)
    
    return np.clip(np.round(blended), 0, 255).astype(np.uint8)

def load_pts(cache_path, img, title, num_pts=-1, reclick=False):
    os.makedirs(os.path.dirname(cache_path) if os.path.dirname(cache_path) else ".", exist_ok=True)

    if os.path.exists(cache_path) and not reclick:
        print(f"loaded from cache: {cache_path}")
        return np.load(cache_path)
    
    while True:
        fig, ax = plt.subplots(figsize=(12, 10))
        ax.imshow(img)
        target_str = f"Click {num_pts} points" if num_pts > 0 else "Click at least 4 points"
        ax.set_title(f"{title}\n{target_str} (Left-click: add | Right-click: undo | Enter: finish)", fontsize=12)
        plt.tight_layout()
        pts = plt.ginput(n=num_pts, timeout=0)
        plt.close(fig)

        pts = np.array(pts, dtype=np.float64)
        if len(pts) < 4:
            print(f"\n[Notice] Only got {len(pts)} points, but at least 4 are required. Please try clicking again.")
            continue
        if num_pts > 0 and len(pts) != num_pts:
            print(f"\n[Notice] Expected {num_pts} points, but got {len(pts)}. Please try clicking again.")
            continue
        break
    
    np.save(cache_path, pts)
    return pts

def visualize_correspondences(img1, img2, pts1, pts2, title="points correspondences", save_path=None):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    canvas = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
    canvas[:h1, :w1] = img1
    canvas[:h2, w1:w1+w2] = img2

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.imshow(canvas)
    ax.axis('off')
    ax.set_title(title, fontsize=12, fontweight='bold')

    for i in range(len(pts1)):
        p1, p2 = pts1[i], pts2[i] + np.array([w1, 0])
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], linestyle='--', linewidth=1, alpha=0.7)
        ax.plot(p1[0], p1[1], 'ro', markersize=6)
        ax.plot(p2[0], p2[1], 'go', markersize=6)
        ax.text(p1[0] + 5, p1[1] - 5, str(i+1), color='red', fontsize=12, fontweight='bold')
        ax.text(p2[0] + 5, p2[1] - 5, str(i+1), color='green', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"saved correspondence visualization to {save_path}")
    plt.close(fig)

def load_scale_img(path, scale=1.0):
    img = Image.open(path).convert('RGB')
    if scale != 1.0:
        img = img.resize((int(img.width * scale), int(img.height * scale)), Image.Resampling.LANCZOS)
    return np.array(img, dtype=np.uint8)

def process_recit(name, plaque_size=(500, 500), scale=0.5, reclick=False):
    path = f"images/{name}.png"
    img = load_scale_img(path, scale=scale)

    cache_file = f"points/{name}_rect_pts.npy"
    prompt = f"click 4 corners of the plaque in {name} order: top left, top right, bottom right, bottom left"
    src_pts = load_pts(cache_file, img, prompt, num_pts=4, reclick=reclick)

    rect_nn, H = rectify_img(img, src_pts, plaque_size, method="nn")
    Image.fromarray(rect_nn).save(f"images/{name}_rect_nn.png")
    print(f"saved images/{name}_rect_nn.png")

    rect_bil, _ = rectify_img(img, src_pts, plaque_size, method="bilinear")
    Image.fromarray(rect_bil).save(f"images/{name}_rect_bil.png")
    print(f"saved images/{name}_rect_bil.png")

    cx, cy = plaque_size[0] // 2, plaque_size[1] // 2
    crop_w = min(120, plaque_size[0] // 4)
    crop_nn = rect_nn[cy-crop_w:cy+crop_w, cx-crop_w:cx+crop_w]
    crop = rect_bil[cy-crop_w:cy+crop_w, cx-crop_w:cx+crop_w]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img)
    poly = np.vstack([src_pts, src_pts[0]])
    axes[0].plot(poly[:, 0], poly[:, 1], 'r-', linewidth=2)
    axes[0].scatter(src_pts[:, 0], src_pts[:, 1], color='yellow', s=50, zorder=5)
    axes[0].set_title(f"Original {name} with 4 Points")
    axes[0].axis('off')

    axes[1].imshow(rect_nn)
    axes[1].set_title("Rectified (Nearest Neighbor)")
    axes[1].axis('off')

    axes[2].imshow(rect_bil)
    axes[2].set_title("Rectified (Bilinear Interpolation)")
    axes[2].axis('off')

    plt.tight_layout()
    comparison_path = f"images/{name}_rect_comparison.png"
    plt.savefig(comparison_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"Saved comparison figure: {comparison_path}")

def process_mosaic(set_name, left_file, right_file, scale=0.5, reclick=False):
    img_left = load_scale_img(left_file, scale=scale)
    img_right = load_scale_img(right_file, scale=scale)

    pts_left = load_pts(f"points/{set_name}_pts_left.npy", img_left, f"{set_name} Left: click 8-12 points spread across the overlap, then press Enter", reclick=reclick)
    pts_right = load_pts(f"points/{set_name}_pts_right.npy", img_right, f"{set_name} Right: click matching points in same order", num_pts=len(pts_left), reclick=reclick)

    visualize_correspondences(img_left, img_right, pts_left, pts_right, title=f"{set_name} correspondences", save_path=f"images/{set_name}_correspondences.png")

    H_right_to_left = computeH(pts_right, pts_left)
    print(f"recovered homography for {set_name} \n {H_right_to_left}")

    naive = blend_img(img_left, img_right, H_right_to_left, blend_mode='average')
    Image.fromarray(naive).save(f"images/{set_name}_mosaic_naive.png")
    print(f"saved naive mosaic: images/{set_name}_mosaic_naive.png")

    blended = blend_img(img_left, img_right, H_right_to_left, blend_mode='distance')
    Image.fromarray(blended).save(f"images/{set_name}_mosaic_blended.png")
    print(f"saved blended mosaic: images/{set_name}_mosaic_blended.png")

def main():
    parser = argparse.ArgumentParser(description="run cs180 project 4a pipeline on your images")
    parser.add_argument("--step", type=str, default="all", choices=["all", "donner", "greek", "LBNL", "lewis", "stadium"],
                        help="Which image set to process (default: all)")
    parser.add_argument("--scale", type=float, default=0.25,
                        help="Scale factor to downsample images (e.g. 0.25 for quick run / ~1400px, 0.5 for ~2800px)")
    parser.add_argument("--reclick", action="store_true", help="Force re-clicking points even if cache exists")
    args = parser.parse_args()

    os.makedirs("points", exist_ok=True)
    os.makedirs("images", exist_ok=True)

    if args.step in ["all", "donner"]:
        process_recit("donner", plaque_size=(500, 500), scale=args.scale, reclick=args.reclick)

    if args.step in ["all", "greek"]:
        process_recit("greek_theater", plaque_size=(500, 500), scale=args.scale, reclick=args.reclick)

    if args.step in ["all", "LBNL"]:
        process_mosaic("LBNL", "images/LBNL_left.png", "images/LBNL_right.png", scale=args.scale, reclick=args.reclick)

    if args.step in ["all", "lewis"]:
        process_mosaic("lewis", "images/lewis_left.png", "images/lewis_right.png", scale=args.scale, reclick=args.reclick)

    if args.step in ["all", "stadium"]:
        process_mosaic("stadium", "images/stadium_left.png", "images/stadium_right.png", scale=args.scale, reclick=args.reclick)

    print("\n[Done!] All specified image outputs have been processed.")

if __name__ == "__main__":
    main()