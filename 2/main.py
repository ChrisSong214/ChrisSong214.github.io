import os
import time
import numpy as np
import scipy.signal
import cv2

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from cs180_proj2_hybrid_starter_code.align_image_code import align_images


# 1.1

def conv2d_2loops(img: np.ndarray, kernel: np.ndarray, padding: str="same") -> np.ndarray:
    """
    2d convolution implementation w/ 2 nested loops

    args:
        img: 2d np array, shape (H, W)
        kernel: 2d np array, shape(kh, kw)
        padding: "same"/"full"
    
    returns:
        convolved img as 2d np array
    """
    assert img.ndim == 2, "img must be 2d grayscale"
    assert kernel.ndim == 2, "kernel must be 2d"

    H, W = img.shape
    kh, kw = kernel.shape

    flipped_kernel = kernel[::-1, ::-1]

    if padding == "same":
        pad_top = kh // 2
        pad_bot = kh - 1 - pad_top
        pad_l = kw // 2
        pad_r = kw - 1 - pad_l
        out_H, out_W = H, W
    elif padding == "full":
        pad_top = kh - 1
        pad_bot = kh - 1
        pad_l = kw - 1
        pad_r = kw - 1
        out_H, out_W = H + kh - 1, W + kw - 1
    else:
        raise ValueError("padding must be 'same' or 'full'")

    padded = np.pad(img, ((pad_top, pad_bot), (pad_l, pad_r)), mode='constant', constant_values=0)

    output = np.zeros((out_H, out_W), dtype=np.float64)

    for i in range(out_H):
        for j in range(out_W):
            patch = padded[i : i + kh, j : j + kw]
            output[i, j] = np.sum(patch * flipped_kernel)
    
    return output

def conv2d_4loops(img: np.ndarray, kernel: np.ndarray, padding: str="same") -> np.ndarray:
    """
    2d convolution implementation w/ 4 nested loops

    args:
        img: 2d np array, shape (H, W)
        kernel: 2d np array, shape(kh, kw)
        padding: "same"/"full"
    
    returns:
        convolved img as 2d np array
    """
    assert img.ndim == 2, "img must be 2d grayscale"
    assert kernel.ndim == 2, "kernel must be 2d"

    H, W = img.shape
    kh, kw = kernel.shape

    flipped_kernel = kernel[::-1, ::-1]

    if padding == "same":
        pad_top = kh // 2
        pad_bot = kh - 1 - pad_top
        pad_l = kw // 2
        pad_r = kw - 1 - pad_l
        out_H, out_W = H, W
    elif padding == "full":
        pad_top = kh - 1
        pad_bot = kh - 1
        pad_l = kw - 1
        pad_r = kw - 1
        out_H, out_W = H + kh - 1, W + kw - 1
    else:
        raise ValueError("padding must be 'same' or 'full'")

    padded = np.pad(img, ((pad_top, pad_bot), (pad_l, pad_r)), mode='constant', constant_values=0)

    output = np.zeros((out_H, out_W), dtype=np.float64)

    for i in range(out_H):
        for j in range(out_W):
            val = 0.0
            for u in range(kh):
                for v in range(kw):
                    val += padded[i + u, j + v] * flipped_kernel[u, v]
            output[i, j] = val
    
    return output

def benchmark_convolutions(img : np.ndarray):
    """
    compare custom 4-loop, 2-loop, and scipy.signal.convolve2d for accuracy/runtime
    """
    print("\n" + "="*60)
    print("part 1.1: benchmarking convolution implementations")
    print("-" * 60)

    test_img = cv2.resize(img, (128, 128)) if img.shape[0] > 128 else img
    kernel = np.ones((9, 9), dtype=np.float64) / 81.0

    # scipy convolve2d
    t0 = time.time()
    scipy_res = scipy.signal.convolve2d(test_img, kernel, mode='same',
    boundary='fill', fillvalue=0)
    t_scipy = time.time() - t0
    print(f"scipy.signal.convolve2d runtime: {t_scipy * 1000:.2f} ms")

    # 2-loop conv2d
    t0 = time.time()
    conv2_res = conv2d_2loops(test_img, kernel, padding='same')
    t_conv2 = time.time() - t0
    max_diff_2 = np.max(np.abs(conv2_res - scipy_res))
    print(f"Custom 2-loop conv2d runtime:    {t_conv2 * 1000:.2f} ms (Max diff vs scipy: {max_diff_2:.2e})")

    # 4-loop conv2d
    small_img = test_img[:64, :64]
    small_scipy = scipy.signal.convolve2d(small_img, kernel, mode='same',
    boundary='fill', fillvalue=0)
    t0 = time.time()
    conv4_res = conv2d_4loops(small_img, kernel, padding='same')
    t_conv4 = time.time() - t0
    max_diff_4 = np.max(np.abs(conv4_res - small_scipy))
    print(f"Custom 4-loop conv2d runtime (64x64):   {t_conv4 * 1000:.2f} ms (Max diff vs scipy: {max_diff_4:.2e})")
    print("=" * 60)


# 1.2

D_x = np.array([[1, -1]], dtype=np.float64)
D_y = np.array([[1], [-1]], dtype=np.float64)

def compute_finite_diffs(img: np.ndarray, thresh: float=0.25):
    """
    compute partial derivatives in x & y using finite difference operators, gradient magnitude, & binarized edge image
    """
    img_flt = img.astype(np.float64)
    if img_flt.max() > 1.0:
        img_flt /= 255.0

    grad_x = scipy.signal.convolve2d(img_flt, D_x, mode='same', boundary='symm')
    grad_y = scipy.signal.convolve2d(img_flt, D_y, mode='same', boundary='symm')

    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    edge_bin = (grad_mag > thresh).astype(np.float64)
    return grad_x, grad_y, grad_mag, edge_bin


# 1.3

def get_gaussian_kernel_2d(ksize: int=9, sigma: float=1.5) -> np.ndarray:
    """
    generate 2d gaussian kernel w/ cv2.getGaussianKernel
    """
    g1d = cv2.getGaussianKernel(ksize, sigma)
    return g1d @ g1d.T

def run_dog_analysis(img: np.ndarray, ksize: int=9, sigma: float=1.5, thresh: float=0.08):
    img_flt = img.astype(np.float64)
    if img_flt.max() > 1.0:
        img_flt /= 255.0
    
    G = get_gaussian_kernel_2d(ksize, sigma)
    
    img_blurred = scipy.signal.convolve2d(img_flt, G, mode='same', boundary='symm')
    grad_x_blur = scipy.signal.convolve2d(img_blurred, D_x, mode='same', boundary='symm')
    grad_y_blur = scipy.signal.convolve2d(img_blurred, D_y, mode='same', boundary='symm')

    grad_mag_blur = np.sqrt(grad_x_blur**2 + grad_y_blur**2)
    edge_blur = (grad_mag_blur > thresh).astype(np.float64)

    DoG_x = scipy.signal.convolve2d(G, D_x, mode='full', boundary='symm')
    DoG_y = scipy.signal.convolve2d(G, D_y, mode='full', boundary='symm')

    grad_x_dog = scipy.signal.convolve2d(img_flt, DoG_x, mode='same', boundary='symm')
    grad_y_dog = scipy.signal.convolve2d(img_flt, DoG_y, mode='same', boundary='symm')

    grad_mag_dog = np.sqrt(grad_x_dog**2 + grad_y_dog**2)
    edge_dog = (grad_mag_dog > thresh).astype(np.float64)

    margin = ksize
    diff_x = np.max(np.abs(grad_x_blur[margin:-margin, margin:-margin] - grad_x_dog[margin:-margin, margin:-margin]))
    diff_y = np.max(np.abs(grad_y_blur[margin:-margin, margin:-margin] - grad_y_dog[margin:-margin, margin:-margin]))
    diff_mag = np.max(np.abs(grad_mag_blur[margin:-margin, margin:-margin] - grad_mag_dog[margin:-margin, margin:-margin]))

    print("\n" + "="*60)
    print("1.3: verifying equivalence (blur+diff vs DoG)")
    print(f"max interior abs diff in Ix:    {diff_x:.2e}")
    print(f"max interior abs diff in Iy:    {diff_y:.2e}")
    print(f"max interior abs diff in |grad|: {diff_mag:.2e}")
    print("="*60)

    return {
        "G": G,
        "DoG_x": DoG_x,
        "DoG_y": DoG_y,
        "img_blurred": img_blurred,
        "grad_x_blur": grad_x_blur,
        "grad_y_blur": grad_y_blur,
        "grad_mag_blur": grad_mag_blur,
        "edge_blur": edge_blur,
        "grad_x_dog": grad_x_dog,
        "grad_y_dog": grad_y_dog,
        "grad_mag_dog": grad_mag_dog,
        "edge_dog": edge_dog
    }
    
def run_p1(img_path: str="inputs/1 - cameraman.jpg", out_dir: str="output/part1", myself_path: str="inputs/1.1 - myself.jpeg"):
    """
    execute part 1 & save generated figures
    """
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(img_path):
        img_path = "inputs/cameraman.png" if os.path.exists("inputs/cameraman.png") else "cameraman.png"

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {img_path}")
    img_flt = img.astype(np.float64) / 255.0

    # 1.1
    benchmark_convolutions(img_flt)

    box_filter = np.ones((9, 9), dtype=np.float64) / 81.0
    img_box = conv2d_2loops(img_flt, box_filter, padding="same")
    img_dx = conv2d_2loops(img_flt, D_x, padding="same")
    img_dy = conv2d_2loops(img_flt, D_y, padding="same")
    
    plt.figure(figsize=(15, 4))
    plt.subplot(1, 4, 1)
    plt.imshow(img_flt, cmap='gray') 
    plt.title("Original Grayscale"); plt.axis('off')
    plt.subplot(1, 4, 2)
    plt.imshow(img_box, cmap='gray')
    plt.title("9x9 Box Filter"); plt.axis('off')
    plt.subplot(1, 4, 3)
    plt.imshow(img_dx, cmap='gray'); 
    plt.title(r"$D_x$ Filter"); plt.axis('off')
    plt.subplot(1, 4, 4)
    plt.imshow(img_dy, cmap='gray')
    plt.title(r"$D_y$ Filter"); plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "part1_1_custom_convolutions.png"), dpi=200)
    plt.close()

    if os.path.exists(myself_path):
        myself_raw = cv2.imread(myself_path, cv2.IMREAD_GRAYSCALE)
        h, w = myself_raw.shape
        scale = 512.0 / max(h, w)
        myself_flt = cv2.resize(myself_raw, (int(w * scale), int(h * scale))).astype(np.float64) / 255.0

        my_box = conv2d_2loops(myself_flt, box_filter, padding="same")
        my_dx = conv2d_2loops(myself_flt, D_x, padding="same")
        my_dy = conv2d_2loops(myself_flt, D_y, padding="same")

        plt.figure(figsize=(15, 4))
        plt.subplot(1, 4, 1)
        plt.imshow(myself_flt, cmap='gray')
        plt.title("Original (Myself)"); plt.axis('off')
        plt.subplot(1, 4, 2)
        plt.imshow(my_box, cmap='gray')
        plt.title("9x9 Box Filter"); plt.axis('off')
        plt.subplot(1, 4, 3)
        plt.imshow(my_dx, cmap='gray')
        plt.title(r"$D_x$ Filter"); plt.axis('off')
        plt.subplot(1, 4, 4)
        plt.imshow(my_dy, cmap='gray')
        plt.title(r"$D_y$ Filter"); plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "part1_1_myself_convolutions.png"), dpi=200)
        plt.close()

    thres_fd = 0.25
    gx_fd, gy_fd, gmag_fd, edge_fd = compute_finite_diffs(img_flt, thresh=thres_fd)

    thresholds = [0.10, 0.25, 0.40, 0.55]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, t in zip(axes, thresholds):
        e = (gmag_fd > t).astype(np.float64)
        ax.imshow(e, cmap='gray')
        ax.set_title(f"Threshold = {t}")
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "part1_2_threshold_comparison.png"), dpi=200)
    plt.close()

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(gx_fd, cmap='gray')
    axes[0].set_title(r"$D_x$ Filter")
    axes[0].axis('off')
    axes[1].imshow(gy_fd, cmap='gray')
    axes[1].set_title(r"$D_y$ Filter")
    axes[1].axis('off')
    axes[2].imshow(gmag_fd, cmap='gray')
    axes[2].set_title(r"Gradient Magnitude")
    axes[2].axis('off')
    axes[3].imshow(edge_fd, cmap='gray')
    axes[3].set_title(f"Edge Detection (Threshold = {thres_fd})")
    axes[3].axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "part1_2_finite_diffs.png"), dpi=200)
    plt.close()

    dog_results = run_dog_analysis(img_flt, ksize=9, sigma=1.5, thresh=0.08)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    im0 = axes[0].imshow(dog_results["grad_mag_blur"], cmap='gray')
    axes[0].set_title("Gradient Magnitude (Blurred + Diff)")
    axes[0].axis('off')
    plt.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].imshow(dog_results["grad_mag_dog"], cmap='gray')
    axes[1].set_title("Gradient Magnitude (DoG)")
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    im2 = axes[2].imshow(dog_results["edge_dog"], cmap='gray')
    axes[2].set_title("Edge Detection (DoG)")
    axes[2].axis('off')
    plt.colorbar(im2, ax=axes[2], fraction=0.046)
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "part1_3_dog_results.png"), dpi=200)
    plt.close()

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    im0 = axes[0].imshow(dog_results["G"], cmap='viridis')
    axes[0].set_title("Gaussian Kernel G")
    axes[0].axis('off')
    plt.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].imshow(dog_results["DoG_x"], cmap='viridis')
    axes[1].set_title("DoG_x Kernel")
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    im2 = axes[2].imshow(dog_results["DoG_y"], cmap='viridis')
    axes[2].set_title("DoG_y Kernel")
    axes[2].axis('off')
    plt.colorbar(im2, ax=axes[2], fraction=0.046)
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "part1_3_gaussian_dog_kernels.png"), dpi=200)
    plt.close()

    fig, axes = plt.subplots(1,2, figsize=(10, 5))
    axes[0].imshow(edge_fd, cmap='gray')
    axes[0].set_title("Edge Detection (Finite Difference)")
    axes[0].axis('off')
    axes[1].imshow(dog_results["edge_dog"], cmap='gray')
    axes[1].set_title("Edge Detection (DoG)")
    axes[1].axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "part1_3_edge_comparison.png"), dpi=200)
    plt.close()

# 2.1

def filter2d(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    applies 2d filter using convolve2d / fftconvolve w/ boundary reflection/handling
    handles both 2d grayscale & 3d rgb images
    """
    if kernel.shape[0] * kernel.shape[1] > 225 or (img.shape[0] * img.shape[1] > 250000):
        if img.ndim == 2:
            return scipy.signal.fftconvolve(img, kernel, mode='same')
        elif img.ndim == 3:
            channels = [
                scipy.signal.fftconvolve(img[:, :, c], kernel, mode='same') for c in range(img.shape[2])
            ]
            return np.stack(channels, axis=-1)
    else:
        if img.ndim == 2:
            return scipy.signal.convolve2d(img, kernel, mode='same', boundary='symm')
        elif img.ndim == 3:
            channels = [
                scipy.signal.convolve2d(img[:, :, c], kernel, mode='same', boundary='symm') for c in range(img.shape[2])
            ]
            return np.stack(channels, axis=-1)
        else:
            raise ValueError("img must be 2d grayscale/3d rgb")

def unsharp_mask(img: np.ndarray, ksize: int=9, sigma: float=1.5, alpha: float=1.0):
    """
    sharpens img using unsharp masking technique:
        1. low pass fitler (blur) w/ gaussian
        2. extract high freq
        3. add scaled high freqs

    derives single combined convolution kernel:
        K_unsharp = (1 + alpha) * e - alpha * G
    """
    img_flt = img.astype(np.float64)
    if img_flt.max() > 1.0:
        img_flt /= 255.0
    
    G = get_gaussian_kernel_2d(ksize, sigma)
    img_blur = filter2d(img_flt, G)
    img_high = img_flt - img_blur
    img_sharp = np.clip(img_flt + alpha * img_high, 0.0, 1.0)

    e = np.zeros_like(G)
    center_y, center_x = G.shape[0] // 2, G.shape[1] // 2
    e[center_y, center_x] = 1.0

    K_unsharp = (1.0 + alpha) * e - alpha * G
    img_sharp_single = np.clip(filter2d(img_flt, K_unsharp), 0.0, 1.0)

    max_diff = np.max(np.abs(img_sharp - img_sharp_single))

    return {
        "img_blur": img_blur,
        "img_high": img_high,
        "img_sharp": img_sharp,
        "img_sharp_single": img_sharp_single,
        "K_unsharp": K_unsharp,
        "max_diff": max_diff
    }

def eval_sharpening_resilience(sharp_img: np.ndarray, ksize: int=9, sigma: float=1.5, alpha: float=1.5):
    """
    eval experiment:
        1. start w/ sharp img
        2. artificially blur
        3. apply unsharp masking
        4. compare
    """
    img_flt = sharp_img.astype(np.float64)
    if img_flt.max() > 1.0:
        img_flt /= 255.0

    G = get_gaussian_kernel_2d(ksize, sigma)
    blurred = filter2d(img_flt, G)
    resharpened = unsharp_mask(blurred, ksize=ksize, sigma=sigma, alpha=alpha)["img_sharp"]

    return {
        "original" : img_flt,
        "blurred": blurred,
        "resharpened": resharpened
    }

# 2.2

def align_imgs(img1: np.ndarray, img2: np.ndarray, pts1=None, pts2=None) -> tuple:
    """
    align im2 -> im1 on two corresponding points
    no points: crop/resize to match
    """
    if pts1 is None or pts2 is None:
        h = min(img1.shape[0], img2.shape[0])
        w = min(img1.shape[1], img2.shape[1])
        return img1[:h, :w], img2[:h, :w]
    
    p1, p2 = np.array(pts1, dtype=np.float32), np.array(pts2, dtype=np.float32)
    dx1, dy1 = p1[1, 0] - p1[0, 0], p1[1, 1] - p1[0, 1]
    dx2, dy2 = p2[1, 0] - p2[0, 0], p2[1, 1] - p2[0, 1]
    
    angle1 = np.arctan2(dy1, dx1) * 180.0 / np.pi
    angle2 = np.arctan2(dy2, dx2) * 180.0 / np.pi
    angle = angle1 - angle2

    scale = np.sqrt(dx1** 2 + dy1**2) / np.sqrt(dx2**2 + dy2**2)
    center = tuple(p2[0])

    M = cv2.getRotationMatrix2D(center, angle, scale)
    M[0, 2] += (p1[0, 0] - p2[0, 0])
    M[1, 2] += (p1[0, 1] - p2[0, 1])

    img2_aligned = cv2.warpAffine(img2, M, (img1.shape[1], img1.shape[0]), flags = cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return img1, img2_aligned

def crop_black_borders(im1: np.ndarray, im2: np.ndarray, thresh: float=0.005, min_coverage: float=0.5) -> tuple:
    """
    crops out black rotation/recentering padding from aligned image pairs
    """
    m1 = np.any(im1 > thresh, axis=-1) if im1.ndim == 3 else (im1 > thresh)
    m2 = np.any(im2 > thresh, axis=-1) if im2.ndim == 3 else (im2 > thresh)
    valid = m1 & m2

    row_cov = valid.mean(axis=1)
    col_cov = valid.mean(axis=0)

    r_idx = np.where(row_cov > min_coverage)[0]
    c_idx = np.where(col_cov > min_coverage)[0]

    if len(r_idx) > 0 and len(c_idx) > 0:
        r0, r1 = int(r_idx[0]), int(r_idx[-1]) + 1
        c0, c1 = int(c_idx[0]), int(c_idx[-1]) + 1
        return im1[r0:r1, c0:c1], im2[r0:r1, c0:c1]
    return im1, im2

def compute_fft_log_mag(img: np.ndarray) -> np.ndarray:
    """
    compute log magnitude of 2d fourir transform for freq analysis
    """
    if img.ndim == 3:
        img_gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float64) / 255.0
    else:
        img_gray = img.astype(np.float64)
    f = np.fft.fft2(img_gray)
    fshift = np.fft.fftshift(f)
    return np.log(np.abs(fshift) + 1e-8)

def create_hybrid_img(
    img_low: np.ndarray,
    img_high: np.ndarray,
    sigma_low: float=6.0,
    sigma_high: float=3.0,
    scale_low: float=1.0,
    scale_high: float=1.0,
    ksize_scale: int=6
):
    """
    create a hybrid img:
        - low pass filter im_low w/ gaussian
        - high pass filter im_high
        - hybrid = scale_low * low + scale_high * high
    """
    img1 = img_low.astype(np.float64)
    if img1.max() > 1.0:
        img1 /= 255.0

    img2 = img_high.astype(np.float64)
    if img2.max() > 1.0:
        img2 /= 255.0
    
    h = min(img1.shape[0], img2.shape[0])
    w = min(img1.shape[1], img2.shape[1])
    img1, img2 = img1[:h, :w], img2[:h, :w]

    ksize_low = int(2 * np.ceil(ksize_scale * sigma_low) + 1)
    G_low = get_gaussian_kernel_2d(ksize_low, sigma_low)
    low_freq = filter2d(img1, G_low)

    ksize_high = int(2 * np.ceil(ksize_scale * sigma_high) + 1)
    G_high = get_gaussian_kernel_2d(ksize_high, sigma_high)
    high_freq = img2 - filter2d(img2, G_high)

    hybrid = np.clip(scale_low * low_freq + scale_high * high_freq, 0.0, 1.0)

    return {
        "img_low": img1,
        "img_high": img2,
        "low_freq": low_freq,
        "high_freq": high_freq,
        "hybrid": hybrid,
        "fft_img1": compute_fft_log_mag(img1),
        "fft_img2": compute_fft_log_mag(img2),
        "fft_low": compute_fft_log_mag(low_freq),
        "fft_high": compute_fft_log_mag(high_freq),
        "fft_hybrid": compute_fft_log_mag(hybrid)
    }

# 2.3

def build_gaussian_stack(img: np.ndarray, num_levels: int=5, sigma_start: float=2.0, ksize_scale: int=6) -> list:
    """
    construct gaussian stack w/o subsampling:
        level 0: original img
        level 1: img blurred w/ increasing gaussian sigma
    """
    img_flt = img.astype(np.float64)
    if img_flt.max() > 1.0:
        img_flt /= 255.0
    
    gaussian_stack = [img_flt]
    for i in range(1, num_levels):
        sigma = sigma_start * (2 ** (i-1))
        ksize = int(2 * np.ceil(ksize_scale * sigma) + 1)
        G = get_gaussian_kernel_2d(ksize, sigma)
        blurred = filter2d(img_flt, G)
        gaussian_stack.append(blurred)
    
    return gaussian_stack

def build_laplacian_stack(gaussian_stack: list) -> list:
    """
    construct laplacian stack from gaussian stack
    """
    num_levels = len(gaussian_stack)
    laplacian_stack = []

    for i in range(num_levels - 1):
        laplacian_stack.append(gaussian_stack[i] - gaussian_stack[i + 1])
    
    laplacian_stack.append(gaussian_stack[-1])
    return laplacian_stack

def reconstruct_from_laplacian_stack(laplacian_stack: list) -> np.ndarray:
    """
    reconstruct original img from laplacian stack
    """
    reconstructed = np.zeros_like(laplacian_stack[0])
    for level in laplacian_stack:
        reconstructed += level
    return np.clip(reconstructed, 0.0, 1.0)

# 2.4

def blend_imgs_multires(
    im_a: np.ndarray,
    im_b: np.ndarray,
    mask: np.ndarray,
    num_levels: int=5,
    sigma_start: float=2.0
):
    """
    multires spline blending:
        1. build laplacian for A, B
        2. build gaussian for M
        3. blend each band
        4. reconstruct blended img
    """
    img_a = im_a.astype(np.float64)
    if img_a.max() > 1.0:
        img_a /= 255.0

    img_b = im_b.astype(np.float64)
    if img_b.max() > 1.0:
        img_b /= 255.0
    
    msk = mask.astype(np.float64)
    if msk.max() > 1.0:
        msk /= 255.0
    
    h = min(img_a.shape[0], img_b.shape[0], msk.shape[0])
    w = min(img_a.shape[1], img_b.shape[1], msk.shape[1])
    img_a, img_b, msk = img_a[:h,:w], img_b[:h,:w], msk[:h,:w]

    if img_a.ndim == 3 and msk.ndim == 2:
        msk = np.repeat(msk[:, :, np.newaxis], 3, axis=2)

    gauss_a = build_gaussian_stack(img_a, num_levels, sigma_start)
    gauss_b = build_gaussian_stack(img_b, num_levels, sigma_start)
    gauss_m = build_gaussian_stack(msk, num_levels, sigma_start)

    lap_a = build_laplacian_stack(gauss_a)
    lap_b = build_laplacian_stack(gauss_b)

    blended = []
    for la, lb, gm in zip(lap_a, lap_b, gauss_m):
        blended.append(gm * la + (1.0 - gm) * lb)
    
    final_blend = reconstruct_from_laplacian_stack(blended)
    
    return {
        "gauss_a": gauss_a,
        "gauss_b": gauss_b,
        "gauss_m": gauss_m,
        "lap_a": lap_a,
        "lap_b": lap_b,
        "final_blend": final_blend
    }

def create_vertical_mask(shape: tuple) -> np.ndarray:
    H, W = shape[0], shape[1]
    msk = np.zeros((H, W), dtype=np.float64)
    msk[:, :W // 2] = 1.0
    return msk

def create_circular_mask(shape: tuple, radius: int=None, center: tuple=None) -> np.ndarray:
    H, W = shape[0], shape[1]
    if center is None:
        center = (W // 2, H // 2)
    if radius is None:
        radius = min(H, W) // 3
    
    Y, X = np.ogrid[:H, :W]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
    return (dist_from_center <= radius).astype(np.float64)

def run_p2(out_dir: str="output/part2", use_cache: bool=True):
    os.makedirs(out_dir, exist_ok=True)
    print("\n" + "="*60)
    print("part 2: fun with frequencies")
    print("-" * 60)

    taj_path = "inputs/2 - taj.jpg" if os.path.exists("inputs/2 - taj.jpg") else "inputs/taj.jpg"
    if os.path.exists(taj_path):
        taj = cv2.imread(taj_path)
        taj_rgb = cv2.cvtColor(taj, cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0

        sharp_res = unsharp_mask(taj_rgb, ksize=9, sigma=1.5, alpha=1.5)
        print(f"part 2.1: single vs 2-step convolution max diff: {sharp_res['max_diff']:.2e}")

        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        axes[0].imshow(taj_rgb)
        axes[0].set_title("original")
        axes[0].axis("off")
        
        axes[1].imshow(sharp_res["img_blur"])
        axes[1].set_title("blurred")
        axes[1].axis("off")

        axes[2].imshow(sharp_res["img_sharp"])
        axes[2].set_title(r"sharpened ($\alpha=1.5$)")
        axes[2].axis("off")

        axes[3].imshow(sharp_res["K_unsharp"], cmap="gray")
        axes[3].set_title("unsharp kernel")
        axes[3].axis("off")

        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "2.1_sharpening.png"), dpi=200)
        plt.close()

        alphas = [0.5, 1.0, 2.0, 4.0]
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        for ax, a in zip(axes, alphas):
            res_a = unsharp_mask(taj_rgb, ksize=9, sigma=1.5, alpha=a)
            ax.imshow(res_a["img_sharp"])
            ax.set_title(rf"sharpened ($\alpha={a}$)")
            ax.axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "2.1_sharpening_alphas.png"), dpi=200)
        plt.close()

    sharp_player_path = "inputs/2.1 - sharp.jpeg"
    if os.path.exists(sharp_player_path):
        sharp_img = cv2.imread(sharp_player_path)
        sharp_rgb = cv2.cvtColor(sharp_img, cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0
        h, w = sharp_rgb.shape[:2]
        scale = 800.0 / max(h, w)
        sharp_resized = cv2.resize(sharp_rgb, (int(w * scale), int(h * scale)))

        eval_res = eval_sharpening_resilience(sharp_resized, ksize=9, sigma=1.5, alpha=2.0)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(eval_res["original"])
        axes[0].set_title("original sharp (Shaedon Sharpe)")
        axes[0].axis("off")

        axes[1].imshow(eval_res["blurred"])
        axes[1].set_title("re-blurred")
        axes[1].axis("off")

        axes[2].imshow(eval_res["resharpened"])
        axes[2].set_title(r"resharpened ($\alpha=2.0$)")
        axes[2].axis("off")

        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "2.1_sharpening_resilience.png"), dpi=200)
        plt.close()

    print("part 2.2: hybrid images")

    def process_hybrid_pair(im1_path, im2_path, name1, name2, sigma_low, sigma_high, out_name, scale_low=1.0, scale_high=1.0):
        if not (os.path.exists(im1_path) and os.path.exists(im2_path)):
            print(f"Skipping {name1}+{name2} because images are missing. Place them at {im1_path} and {im2_path}.")
            return

        cache_f1 = f"inputs/aligned_{name1}.npy"
        cache_f2 = f"inputs/aligned_{name2}.npy"
        if use_cache and os.path.exists(cache_f1) and os.path.exists(cache_f2):
            print(f"Loading aligned {name1} and {name2} from cache...")
            im1_aligned = np.load(cache_f1)
            im2_aligned = np.load(cache_f2)
        else:
            print(f"Aligning {name1} and {name2}... PLEASE CLICK 2 POINTS ON EACH IMAGE IN THE POPUP.")
            def read_rgb(p):
                raw = cv2.imread(p, cv2.IMREAD_UNCHANGED)
                if raw.ndim == 3 and raw.shape[2] == 4:
                    alpha = raw[:, :, 3:4].astype(np.float64) / 255.0
                    rgb = cv2.cvtColor(raw[:, :, :3], cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0
                    return rgb * alpha + (1.0 - alpha) * 1.0
                elif raw.ndim == 3:
                    return cv2.cvtColor(raw, cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0
                else:
                    g = raw.astype(np.float64) / 255.0
                    return np.stack([g, g, g], axis=-1)

            im1_rgb = read_rgb(im1_path)
            im2_rgb = read_rgb(im2_path)
            
            im1_aligned, im2_aligned = align_images(im1_rgb, im2_rgb)
            np.save(cache_f1, im1_aligned)
            np.save(cache_f2, im2_aligned)

        im1_aligned, im2_aligned = crop_black_borders(im1_aligned, im2_aligned)

        hybrid_res = create_hybrid_img(im1_aligned, im2_aligned, sigma_low=sigma_low, sigma_high=sigma_high, scale_low=scale_low, scale_high=scale_high)

        fig, axes = plt.subplots(1, 5, figsize=(20, 4))
        if hybrid_res["img_low"].ndim == 3:
            axes[0].imshow(hybrid_res["img_low"])
        else:
            axes[0].imshow(hybrid_res["img_low"], cmap="gray")
        axes[0].set_title(f"low-pass ({name1})")
        axes[0].axis("off")

        if hybrid_res["img_high"].ndim == 3:
            axes[1].imshow(hybrid_res["img_high"])
        else:
            axes[1].imshow(hybrid_res["img_high"], cmap="gray")
        axes[1].set_title(f"high-pass ({name2})")
        axes[1].axis("off")

        if hybrid_res["low_freq"].ndim == 3:
            axes[2].imshow(np.clip(hybrid_res["low_freq"], 0, 1))
        else:
            axes[2].imshow(hybrid_res["low_freq"], cmap="gray")
        axes[2].set_title(f"low frequencies ($\\sigma={sigma_low}$)")
        axes[2].axis("off")

        if hybrid_res["high_freq"].ndim == 3:
            axes[3].imshow(np.clip(hybrid_res["high_freq"] + 0.5, 0.0, 1.0))
        else:
            axes[3].imshow(np.clip(hybrid_res["high_freq"] + 0.5, 0.0, 1.0), cmap="gray")
        axes[3].set_title(f"high frequencies ($\\sigma={sigma_high}$)")
        axes[3].axis("off")

        if hybrid_res["hybrid"].ndim == 3:
            axes[4].imshow(hybrid_res["hybrid"])
        else:
            axes[4].imshow(hybrid_res["hybrid"], cmap="gray")
        axes[4].set_title(f"hybrid {name1}-{name2}")
        axes[4].axis("off")

        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"2.2_hybrid_{out_name}_process.png"), dpi=200)
        plt.close()

        fig, axes = plt.subplots(1, 5, figsize=(20, 4))
        axes[0].imshow(hybrid_res["fft_img1"], cmap="magma")
        axes[0].set_title(f"FFT: {name1}")
        axes[0].axis("off")

        axes[1].imshow(hybrid_res["fft_img2"], cmap="magma")
        axes[1].set_title(f"FFT: {name2}")
        axes[1].axis("off")

        axes[2].imshow(hybrid_res["fft_low"], cmap="magma")
        axes[2].set_title("FFT: low-pass")
        axes[2].axis("off")

        axes[3].imshow(hybrid_res["fft_high"], cmap="magma")
        axes[3].set_title("FFT: high-pass")
        axes[3].axis("off")

        axes[4].imshow(hybrid_res["fft_hybrid"], cmap="magma")
        axes[4].set_title("FFT: hybrid")
        axes[4].axis("off")

        plt.suptitle("2D Fourier Transform (Log Magnitude) Frequency Analysis", fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"2.2_hybrid_{out_name}_fft.png"), dpi=200)
        plt.close()

    # Derek & Nutmeg
    process_hybrid_pair("inputs/2.2 - derek.jpg", "inputs/2.2 - nutmeg.jpg", "derek", "nutmeg", 7.0, 4.0, "derek_nutmeg")
    # Cat & Leopard
    leopard_p = "inputs/2.2 - leopard.png" if os.path.exists("inputs/2.2 - leopard.png") else "inputs/2.2 - leopard.jpg"
    process_hybrid_pair("inputs/2.2 - cat.jpg", leopard_p, "cat", "leopard", 6.0, 3.0, "cat_leopard")
    # Apple & Skull
    process_hybrid_pair("inputs/2.2 - apple.png", "inputs/2.2 - skull.png", "apple", "skull", 12.0, 8.0, "apple_skull", scale_low=1.0, scale_high=1.8)

    print("part 2.3 & 2.4: multiresolution blending")
    apple_path = "inputs/2 - apple.jpeg"
    orange_path = "inputs/2 - orange.jpeg"
    if os.path.exists(apple_path) and os.path.exists(orange_path):
        apple = cv2.imread(apple_path)
        apple_rgb = cv2.cvtColor(apple, cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0

        orange = cv2.imread(orange_path)
        orange_rgb = cv2.cvtColor(orange, cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0

        mask_v = create_vertical_mask(apple_rgb.shape)
        oraple_res = blend_imgs_multires(apple_rgb, orange_rgb, mask_v, num_levels=5, sigma_start=2.0)

        num_lvls = len(oraple_res["gauss_a"])
        fig, axes = plt.subplots(1, num_lvls, figsize=(3 * num_lvls, 3))
        for i, lvl in enumerate(oraple_res["gauss_a"]):
            axes[i].imshow(lvl)
            axes[i].set_title(f"Gauss Level {i}")
            axes[i].axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "2.3_gaussian_stack.png"), dpi=200)
        plt.close()

        fig, axes = plt.subplots(1, num_lvls, figsize=(3 * num_lvls, 3))
        for i, lvl in enumerate(oraple_res["lap_a"]):
            axes[i].imshow(np.clip(lvl + 0.5, 0.0, 1.0))
            axes[i].set_title(f"Laplacian Level {i}")
            axes[i].axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "2.3_laplacian_stack.png"), dpi=200)
        plt.close()

        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        axes[0].imshow(apple_rgb)
        axes[0].set_title("apple")
        axes[0].axis("off")

        axes[1].imshow(orange_rgb)
        axes[1].set_title("orange")
        axes[1].axis("off")

        axes[2].imshow(mask_v, cmap="gray")
        axes[2].set_title("vertical mask")
        axes[2].axis("off")

        axes[3].imshow(oraple_res["final_blend"])
        axes[3].set_title("multires blend (Oraple)")
        axes[3].axis("off")

        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "2.4_multires_blend.png"), dpi=200)
        plt.close()

    helmet1_path = "inputs/2.4 - football helmet.jpg"
    helmet2_path = "inputs/2.4 - batting helmet.jpg"
    if os.path.exists(helmet1_path) and os.path.exists(helmet2_path):
        helmet1 = cv2.imread(helmet1_path)
        helmet1_rgb = cv2.cvtColor(helmet1, cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0

        helmet2 = cv2.imread(helmet2_path)
        helmet2_rgb = cv2.cvtColor(helmet2, cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0

        h_h = min(helmet1_rgb.shape[0], helmet2_rgb.shape[0])
        w_h = min(helmet1_rgb.shape[1], helmet2_rgb.shape[1])
        helmet1_rgb = cv2.resize(helmet1_rgb, (w_h, h_h))
        helmet2_rgb = cv2.resize(helmet2_rgb, (w_h, h_h))

        mask_helmet_v = create_vertical_mask(helmet1_rgb.shape)
        helmet_blend_res = blend_imgs_multires(helmet1_rgb, helmet2_rgb, mask_helmet_v, num_levels=5, sigma_start=2.0)

        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        axes[0].imshow(helmet1_rgb)
        axes[0].set_title("football helmet")
        axes[0].axis("off")

        axes[1].imshow(helmet2_rgb)
        axes[1].set_title("batting helmet")
        axes[1].axis("off")

        axes[2].imshow(mask_helmet_v, cmap="gray")
        axes[2].set_title("vertical mask")
        axes[2].axis("off")

        axes[3].imshow(helmet_blend_res["final_blend"])
        axes[3].set_title("multires blend (Helmets)")
        axes[3].axis("off")

        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "2.4_helmet_multires_blend.png"), dpi=200)
        plt.close()

    galaxy_path = "inputs/2.4 - galaxy.jpg"
    leopard_path = "inputs/2.2 - leopard.png" if os.path.exists("inputs/2.2 - leopard.png") else "inputs/2.2 - leopard.jpg"
    if os.path.exists(galaxy_path) and os.path.exists(leopard_path):
        leopard = cv2.imread(leopard_path)
        leopard_rgb = cv2.cvtColor(leopard, cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0

        galaxy = cv2.imread(galaxy_path)
        galaxy_rgb = cv2.cvtColor(galaxy, cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0

    print("Please click the LEFT eye, then the RIGHT eye of the leopard.")
    plt.imshow(leopard_rgb)
    plt.title("Click LEFT eye, then RIGHT eye")
    pts = plt.ginput(2, timeout=-1)
    plt.close()

    if len(pts) < 2:
        print("Didn't get 2 points, falling back to defaults")
        h_s, w_s = leopard_rgb.shape[:2]
        pt_left = (int(w_s * 0.4), int(h_s * 0.4))
        pt_right = (int(w_s * 0.6), int(h_s * 0.4))
    else:
        pt_left, pt_right = pts

    galaxy_aligned = np.zeros_like(leopard_rgb)
    eye_dist = np.sqrt((pt_left[0] - pt_right[0])**2 + (pt_left[1] - pt_right[1])**2)
    eye_radius = int(eye_dist * 0.22)
    galaxy_size = int(eye_radius * 2.5)
    galaxy_resized = cv2.resize(galaxy_rgb, (galaxy_size, galaxy_size))

    def paste_center(bg, fg, cx, cy):
        h, w = fg.shape[:2]
        H, W = bg.shape[:2]
        y1 = max(0, int(cy - h/2))
        y2 = min(H, int(cy + h/2))
        x1 = max(0, int(cx - w/2))
        x2 = min(W, int(cx + w/2))
        fy1 = max(0, int(h/2 - cy))
        fx1 = max(0, int(w/2 - cx))
        bg[y1:y2, x1:x2] = fg[fy1:fy1+(y2-y1), fx1:fx1+(x2-x1)]

    paste_center(galaxy_aligned, galaxy_resized, pt_left[0], pt_left[1])
    paste_center(galaxy_aligned, galaxy_resized, pt_right[0], pt_right[1])

    mask_circ = np.zeros(leopard_rgb.shape[:2], dtype=np.float64)
    Y, X = np.ogrid[:leopard_rgb.shape[0], :leopard_rgb.shape[1]]
    
    dist_left = np.sqrt((X - pt_left[0])**2 + (Y - pt_left[1])**2)
    mask_circ[dist_left <= eye_radius] = 1.0
    
    dist_right = np.sqrt((X - pt_right[0])**2 + (Y - pt_right[1])**2)
    mask_circ[dist_right <= eye_radius] = 1.0

    blend_galaxy = blend_imgs_multires(galaxy_aligned, leopard_rgb, mask_circ, num_levels=5, sigma_start=2.0)

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    axes[0].imshow(leopard_rgb)
    axes[0].set_title("leopard")
    axes[0].axis("off")

    axes[1].imshow(galaxy_aligned)
    axes[1].set_title("galaxies placed")
    axes[1].axis("off")

    axes[2].imshow(mask_circ, cmap="gray")
    axes[2].set_title("two-eye mask")
    axes[2].axis("off")

    axes[3].imshow(blend_galaxy["final_blend"])
    axes[3].set_title("galaxies in leopard eyes")
    axes[3].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "2.4_irregular_blend.png"), dpi=200)
    plt.close()

    print(f"\nPart 2 execution finished. Outputs saved to {out_dir}!")


if __name__ == "__main__":
    import sys
    use_cache = "--no-cache" not in sys.argv
    run_p1("inputs/1 - cameraman.jpg", "output/part1", "inputs/1.1 - myself.jpeg")
    run_p2("output/part2", use_cache=use_cache)