"""
Combined filter bank method: applies 91 handcrafted filters, pools them
to 273 maps, then runs FFT autocorrelation to find the tile period.
"""

import math
import sys, os
import numpy as np
import cv2
from scipy.ndimage import uniform_filter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import (high_pass_filter, fft_autocorrelation,
                   find_period_from_profile, multiscale_pool)


# built once on first call, reused after that
_gabor_cache = None
_lm_cache = None
_schmid_cache = None


def _build_gabor_filters(cfg):
    """
    Build 24 Gabor filter kernels (3 scales x 8 orientations).

    Params
        cfg: dict, filterbank settings from config.yaml

    Returns
        list of 2D numpy arrays, each one a Gabor kernel
    """
    filters = []
    ksize = cfg.get("gabor_kernel_size", 21)
    for sigma in cfg.get("gabor_scales", [3, 5, 9]):
        lamda = sigma * 2.5
        for i in range(cfg.get("gabor_orientations", 8)):
            theta = i * math.pi / cfg.get("gabor_orientations", 8)
            k = cv2.getGaborKernel((ksize, ksize), sigma, theta, lamda,
                                   0.5, 0, cv2.CV_64F)
            k /= np.abs(k).sum() + 1e-10
            filters.append(k)
    return filters


def _build_lm_bank(cfg):
    """
    Build 48 Leung-Malik filters (18 bars + 18 edges + 4 Gaussians + 8 LoG).

    Params
        cfg: dict, filterbank settings from config.yaml

    Returns
        3D numpy array
    """
    sup = cfg.get("lm_support", 11)
    n_orient = cfg.get("lm_n_orientations", 6)
    scalex = np.sqrt(2) * np.array([0.1, 0.2, 0.3])

    n_bar = len(scalex) * n_orient
    n_edge = n_bar
    nf = n_bar + n_edge + 12

    F = np.zeros((sup, sup, nf))
    hsup = (sup - 1) / 2
    x, y = np.meshgrid(np.arange(-hsup, hsup + 1),
                        np.arange(-hsup, hsup + 1))
    orgpts = np.array([x.flatten(), y.flatten()])

    def g1d(sigma, mean, x, order):
        """1D Gaussian or its 1st/2nd derivative."""
        x_ = np.array(x) - mean
        var = sigma ** 2
        g = (1 / np.sqrt(2 * np.pi * var)) * np.exp(-x_ * x_ / (2 * var))
        if order == 0:
            return g
        elif order == 1:
            return -g * x_ / var
        return g * (x_ * x_ - var) / (var ** 2)

    # Step 1: build oriented bars and edges
    count = 0
    for scale in range(len(scalex)):
        for orient in range(n_orient):
            angle = math.pi * orient / n_orient
            c, s = math.cos(angle), math.sin(angle)
            rotpts = np.array([[c, -s], [s, c]]) @ orgpts

            gx = g1d(3 * scalex[scale], 0, rotpts[0], 0)
            gy = g1d(scalex[scale], 0, rotpts[1], 1)
            F[:, :, count] = (gx * gy).reshape(sup, sup)

            gy2 = g1d(scalex[scale], 0, rotpts[1], 2)
            F[:, :, count + n_edge] = (gx * gy2).reshape(sup, sup)
            count += 1

    # Step 2: build isotropic Gaussians and LoG filters
    count = n_bar + n_edge
    sc = np.sqrt(2) * np.array([0.1, 0.2, 0.3, 0.4])

    for i in range(len(sc)):
        var = sc[i] ** 2
        F[:, :, count] = ((1 / np.sqrt(2 * np.pi * var))
                          * np.exp(-(x**2 + y**2) / (2 * var)))
        count += 1

    for i in range(len(sc)):
        var = sc[i] ** 2
        g2d = ((1 / np.sqrt(2 * np.pi * var))
               * np.exp(-(x**2 + y**2) / (2 * var)))
        F[:, :, count] = g2d * ((x**2 + y**2) - var) / (var ** 2)
        count += 1

    for i in range(len(sc)):
        var = (3 * sc[i]) ** 2
        g2d = ((1 / np.sqrt(2 * np.pi * var))
               * np.exp(-(x**2 + y**2) / (2 * var)))
        F[:, :, count] = g2d * ((x**2 + y**2) - var) / (var ** 2)
        count += 1

    return F


def _build_schmid_bank(cfg):
    """
    Build 13 Schmid rotation-invariant filters.

    Params
        cfg: dict, filterbank settings from config.yaml

    Returns
        3D numpy array (support, support, 13)
    """
    params = cfg.get("schmid_params", [
        [2,1],[4,1],[4,2],[6,1],[6,2],[6,3],
        [8,1],[8,2],[8,3],[10,1],[10,2],[10,3],[10,4]])
    sup = cfg.get("lm_support", 11)
    hsup = (sup - 1) // 2

    F = np.zeros((sup, sup, len(params)))
    x, y = np.meshgrid(np.arange(-hsup, hsup + 1),
                        np.arange(-hsup, hsup + 1))
    r = np.sqrt(x**2 + y**2)

    for i, (sigma, tau) in enumerate(params):
        fm = np.cos(r * math.pi * tau / sigma) * np.exp(-r**2 / (2 * sigma**2))
        fm -= fm.mean()
        fm /= np.abs(fm).sum() + 1e-10
        F[:, :, i] = fm
    return F


def _extract_features(img, cfg, gabor_filters, lm_bank, schmid_bank):
    """
    Apply all 91 filters to the image.

    Params
        img:           2D numpy array (H, W), float64 high-pass filtered image
        cfg:           dict, filterbank settings
        gabor_filters: list of 2D arrays, 24 Gabor kernels
        lm_bank:       3D array (K, K, 48), Leung-Malik kernels
        schmid_bank:   3D array (K, K, 13), Schmid kernels

    Returns
        list of 91 2D numpy arrays, one per filter response
    """
    maps = []

    # Step 1: Gabor responses with ReLU (24 maps)
    for k in gabor_filters:
        resp = cv2.filter2D(img, cv2.CV_64F, k)
        maps.append(np.maximum(resp, 0))

    # Step 2: LoG at multiple scales (3 maps)
    for sigma in cfg.get("log_sigmas", [2, 5, 10]):
        blurred = cv2.GaussianBlur(img, (0, 0), sigma)
        maps.append(np.abs(cv2.Laplacian(blurred, cv2.CV_64F)))

    # Step 3: gradient magnitude (1 map)
    gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
    gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
    maps.append(np.sqrt(gx**2 + gy**2))

    # Step 4: local texture variance (2 maps)
    for ks in cfg.get("texture_kernel_sizes", [7, 15]):
        mu = uniform_filter(img, size=ks)
        mu2 = uniform_filter(img**2, size=ks)
        maps.append(np.sqrt(np.maximum(mu2 - mu**2, 0)))

    # Step 5: Leung-Malik responses (48 maps)
    for i in range(lm_bank.shape[2]):
        resp = cv2.filter2D(img, cv2.CV_64F, lm_bank[:, :, i].astype(np.float64))
        maps.append(np.abs(resp))

    # Step 6: Schmid responses (13 maps)
    for i in range(schmid_bank.shape[2]):
        resp = cv2.filter2D(img, cv2.CV_64F, schmid_bank[:, :, i].astype(np.float64))
        maps.append(np.abs(resp))

    return maps


def detect(gray, cfg):
    """
    Full filter bank pipeline: high-pass, 91 filters, pool to 273,
    FFT autocorrelation, period extraction.

    Params
        gray: 2D numpy array (H, W), uint8 grayscale image
        cfg:  dict, filterbank settings from config.yaml

    Returns
        tuple (h_tiles, v_tiles, confidence)
            h_tiles:    int, number of horizontal repetitions
            v_tiles:    int, number of vertical repetitions
            confidence: float, fixed at 0.90
    """
    global _gabor_cache, _lm_cache, _schmid_cache

    # Step 1: build filter banks on first call (cached after that)
    if _gabor_cache is None:
        _gabor_cache = _build_gabor_filters(cfg)
        _lm_cache = _build_lm_bank(cfg)
        _schmid_cache = _build_schmid_bank(cfg)

    # Step 2: high-pass filter to remove illumination gradients
    hp = high_pass_filter(gray).astype(np.float64)

    # Step 3: apply all 91 filters
    feature_maps = _extract_features(hp, cfg, _gabor_cache, _lm_cache, _schmid_cache)

    # Step 4: multi-scale pooling (x1, x2, x4) gives 273 maps
    scales = tuple(cfg.get("pool_scales", [1, 2, 4]))
    pooled = multiscale_pool(feature_maps, scales)

    # Step 5: average FFT autocorrelation over all 273 maps
    ac = fft_autocorrelation(pooled, tukey_alpha=cfg.get("tukey_alpha", 0.1))

    # Step 6: extract horizontal and vertical periods from autocorrelation
    hr = cfg.get("peak_height_ratio", 0.15)
    pr = cfg.get("peak_prominence_ratio", 0.05)
    H, W = gray.shape

    h_period = find_period_from_profile(ac[0, :], W, height_ratio=hr,
                                         prominence_ratio=pr)
    v_period = find_period_from_profile(ac[:, 0], H, height_ratio=hr,
                                         prominence_ratio=pr)

    h_tiles = max(1, round(W / h_period))
    v_tiles = max(1, round(H / v_period))
    return h_tiles, v_tiles, 0.90
