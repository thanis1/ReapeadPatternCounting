"""
Shared helper functions used by all detection methods.
"""

import numpy as np
import cv2
from math import gcd
from functools import reduce
from scipy.signal import find_peaks
from scipy.signal.windows import tukey


def high_pass_filter(img_gray):
    """
    Remove slow illumination gradients while keeping tile edges.
    Subtracts a large Gaussian blur from the image: I_hp = I - blur(I) + 128

    Params
        img_gray: 2D numpy array (H, W)

    Returns
        2D numpy array (H, W)
    """
    H, W = img_gray.shape

    # Step 1: pick a blur kernel proportional to image size
    blur_size = max(H, W) // 4
    if blur_size % 2 == 0:
        blur_size += 1

    # Step 2: compute the low-frequency component
    low_freq = cv2.GaussianBlur(img_gray.astype(np.float64),
                                (blur_size, blur_size), 0)

    # Step 3: subtract and re-centre around 128
    hp = np.clip(img_gray.astype(np.float64) - low_freq + 128, 0, 255)
    return hp.astype(np.uint8)


def fft_autocorrelation(feature_maps, tukey_alpha=0.1):
    """
    Compute the average normalised 2D autocorrelation across a set of feature maps

    Params
        feature_maps: list of 2D numpy arrays, all same shape (H, W)
        tukey_alpha:  float, taper parameter for the Tukey window (0 to 1)

    Returns
        2D numpy array (H, W), averaged autocorrelation
    """
    if not feature_maps:
        raise ValueError("empty feature map list")

    H, W = feature_maps[0].shape

    # Step 1: build a 2D Tukey window to suppress edge effects
    window = np.outer(tukey(H, tukey_alpha), tukey(W, tukey_alpha))

    ac_sum = np.zeros((H, W), dtype=np.float64)
    n_valid = 0

    for fm in feature_maps:
        # Step 2: mean-subtract and apply window
        fm_win = (fm - fm.mean()) * window

        # Step 3: FFT, compute power spectrum, inverse FFT
        power = np.abs(np.fft.fft2(fm_win)) ** 2
        ac = np.fft.ifft2(power).real

        # Step 4: normalise so lag-0 equals 1, then accumulate
        if ac[0, 0] > 1e-10:
            ac /= ac[0, 0]
            ac_sum += ac
            n_valid += 1

    # Step 5: average over all valid maps
    if n_valid > 0:
        ac_sum /= n_valid
    return ac_sum


def find_period_from_profile(profile, dim, min_period=10,
                              height_ratio=0.15, prominence_ratio=0.05):
    """
    Extract the fundamental repetition period from a 1D autocorrelation
    profile by finding peaks and computing their GCD.

    Params
        profile:          1D numpy array, autocorrelation values
        dim:              int, image dimension (width or height)
        min_period:       int, reject periods smaller than this
        height_ratio:     float, minimum peak height as fraction of max
        prominence_ratio: float, minimum peak prominence as fraction of max

    Returns
        float, estimated period in pixels
    """
    # Step 1: take lags 1 to dim/2 (skip the trivial lag-0 peak)
    s = profile[1:dim // 2]
    if len(s) == 0 or s.max() <= 0:
        return float(dim)

    # Step 2: find peaks in the autocorrelation
    peaks, props = find_peaks(
        s,
        height=s.max() * height_ratio,
        distance=max(5, dim // 60),
        prominence=s.max() * prominence_ratio,
    )
    if len(peaks) == 0:
        return float(dim)

    # Step 3: sort peaks by prominence and take top 6
    order = np.argsort(props["prominences"])[::-1]
    top = (peaks[order] + 1)[:6].astype(int)
    top = top[top > 0]

    # Step 4: try GCD of the peak positions to find the fundamental
    if len(top) >= 2:
        g = reduce(gcd, top)
        if g >= min_period:
            return float(g)

    return float(top[0])


def histogram_vote(displacements, dim, min_votes=5):
    """
    Find the fundamental period from a bag of displacement values by
    building a histogram, finding peaks, and checking if they are
    integer multiples of a common base.

    Params
        displacements: list of floats, raw displacement values
        dim:           int, image dimension (width or height)
        min_votes:     int, minimum number of values required

    Returns
        float, estimated period in pixels, or None if not enough data
    """
    if len(displacements) < min_votes:
        return None

    # Step 1: filter out displacements near the image edges
    arr = np.array(displacements)
    arr = arr[(arr > dim * 0.03) & (arr < dim * 0.97)]
    if len(arr) < min_votes:
        return None

    # Step 2: build histogram with adaptive bin count
    n_bins = max(20, dim // max(5, dim // 50))
    hist, edges = np.histogram(arr, bins=n_bins,
                               range=(dim * 0.03, dim * 0.97))
    centers = (edges[:-1] + edges[1:]) / 2

    # Step 3: find peaks in the histogram
    peaks, _ = find_peaks(hist, height=max(3, hist.max() * 0.1),
                          distance=max(2, n_bins // 15))
    if len(peaks) == 0:
        return None

    # Step 4: sort by vote count, take top candidates
    peak_pos = centers[peaks][np.argsort(hist[peaks])[::-1]]
    int_pks = np.round(peak_pos[:8]).astype(int)
    int_pks = int_pks[int_pks > 0]
    bin_w = max(5, dim // 50)

    # Step 5: check if peaks are integer multiples of the smallest candidate
    if len(int_pks) >= 2:
        for candidate in np.sort(int_pks):
            if candidate < 10:
                continue
            ratios = int_pks / candidate
            if np.all(np.abs(ratios - np.round(ratios)) < bin_w / candidate):
                return float(candidate)

    return float(peak_pos[0])


def multiscale_pool(feature_maps, scales=(1, 2, 4)):
    """
    Create coarser versions of each feature map by downsampling then
    upsampling back. This captures both fine and coarse periodic structure.

    Params
        feature_maps: list of 2D numpy arrays, all same shape (H, W)
        scales:       tuple of ints, pooling factors (1 means no change)

    Returns
        list of 2D numpy arrays
    """
    H, W = feature_maps[0].shape
    pooled = []
    for fm in feature_maps:
        for s in scales:
            if s == 1:
                # Step 1: scale=1 means keep original
                pooled.append(fm)
            else:
                # Step 2: downsample then upsample back to original size
                down = cv2.resize(fm, (W // s, H // s),
                                  interpolation=cv2.INTER_AREA)
                up = cv2.resize(down, (W, H),
                                interpolation=cv2.INTER_LINEAR)
                pooled.append(up)
    return pooled
