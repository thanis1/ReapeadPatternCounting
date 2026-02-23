"""
ORB displacement voting: detects keypoints, self-matches them,
and uses histogram voting to find the tile period.
"""

import sys, os
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import histogram_vote


def detect(gray, cfg):
    """
    Run ORB-based pattern detection on a grayscale image.

    Params
        gray: 2D numpy array (H, W), uint8 grayscale image
        cfg:  dict, ORB settings from config.yaml

    Returns
        tuple (h_tiles, v_tiles, confidence)
            h_tiles:    int, number of horizontal repetitions
            v_tiles:    int, number of vertical repetitions
            confidence: float between 0 and 1
    """
    H, W = gray.shape
    n_feat = cfg.get("n_features", 3000)
    match_k = cfg.get("match_k", 6)
    hamming_thr = cfg.get("hamming_threshold", 50)
    min_disp = cfg.get("min_displacement", 0.05)
    min_votes = cfg.get("min_votes", 5)

    # Step 1: detect ORB keypoints and compute descriptors
    detector = cv2.ORB_create(nfeatures=n_feat)
    keypoints, descriptors = detector.detectAndCompute(gray, None)

    if descriptors is None or len(keypoints) < 20:
        return 1, 1, 0.0

    # Step 2: self-match every descriptor against every other
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(descriptors, descriptors, k=match_k)

    # Step 3: collect displacement vectors between matched keypoints
    h_disps, v_disps = [], []
    for group in matches:
        for m in group:
            if m.queryIdx == m.trainIdx or m.distance > hamming_thr:
                continue
            dx = abs(keypoints[m.trainIdx].pt[0] - keypoints[m.queryIdx].pt[0])
            dy = abs(keypoints[m.trainIdx].pt[1] - keypoints[m.queryIdx].pt[1])
            if dx > W * min_disp:
                h_disps.append(dx)
            if dy > H * min_disp:
                v_disps.append(dy)

    # Step 4: histogram voting to find dominant period
    h_period = histogram_vote(h_disps, W, min_votes)
    v_period = histogram_vote(v_disps, H, min_votes)

    # Step 5: convert period to tile count and compute confidence
    if h_period and v_period:
        h_tiles = max(1, round(W / h_period))
        v_tiles = max(1, round(H / v_period))
        ## I have added default confidence 0.9, this should be selected automatically  for better performance
        return h_tiles, v_tiles, 0.9

    return 1, 1, 0.0
