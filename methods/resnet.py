"""
ResNet feature autocorrelation: hooks into early layers of a pretrained
ResNet18, upsamples the feature maps, then runs FFT autocorrelation.
"""

import sys, os
import numpy as np
import cv2
import torch
import torchvision.models as models
from utils import fft_autocorrelation, find_period_from_profile, multiscale_pool


_model = None
_device = None
_hooks_output = {}


def _load_model(cfg):
    """
    Load pretrained ResNet18 and attach forward hooks to capture
    intermediate feature maps. Only runs once, cached after that.

    Params
        cfg: dict, resnet settings from config.yaml
    """
    global _model, _device
    if _model is not None:
        return

    # Step 1: pick device and load pretrained weights
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _model = models.resnet18(
        weights=models.ResNet18_Weights.DEFAULT).to(_device)
    _model.eval()

    # Step 2: register hooks on target layers
    def make_hook(name):
        def hook(mod, inp, out):
            _hooks_output[name] = out.detach().cpu().numpy()
        return hook

    layers = cfg.get("layers", ["relu", "layer1", "layer2"])
    named = dict(_model.named_modules())
    for name in layers:
        named[name].register_forward_hook(make_hook(name))


def detect(gray, cfg, img_bgr=None):
    """
    Extract ResNet features, upsample, pool, and run FFT autocorrelation
    to find the tile period.

    Params
        gray:    2D numpy array (H, W), uint8 grayscale image
        cfg:     dict, resnet settings from config.yaml
        img_bgr: 3D numpy array (H, W, 3), uint8 BGR colour image (optional,
                 used instead of gray when available for better features)

    Returns
        tuple (h_tiles, v_tiles, confidence)
            h_tiles:    int, number of horizontal repetitions
            v_tiles:    int, number of vertical repetitions
            confidence: float, fixed at 0.85
    """
    _load_model(cfg)

    # Step 1: prepare RGB input
    if img_bgr is not None:
        H, W = img_bgr.shape[:2]
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    else:
        H, W = gray.shape
        rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    # Step 2: apply ImageNet normalisation
    img_f = rgb.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_norm = (img_f - mean) / std

    # Step 3: forward pass through conv1 to layer2
    tensor = torch.from_numpy(img_norm.transpose(2, 0, 1)[None]).to(_device)
    _hooks_output.clear()
    with torch.no_grad():
        _model(tensor)

    # Step 4: collect feature maps and upsample to full resolution
    fmaps = []
    for name, feat in _hooks_output.items():
        for ch in range(feat.shape[1]):
            fm = feat[0, ch]
            if fm.shape[0] != H or fm.shape[1] != W:
                fm = cv2.resize(fm, (W, H), interpolation=cv2.INTER_LINEAR)
            fmaps.append(fm.astype(np.float64))

    # Step 5: multi-scale pooling (x1, x2, x4) gives 768 maps
    scales = tuple(cfg.get("pool_scales", [1, 2, 4]))
    pooled = multiscale_pool(fmaps, scales)

    # Step 6: average FFT autocorrelation and extract periods
    ac = fft_autocorrelation(pooled, tukey_alpha=0.1)

    h_period = find_period_from_profile(ac[0, :], W)
    v_period = find_period_from_profile(ac[:, 0], H)

    h_tiles = max(1, round(W / h_period))
    v_tiles = max(1, round(H / v_period))
     ## I have added default confidence 0.9, this should be selected automatically  for better performance
    return h_tiles, v_tiles, 0.9
