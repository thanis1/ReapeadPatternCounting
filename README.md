# Repeated Pattern Detector

Detects how many times a 2D pattern repeats horizontally and vertically in an image.

```
python detect.py testimage.png
4 3
```

## Methods

Three methods working in different domains, combined via ensemble voting.

### Method 1: ORB Displacement Voting (orb.py)

Finds ORB [1] keypoints and self-matches them. Since the pattern repeats, the same keypoint shows up at regular intervals, so displacements between matches cluster at multiples of the tile size. Histogram voting + GCD extracts the period. Very fast (~15 ms), struggles with blur.

### Method 2: Combined Filter Bank (filterbank.py)

Applies 91 filters (24 Gabor + 6 second-order + 48 Leung-Malik [2] + 13 Schmid [3]), pools each at 3 scales to get 273 feature maps, then runs FFT autocorrelation on all of them. Averaging that many autocorrelations kills the noise and the periodic peaks stand out clearly. Most reliable method, no PyTorch needed.

### Method 3: ResNet Feature Autocorrelation (resnet.py)

Hooks into early layers of a pretrained ResNet18 [4] to grab 256 feature channels, upsamples them, pools to 768 maps, then runs the same FFT autocorrelation pipeline. This approach is inspired by Lettry et al. [5] who showed that CNN activations encode spatial repetitions, and by Qu et al. [6] who combined CNN features with autocorrelation for repeated pattern detection. Needs PyTorch and may struggle on images far from ImageNet.

### Ensemble (ensemble.py)

Each method votes (h, v) weighted by base_weight * confidence. Highest score per axis wins. Default weights: ORB 0.15, FilterBank 0.50, ResNet 0.35.

### Utilities (utils.py)

Shared functions: high-pass filtering, FFT autocorrelation, peak detection + GCD, histogram voting, multi-scale pooling.

## Setup

### Conda

```bash
conda env create -f environment.yml
conda activate pattern-detector
```

### pip

```bash
python -m venv .venv
source .venv/bin/activate        # Linux / Mac
# .venv\Scripts\activate         # Windows
pip install -r requirements.txt
```

PyTorch is only needed for Method 3. Without it the ensemble falls back to ORB + FilterBank.

## Usage

```
python detect.py <image> [-m METHOD] [-c CONFIG] [-o OUTPUT]
```

`image` -- input image (PNG, JPG, BMP)
`-m` -- `ensemble` (default), `orb`, `filterbank`, or `resnet`
`-c` -- YAML config file (default: `config.yaml`)
`-o` -- save results to JSON

```bash
python detect.py testimage.png
python detect.py testimage.png -m orb
python detect.py testimage.png -m filterbank
python detect.py testimage.png -o result.json
```

Output: two integers to stdout, `horizontal vertical`.

## Project Structure

```
ReapeadPatternCounting/
    detect.py              entry point
    config.yaml            parameters
    utils.py               shared utilities
    requirements.txt
    environment.yml
    methods/
        __init__.py
        orb.py             Method 1
        filterbank.py      Method 2
        resnet.py          Method 3
        ensemble.py        voting
```

## Requirements

- Python >= 3.11
- NumPy >= 1.24, OpenCV >= 4.8, SciPy >= 1.11, PyYAML >= 6.0
- PyTorch >= 2.0 + TorchVision >= 0.15 (ResNet only)

## References

[1] E. Rublee, V. Rabaud, K. Konolige, and G. Bradski, "ORB: An efficient alternative to SIFT or SURF," in *Proc. IEEE International Conference on Computer Vision (ICCV)*, 2011, pp. 2564-2571.

[2] T. Leung and J. Malik, "Representing and recognizing the visual appearance of materials using three-dimensional textons," *International Journal of Computer Vision*, vol. 43, no. 1, pp. 29-44, 2001.

[3] C. Schmid, "Constructing models for content-based image retrieval," in *Proc. IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, vol. 2, pp. 39-45, 2001.

[4] K. He, X. Zhang, S. Ren, and J. Sun, "Deep Residual Learning for Image Recognition," in *Proc. IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 2016, pp. 770-778.

[5] L. Lettry, M. Perdoch, K. Vanhoey, and L. Van Gool, "Repeated Pattern Detection Using CNN Activations," in *Proc. IEEE Winter Conference on Applications of Computer Vision (WACV)*, 2017, pp. 47-55.

[6] H. Qu, Y. Zhou, K. P. Lam, and G. Brunnett, "Efficient and Effective Detection of Repeated Pattern from Fronto-Parallel Images with Unknown Visual Contents," *Eng. Proc.*, vol. 6, no. 1, 2025.
