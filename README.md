# Repeated Pattern Detector

Detects how many times a 2D pattern repeats horizontally and vertically in an image.

```
python detect.py testimage.png
4 3
```

## Methods

Three methods, each working in a different domain, combined through weighted ensemble voting.

### Method 1: ORB Displacement Voting (orb.py)

Spatial domain approach. Detects ORB keypoints (FAST corners + BRIEF descriptors) and self-matches them using brute-force Hamming distance. In a tiled image the same feature appears at regular intervals, so displacement vectors between matched pairs cluster at multiples of the tile size. Histogram voting followed by GCD refinement extracts the fundamental period.

Fast (~15 ms) but sensitive to noise and blur.

Reference: Rublee, E., Rabaud, V., Konolige, K. & Bradski, G. "ORB: An efficient alternative to SIFT or SURF." ICCV, 2011, pp. 2564-2571.

### Method 2: Combined Filter Bank (filterbank.py)

Frequency domain approach. Applies 91 handcrafted filters from four families:

- 24 Gabor filters (3 scales x 8 orientations), oriented edge and texture detectors
- 6 second-order features: Laplacian-of-Gaussian, gradient magnitude, local variance
- 48 Leung-Malik filters: bars, edges, Gaussians, LoG at multiple orientations and scales
- 13 Schmid filters: rotation-invariant cosine-Gaussian ring patterns

Each response is multi-scale pooled (x1, x2, x4), producing 273 feature maps. FFT-based autocorrelation (Wiener-Khinchin theorem) on each map reveals periodic peaks at the tile spacing. Averaging 273 autocorrelations suppresses noise while reinforcing the true period. Peak detection followed by GCD gives the fundamental period.

Most robust handcrafted approach. Does not require PyTorch.

References:
- Leung, T. & Malik, J. "Representing and recognizing the visual appearance of materials using three-dimensional textons." International Journal of Computer Vision, 43(1):29-44, 2001.
- Schmid, C. "Constructing models for content-based image retrieval." CVPR, vol. 2, pp. 39-45, 2001.
- Varma, M. & Zisserman, A. "A statistical approach to texture classification from single images." International Journal of Computer Vision, 62(1-2):61-81, 2005.

### Method 3: ResNet Feature Autocorrelation (resnet.py)

Learned CNN features approach. Uses early layers of a pretrained ResNet18 (ImageNet weights) as a feature extractor. Forward hooks on relu, layer1, and layer2 capture 256 feature channels encoding edges, textures, and shapes. These are upsampled to full resolution and multi-scale pooled to 768 maps, then fed through the same FFT autocorrelation pipeline as Method 2.

Richer features than any handcrafted bank, but possible domain gap on images far from ImageNet.

Reference: He, K., Zhang, X., Ren, S. & Sun, J. "Deep Residual Learning for Image Recognition." CVPR, 2016, pp. 770-778.

### Ensemble Voting (ensemble.py)

Each method outputs (h, v, confidence). The ensemble multiplies each method's base weight by its confidence and votes per axis. Highest total weight wins.

Default weights: ORB = 0.15, FilterBank = 0.50, ResNet = 0.35.

### Shared Utilities (utils.py)

Common functions used by the filter bank and ResNet methods:
- high_pass_filter: removes illumination gradients via Gaussian subtraction
- fft_autocorrelation: Wiener-Khinchin 2D autocorrelation with Tukey windowing
- find_period_from_profile: peak detection + GCD on 1D autocorrelation slices
- histogram_vote: histogram-based period extraction from displacement vectors
- multiscale_pool: downsample/upsample pooling at multiple scales

## Setup

### Option A: Conda

```bash
conda env create -f environment.yml
conda activate pattern-detector
```

### Option B: pip

```bash
python -m venv .venv
source .venv/bin/activate        # Linux / Mac
# .venv\Scripts\activate         # Windows
pip install -r requirements.txt
```

PyTorch is only needed for Method 3. Without it you can still run ORB and FilterBank individually.

## Usage

```
python detect.py <image> [-m METHOD] [-c CONFIG] [-o OUTPUT]
```

| Argument       | Description                                             |
|----------------|---------------------------------------------------------|
| `image`        | Path to input image (PNG, JPG, BMP, grayscale or RGB)   |
| `-m, --method` | `ensemble` (default), `orb`, `filterbank`, or `resnet`  |
| `-c, --config` | YAML config file (default: `config.yaml`)               |
| `-o, --output` | Save results to a JSON file                             |

### Examples

```bash
# ensemble (all three methods)
python detect.py testimage.png

# single method
python detect.py testimage.png -m orb
python detect.py testimage.png -m filterbank
python detect.py testimage.png -m resnet

# save detailed results to json
python detect.py testimage.png -o result.json
```

### Output

Two integers printed to stdout:

```
horizontal_repetitions vertical_repetitions
```

## Project Structure

```
pattern_detector/
    detect.py              main entry point
    config.yaml            tunable parameters
    utils.py               shared utilities (FFT, peak detection, pooling)
    requirements.txt       pip dependencies
    environment.yml        conda environment
    methods/
        __init__.py
        orb.py             Method 1: ORB displacement voting
        filterbank.py      Method 2: combined filter bank
        resnet.py          Method 3: ResNet feature autocorrelation
        ensemble.py        weighted confidence voting
```

## Requirements

- Python >= 3.11
- NumPy >= 1.24
- OpenCV >= 4.8
- SciPy >= 1.11
- PyYAML >= 6.0
- PyTorch >= 2.0 and TorchVision >= 0.15 (for ResNet method only)
