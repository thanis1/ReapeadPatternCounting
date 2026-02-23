# Repeated Pattern Detector

## Problem

Given an image containing a single repeated 2D pattern arranged in a regular grid,
determine how many times the pattern repeats horizontally and vertically.

## Approach

The solution combines three complementary detection methods into an ensemble,
each operating in a different signal domain to capture what the others might miss.

### Method 1: ORB Displacement Voting (Time/Spatial Domain)

ORB works directly on pixel coordinates. It detects keypoints (corners, blobs)
and computes a compact binary descriptor for each one. Because the pattern repeats,
the same keypoint appears at regular spatial intervals. By self-matching all
descriptors and collecting the displacement vectors between matched pairs, we get
a histogram that clusters at multiples of the tile size. A GCD refinement step
extracts the fundamental period from these clusters.

This method is very fast (~15 ms) but sensitive to noise and blur since it relies
on sharp local features that degrade under distortion.

### Method 2: Combined Filter Bank (Frequency Domain)

This method operates in the frequency domain. It applies 91 handcrafted filters
to the image:

- 24 Gabor filters (oriented edge/texture detectors at multiple scales)
- 6 second-order features (Laplacian-of-Gaussian blobs, gradient magnitude, local variance)
- 48 Leung-Malik filters (bars, edges, Gaussians, LoG at multiple orientations)
- 13 Schmid filters (rotation-invariant ring patterns)

Each filter response is multi-scale pooled (x1, x2, x4), producing 273 feature maps.
The FFT-based autocorrelation of each map reveals periodic peaks at the tile spacing.
Averaging 273 autocorrelations suppresses noise while reinforcing the true period
(SNR improves by a factor of sqrt(273) ~ 16.5x). Peak detection followed by GCD
gives the fundamental horizontal and vertical periods.

This is the most robust handcrafted approach and does not require PyTorch.

### Method 3: ResNet Feature Autocorrelation (Learned CNN Features)

Instead of handcrafted filters, this method uses the early layers of a pretrained
ResNet18 (trained on ImageNet) as a feature extractor. Hooks on three layers
(conv1, layer1, layer2) capture 256 feature channels that encode edges, textures,
and shapes the network has learned. These are upsampled to full resolution and
multi-scale pooled to 768 maps, then fed through the same FFT autocorrelation
pipeline as Method 2.

The learned features are richer and more diverse than any handcrafted bank, but
there may be a domain gap if the input image looks nothing like ImageNet data.

### Why an Ensemble?

Each method sees the image differently:

- ORB finds repeated spatial landmarks but fails on smooth or noisy patterns
- The filter bank detects periodic frequency signatures but uses fixed filter shapes
- ResNet captures learned visual concepts but depends on pretrained weights

By having all three vote on the grid dimensions (weighted by their confidence),
the ensemble is more robust than any single method. When they agree, confidence
is high. When one fails (e.g. ORB on a blurred image), the other two outvote it.

## Setup

### Option A: Conda

```bash
conda env create -f environment.yml
conda activate pattern-detector
```

### Option B: pip (no Conda)

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / Mac
source .venv/bin/activate

pip install -r requirements.txt
```

**Note:** PyTorch is only required for the ResNet method. If you skip it,
the ensemble will still run using ORB + FilterBank:

```bash
pip install numpy opencv-python-headless scipy pyyaml
```

## Usage

```bash
python detect.py <image_path> [options]
```

### Input

A single image file (PNG, JPG, BMP) containing a repeated 2D pattern.
The image can be grayscale or RGB.

### Output

Two integers printed to stdout, separated by a space:

```
horizontal_repetitions vertical_repetitions
```

### Examples

```bash
# Default: run all three methods + ensemble
python detect.py testimage.png
# Output: 4 3

# With logging to see what is happening
python detect.py testimage.png -v

# Run a single method
python detect.py testimage.png --method orb
python detect.py testimage.png --method filterbank
python detect.py testimage.png --method resnet

# Save detailed logs to file
python detect.py testimage.png -vv --log-file run.log
```

### Options

| Flag              | Description                                  |
|-------------------|----------------------------------------------|
| `--method`, `-m`  | `ensemble` (default), `orb`, `filterbank`, `resnet` |
| `--config`, `-c`  | Path to YAML config file (default: `config.yaml`)   |
| `-v`              | Show INFO logs (image size, methods, timing)         |
| `-vv`             | Show DEBUG logs (config values, pixel stats)         |
| `--log-file`      | Also write all logs to this file                     |
