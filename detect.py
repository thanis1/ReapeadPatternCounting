"""
Main script: loads an image, runs detection methods, prints grid dimensions.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import yaml

from methods import orb, filterbank, ensemble

# Check Resent availability 
try:
    from methods import resnet
    _has_resnet = True
except Exception:
    _has_resnet = False


DEFAULT_WEIGHTS = {"orb": 0.25, "filterbank": 0.4, "resnet": 0.35}


def load_config(path):
    """
    Read YAML config file or fall back to built-in defaults.

    Params
        path: string, path to the YAML file

    Returns
        dict with sections: orb, filterbank, resnet, ensemble
    """
    if path and Path(path).exists():
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {
        "orb": {},
        "filterbank": {},
        "resnet": {},
        "ensemble": {"weights": DEFAULT_WEIGHTS},
    }


def run_method(name, fn, *args, **kwargs):
    """
    Execute a detection method, print its result and elapsed time.

    Params
        name:   string, display name for printing (e.g. "ORB")
        fn:     callable, the detection function to run
        *args:  positional arguments forwarded to fn
        **kwargs: keyword arguments forwarded to fn

    Returns
        tuple on success, whatever fn returns
        None on failure
    """
    print(f"\n  running {name} ...")
    t0 = time.perf_counter()

    try:
        result = fn(*args, **kwargs)
    except Exception as e:
        dt = time.perf_counter() - t0
        print(f"  {name} FAILED: {e}  ({dt:.3f}s)")
        return None

    dt = time.perf_counter() - t0

    if len(result) == 3:
        print(f"  {name} result: {result[0]} x {result[1]}  "
              f"conf={result[2]:.3f}  ({dt:.3f}s)")
    else:
        print(f"  {name} result: {result[0]} x {result[1]}  ({dt:.3f}s)")

    return result


def main():
    # Step 1: parse command-line arguments
    parser = argparse.ArgumentParser(description="Detect repeated pattern grid size.")
    parser.add_argument("image", help="input image path")
    parser.add_argument("-c", "--config", default="config.yaml")
    parser.add_argument("-m", "--method", default="ensemble",
                        choices=["orb", "filterbank", "resnet", "ensemble"])
    parser.add_argument("-o", "--output", default=None,
                        help="save results to json")
    args = parser.parse_args()

    # Step 2: load image
    img_bgr = cv2.imread(args.image)
    if img_bgr is None:
        sys.exit(f"ERROR: can't read '{args.image}'")

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape
    print(f"Image: {args.image}  ({W}x{H})")

    # Step 3: load config
    cfg = load_config(args.config)
    results = {}
    t_total = time.perf_counter()

    # Step 4: run selected methods, skipping any that fail
    if args.method in ("orb", "ensemble"):
        r = run_method("ORB", orb.detect, gray, cfg.get("orb", {}))
        if r is not None:
            results["orb"] = r

    if args.method in ("filterbank", "ensemble"):
        r = run_method("FilterBank", filterbank.detect, gray,
                       cfg.get("filterbank", {}))
        if r is not None:
            results["filterbank"] = r

    if args.method in ("resnet", "ensemble"):
        if not _has_resnet:
            print("\n  skipping ResNet (torch/torchvision not installed)")
        else:
            r = run_method("ResNet", resnet.detect, gray,
                           cfg.get("resnet", {}), img_bgr=img_bgr)
            if r is not None:
                results["resnet"] = r

    # Step 5: combine results or exit if everything failed
    if not results:
        sys.exit("ERROR: all methods failed, no results")

    if args.method == "ensemble" and len(results) > 1:
        weights = cfg.get("ensemble", {}).get("weights", DEFAULT_WEIGHTS)
        h, v = run_method("Ensemble", ensemble.vote, results, weights)
    else:
        name = list(results.keys())[0]
        h, v = results[name][0], results[name][1]

    dt_total = time.perf_counter() - t_total
    print(f"\nDone in {dt_total:.3f}s")
    print(f"{h} {v}")

    # Step 6: save to json if requested
    if args.output:
        out = {
            "image": Path(args.image).name,
            "method": args.method,
            "horizontal": h, "vertical": v,
            "time": round(dt_total, 3),
        }
        for mname, r in results.items():
            if len(r) >= 3:
                out[mname] = {"h": r[0], "v": r[1], "conf": round(r[2], 4)}
            else:
                out[mname] = {"h": r[0], "v": r[1]}

        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)
        print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()