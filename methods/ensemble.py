"""
Ensemble voting: combines predictions from all methods using
weighted confidence-scaled voting. Highest total weight per axis wins.
"""

from collections import defaultdict


def vote(results, weights):
    """
    Weighted confidence-scaled voting across methods.

    Params
        results: dict mapping method name to (h_tiles, v_tiles, confidence)
        weights: dict mapping method name to base weight (should sum to 1.0)

    Returns
        tuple (h_winner, v_winner)
    """
    h_votes = defaultdict(float)
    v_votes = defaultdict(float)

    # Step 1: accumulate weighted votes for each prediction
    for name, (h, v, conf) in results.items():
        w = weights.get(name, 0.0) * conf
        h_votes[h] += w
        v_votes[v] += w

    # Step 2: pick the prediction with highest total weight
    return (max(h_votes, key=h_votes.get),
            max(v_votes, key=v_votes.get))
