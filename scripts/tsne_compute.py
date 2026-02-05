#!/usr/bin/env python3
"""
t-SNE dimensionality reduction script.
Reads embeddings from stdin as JSON, outputs 3D projections to stdout.
"""

import sys
import json
import numpy as np
from sklearn.manifold import TSNE


def main():
    # Read JSON from stdin
    data = json.load(sys.stdin)

    embeddings = data.get("embeddings", [])
    n_samples = len(embeddings)

    if n_samples == 0:
        json.dump({"projections": []}, sys.stdout)
        return

    # Extract IDs and vectors
    ids = [item["id"] for item in embeddings]
    vectors = np.array([item["vector"] for item in embeddings], dtype=np.float32)

    # Handle edge cases
    if n_samples == 1:
        # Can't do t-SNE with 1 sample, return origin
        json.dump({"projections": [{"id": ids[0], "x": 0.0, "y": 0.0, "z": 0.0}]}, sys.stdout)
        return

    if n_samples == 2:
        # With 2 samples, just place them apart on x-axis
        json.dump({
            "projections": [
                {"id": ids[0], "x": -1.0, "y": 0.0, "z": 0.0},
                {"id": ids[1], "x": 1.0, "y": 0.0, "z": 0.0},
            ]
        }, sys.stdout)
        return

    # Adjust perplexity for small datasets (must be < n_samples)
    perplexity = min(30, max(5, (n_samples - 1) // 3))

    # Run t-SNE
    tsne = TSNE(
        n_components=3,
        perplexity=perplexity,
        random_state=42,
        max_iter=1000,
        init="pca",
    )
    projections = tsne.fit_transform(vectors)

    # Normalize to [-1, 1] range for visualization
    max_abs = np.abs(projections).max()
    if max_abs > 0:
        projections = projections / max_abs

    # Build output
    results = []
    for i, proj in enumerate(projections):
        results.append({
            "id": ids[i],
            "x": float(proj[0]),
            "y": float(proj[1]),
            "z": float(proj[2]),
        })

    json.dump({"projections": results}, sys.stdout)


if __name__ == "__main__":
    main()
