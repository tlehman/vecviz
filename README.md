#  VecViz: Dimensionality Reduction for LLM embeddings

This repository visualizes the embedding space for the llama3.2 model.

Mathematically, $A^*$ is the set of all finite-length strings. Then $\mathbb{R}^d$ is the set of all $d$-dimensional real vectors.

$$A^* \to \mathbb{R}^d$$

## Visualize any prompt as a point in 3-dimensional space

Let $p \in A^*$ be a prompt, then when you pass that prompt into the [ollama](https://github.com/ollama/ollama) embedding API, it produces a vector of dimension $d = 3072$.

To visualize these high-dimensional embeddings, we use dimensionality reduction techniques to project from $\mathbb{R}^{3072}$ down to $\mathbb{R}^3$.

## Demo

[![VecViz Demo](https://img.youtube.com/vi/8_ow2s_KkL0/0.jpg)](https://youtu.be/8_ow2s_KkL0?si=tauIQpGtnwZukTwe)

### Dimensionality Reduction Methods

Common techniques for reducing dimensionality while preserving structure:

- **PCA (Principal Component Analysis)**: Linear projection that maximizes variance
- **t-SNE**: Non-linear method that preserves local neighborhood structure
- **UMAP**: Balances local and global structure preservation

## Usage

1. Install and run [ollama](https://github.com/ollama/ollama)
2. Pull the llama3.2 model: `ollama pull llama3.2`
3. Generate embeddings using the API:

```bash
curl http://localhost:11434/api/embed -d '{
  "model": "llama3.2",
  "input": "Your prompt here"
}'
```

The response will contain a 3072-dimensional vector that can then be reduced to 3D for visualization.

llama3.2 has $d = 3072$
