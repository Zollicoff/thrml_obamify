# thrml-Obamify

**thrml-Obamify** is an experimental image transformation that uses [Extropic's `thrml` library](https://github.com/extropic-ai/thrml) to recreate any input image in the style of Barack Obama. Instead of neural nets or classical filters, it uses thermodynamic sampling via probabilistic graphical models to "cool" an image toward an Obama-like equilibrium.

**✅ This implementation properly uses thrml's API** including:
- `CategoricalNode` for multi-state pixel variables
- `CategoricalEBMFactor` for energy-based model definition
- `sample_states()` for blocked Gibbs sampling
- `FactorSamplingProgram` for coordinating the sampling process

---

## How it works

### Probabilistic Graphical Model

Each pixel is represented as a `CategoricalNode` that can take one of K discrete states (palette colors extracted from the Obama reference image). The pixels form a 2D grid graph with 4-connectivity (neighbors: up, down, left, right).

### Energy Function

The system minimizes an energy function with three terms:

* **Content** (unary): Penalizes pixel colors that differ from the source image
* **Style** (unary): Rewards using colors frequent in the Obama reference
* **Smoothness** (pairwise): Potts model encouraging neighboring pixels to match

Energy is defined using `thrml.models.discrete_ebm.CategoricalEBMFactor` with:
- Unary biases for content + style terms
- Pairwise weights for smoothness (Potts coupling)

### Sampling

Blocked Gibbs sampling via `thrml.sample_states()` with:
- Checkerboard blocking for parallel updates
- Simulated annealing (β: 0.5 → 8.0)
- Progressive cooling to find low-energy configurations

---

## Quick start

### 1) Create an environment

macOS (Metal / Apple Silicon):

```bash
conda env create -f environment-macos-metal.yml
conda activate thrml-obamify
```

Windows (CPU for development only):

```bash
conda env create -f environment-windows-cpu.yml
conda activate thrml-obamify
```

### 2) Run the CLI

```bash
obamify \
  --source assets/image1.png \
  --target assets/obama.jpg \
  --steps 500 \
  --palette_size 24 \
  --save_gif
```

### 3) Outputs

```
output/
  obamify_YYYYMMDD-HHMMSS.png   # final image (timestamped)
  obamify_YYYYMMDD-HHMMSS.gif   # optional animation of the cooling process
```

---

## Parameters

| Flag              | Description                                  | Default |
| ----------------- | -------------------------------------------- | ------- |
| `--steps`         | Number of sampling steps                     | 500     |
| `--beta_start`    | Starting inverse temperature                 | 0.5     |
| `--beta_end`      | Final inverse temperature                    | 8.0     |
| `--palette_size`  | Number of colors taken from the target image | 24      |
| `--alpha`         | Weight on content vs style (0..1)            | 0.5     |
| `--lambda_smooth` | Smoothness weight for neighbors              | 0.8     |

---

## Notes

* Start small (e.g., 128×128) to get quick feedback.
* Try grayscale first, then color palettes of 16–32 colors.
* Save intermediate frames to see the cooling process.

---

## Acknowledgments

* [`thrml`](https://github.com/extropic-ai/thrml) for discrete sampling utilities
* [Original Obamify](https://github.com/Spu7Nix/obamify) for the meme inspiration

---

## License

MIT. Credit `thrml` and the original `Obamify` projects.
