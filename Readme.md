# halo-legibility

Code for reproducing the results from:

> **Ranking handwriting images like a human: AI-based legibility assessment by comparative ranking on the HaLO dataset**
> Meike Bauer, Marco Lents, Erik Schmidt, Tim Hamann, Lukas Pieger, Susanne Salata, Francesco Di Salvo, Tal Hoffmann, Jens Barth, Christian Ledig

This repository contains the `halo_legibility` Python package and analysis scripts used in the paper. The package provides model definitions, a Bayesian Bradley-Terry ranking model, data loading utilities, and a training loop. The scripts reproduce all tables and analyses from the paper.

Dataset: [MarcoLents/HaLO on HuggingFace](https://huggingface.co/datasets/MarcoLents/HaLO)

## Installation

```bash
pip install .
```

Feature files are downloaded automatically from HuggingFace when running the scripts.

## Model architectures

All models are regression heads on frozen 1024-dimensional Pixtral-ViT embeddings. They predict a scalar legibility score per sample and compare pairs via `sigmoid(score1 - score2)`.

| Model | Architecture | Description |
|-------|-------------|-------------|
| `m0` | 1024 → 1 | Linear probe (dot product) |
| `m1` | 1024 → 256 → 1 | One hidden layer |
| `m2` | 1024 → 512 → 256 → 1 | Two hidden layers |
| `m3` | 1024 → 512 → 256 → 256 → 1 | Three hidden layers |

## Train models

```bash
python train_model.py --model m0
python train_model.py --model m1
python train_model.py --model m2
python train_model.py --model m3
```

Training progress can be tracked with TensorBoard (`runs/` directory). Run `python train_model.py -h` for all options (batch size, learning rate, dropout, etc.).

## Reproduce results

All analysis scripts support `--no-ml` to skip sections that require trained models.

**Ranking analysis** (Tables 1-2, 4): Bootstrap analysis of the Bayesian Bradley-Terry model, inter-annotator ranking agreement, and ML model evaluation on the test set.

```bash
python ranking_analysis.py --resamples 1000
```

**Absolute annotation analysis** (Tables 3, 5): Krippendorff's alpha for inter-rater agreement on Likert-scale annotations, and agreement between the linear probe and human annotators.

```bash
python absolute_analysis.py
```

**McNemar's test** (Section "Regression heads"): Pairwise significance tests between model architectures.

```bash
python mcnemar_test.py
```

**Concept vectors** (Section "Pixtral-ViT Concept Vectors"): Logistic regression probes for binary characteristics and Ridge regression for aspect ratio.

```bash
python find_concepts.py
```

**Concept comparison** (Section "Pixtral-ViT Concept Vectors"): Cosine similarities between concept vectors and the legibility direction, Pearson correlations between concept projections and model scores, and direction removal analysis.

```bash
python concept_comparison.py
```

**Figures** (Figures 2-4): Generate paper figures from analysis results.

```bash
python generate_plots.py
```

Results are written to the `results/` directory. Figures are written to the `figures/` directory.

## Package contents

The `halo_legibility` package provides:

- `halo_legibility.model` — `ParallelFeatureNetwork`, `build_model`, `load_model`
- `halo_legibility.rank` — `BTModel` (Bayesian Bradley-Terry via PyMC), `fit_model`, `eval_model`
- `halo_legibility.loader` — `HaLOPixtralFeatures` dataset, `score_images`
- `halo_legibility.train` — Training loop with early stopping and TensorBoard logging
- `halo_legibility.utils` — `parse_model_name`, `discover_models`

## Citation

```bibtex
@article{bauer2025halo,
  title={Ranking handwriting images like a human: AI-based legibility assessment by comparative ranking on the HaLO dataset},
  author={Bauer, Meike and Lents, Marco and Schmidt, Erik and Hamann, Tim and Pieger, Lukas and Salata, Susanne and Di Salvo, Francesco and Hoffmann, Tal and Barth, Jens and Ledig, Christian},
  year={2026}
}
```

## License

[MIT](LICENSE)
