# DeFOMO

A personal project exploring fake news detection with text, image and graph models.
It includes text-only, image-only and multimodal (text + image + graph) detectors,
trained and compared on several public datasets.

This repository contains the detector code only. Datasets, model weights, logs and
other outputs are not included (see `.gitignore`).

## Datasets

- Fakeddit
- FakeNewsNet (PolitiFact / GossipCop)
- CHECKED
- WELFake
- PolitiFact (UPFD graphs)

Download them separately and point the scripts at your local copies.

## Layout

| Path | Contents |
| --- | --- |
| `disertation/code/benchmarks/` | Matched comparison of baselines (DistilBERT, BERT, RoBERTa, ResNet50, BERT+ResNet concat, CLIP linear, SpotFake, CAFE) and DeFOMO across datasets and seeds. `scripts/run_matrix.py` is the entry point. |
| `disertation/code/checked/` | Multimodal training/testing on the CHECKED dataset, including the GNN module and split creation. |
| `disertation/code/fakeddit/` | Multimodal training/testing for Fakeddit. |
| `disertation/code/fakenewsnet/` | Training/testing for FakeNewsNet. |
| `fyp_code/` | BERT and hybrid (BERT + graph) classifier: `train.py`, `train_cv.py`, `evaluate.py`, `predict.py`, plus `models/`, `data/` and `utils/`. |
| `mine/` | Earlier experiments: DistilBERT training, multimodal (ViT) models, Fakeddit and MediaRank code. |
| `politifact_trainer/` | `train_politifact.py`: PolitiFact trainer for text, multimodal and propagation settings. |
| `faknow/` | PolitiFact training script built on FaKnow. |
| `bert/`, `evaluate_on_welfake.py` | BERT test script and WELFake evaluation. |

## Setup

Python 3 with PyTorch, Hugging Face `transformers`, scikit-learn, pandas, numpy,
Pillow and tqdm. Some scripts also need `torch_geometric` (graph models) and
TensorBoard. There is no pinned requirements file yet.

## Example usage

```bash
# Text / multimodal / propagation trainer on PolitiFact
python politifact_trainer/train_politifact.py --modality text \
    --train-metadata data/politifact/train.json \
    --val-metadata data/politifact/val.json \
    --test-metadata data/politifact/test.json

# Benchmark matrix (see --help for models, datasets and seeds)
python disertation/code/benchmarks/scripts/run_matrix.py --help
```

Scripts expect local data and API tokens (e.g. Hugging Face) to be supplied via
your own environment; none are stored in this repository.
