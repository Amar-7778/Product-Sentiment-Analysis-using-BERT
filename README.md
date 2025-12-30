# Product Sentiment Analysis using BERT

A lightweight, easy-to-follow project for performing sentiment analysis on product reviews using a BERT-based model (Hugging Face Transformers). This repository contains code, examples, and configuration to train, evaluate, and run inference with a BERT model fine-tuned for binary/multi-class sentiment classification.

## Table of contents

- [Features](#features)
- [Repository structure](#repository-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Preparing the data](#preparing-the-data)
- [Training](#training)
- [Evaluation](#evaluation)
- [Inference](#inference)
- [Tips and notes](#tips-and-notes)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Features

- Fine-tune a BERT model (e.g. `bert-base-uncased`) for sentiment classification
- Support for training, evaluation, and inference scripts
- Configurable hyperparameters and reproducible training setup
- Example commands for CPU/GPU usage

## Repository structure

(If some files below are not present in this repo, adapt or create them.)

- `data/` - raw and processed datasets (CSV / JSONL with `text` and `label` columns)
- `src/` or `scripts/` - training, evaluation, and inference scripts
  - `train.py` - script to fine-tune BERT
  - `evaluate.py` - evaluate saved model on a test set
  - `predict.py` - run single-sentence or batch inference
- `models/` - saved checkpoints
- `requirements.txt` - Python dependencies
- `README.md` - this file

## Requirements

- Python 3.8+
- pip
- Recommended: CUDA-enabled GPU for training

Python packages typically used:

```bash
transformers>=4.0.0
datasets
torch>=1.7
scikit-learn
pandas
numpy
python-dotenv  # optional, for config
```

Install requirements:

```bash
python -m pip install -r requirements.txt
```

If you don't have a `requirements.txt`, you can install the minimal set:

```bash
pip install transformers torch pandas scikit-learn datasets
```

## Preparing the data

Prepare your dataset in CSV (or JSON) format with at least two columns:

- `text` — the product review text
- `label` — the sentiment label (e.g. `0`/`1` for negative/positive, or `0`/`1`/`2` for 3-class)

Example CSV header:

```csv
text,label
"This product is fantastic!",1
"Terrible build quality.",0
```

Split your data into `train`, `validation`, and `test` sets. Place them under `data/` as `train.csv`, `valid.csv`, `test.csv`.

## Training

A typical training command (adapt parameter names to the training script in this repo):

```bash
python src/train.py \
  --train_file data/train.csv \
  --validation_file data/valid.csv \
  --model_name_or_path bert-base-uncased \
  --output_dir models/finetuned-bert \
  --per_device_train_batch_size 16 \
  --per_device_eval_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --max_seq_length 128 \
  --save_steps 500 \
  --evaluation_strategy steps \
  --logging_steps 100
```

Notes:
- Use `CUDA_VISIBLE_DEVICES=0` or set device in your script to run on GPU
- Lower batch size if you run out of memory
- Adjust `max_seq_length` depending on review length

## Evaluation

After training, evaluate the model on the held-out test set:

```bash
python src/evaluate.py \
  --model_dir models/finetuned-bert \
  --test_file data/test.csv \
  --batch_size 64
```

Expected metrics: accuracy, precision, recall, F1 score (per-class and macro), and a confusion matrix.

## Inference

Single-sentence prediction example:

```bash
python src/predict.py --model_dir models/finetuned-bert --text "This product exceeded my expectations"
```

Batch prediction (file input):

```bash
python src/predict.py --model_dir models/finetuned-bert --input_file data/unlabeled_reviews.csv --output_file predictions.csv
```

The `predict.py` should load the tokenizer + model from `model_dir` and output probabilities and predicted labels.

## Tips and notes

- If you want to experiment with different pretrained models, try `distilbert-base-uncased` (faster) or domain-adapted BERT models.
- Use mixed precision training (`fp16`) if available to speed up training and reduce memory usage.
- For imbalanced datasets, consider weighted loss or oversampling the minority class.
- Save best checkpoints (by validation F1) rather than only saving every N steps.

## Results

Add your training and evaluation results here once you run experiments.

Example placeholder:

- Dataset: ExampleProductReviews (X, Y)
- Best validation F1: 0.92
- Test accuracy: 0.90

## Contributing

Contributions are welcome — please open issues or pull requests for bug fixes, new features, or improvements. Consider adding:

- More preprocessing scripts
- Support for additional datasets and formats
- Additional evaluation metrics and visualizations

## License

This project is provided under the MIT License. See LICENSE for details.

## Contact

Created by Amar-7778. For questions or help, open an issue in this repository.
