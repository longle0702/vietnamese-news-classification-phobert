# 🇻🇳 Vietnamese News Classification with PhoBERT

A Vietnamese news article classifier fine-tuned on **PhoBERT** (`vinai/phobert-base`) that categorises articles into **10 topics** with a weighted F1-score of **91.4%** on the held-out test set. This project is an upgraded continuation of my earlier NLP classification work during my Bachelor's studies.

---

## 🧠 Overview

This project fine-tunes [PhoBERT-base](https://huggingface.co/vinai/phobert-base) — a RoBERTa-based language model pre-trained specifically on Vietnamese text — for multi-class news article classification.

**Pipeline at a glance:**

1. Raw Vietnamese news articles are cleaned and tokenised (via `pyvi`) into preprocessed category files under `clean-data/`.
2. `prepare_data.py` maps the cleaned category files into labelled Pandas DataFrames and performs a stratified train / val / test split.
3. `train.py` fine-tunes PhoBERT-base using a standard AdamW + linear warmup schedule, with early stopping and best-checkpoint saving.

---

## 🗂 Categories

The model classifies articles into the following 10 Vietnamese news topics:

| ID | Category (Vietnamese) | Translation |
|----|-----------------------|-------------|
| 0 | Chinh tri Xa hoi | Politics & Society |
| 1 | Doi song | Lifestyle |
| 2 | Khoa hoc | Science |
| 3 | Kinh doanh | Business |
| 4 | Phap luat | Law |
| 5 | Suc khoe | Health |
| 6 | The gioi | World |
| 7 | The thao | Sports |
| 8 | Van hoa | Culture |
| 9 | Vi tinh | Technology |

---

## 🏗 Project Structure

```
vietnamese-news-classification-transformer/
├── clean-data/                  # Preprocessed text files (one .txt per category)
│   ├── train/                   # Source for training data  (~50 k articles)
│   └── test/                    # Source for val + test data (~33 k articles)
├── src/
│   ├── preprocessing.py         # Vietnamese text normalisation, segmentation, TF-IDF utilities
│   ├── prepare_data.py          # Dataset loader and train/val/test splitter
│   └── train.py                 # Fine-tuning script (PhoBERT + callbacks)
├── vnct/                        # All training outputs
│   ├── best_model/              # Best checkpoint (saved by val loss)
│   ├── model.safetensors        # Final model weights
│   ├── config.json              # Model config
│   ├── vocab.txt / bpe.codes    # Tokenizer vocabulary files
│   ├── label_map.json           # Category → integer label mapping
│   ├── confusion_matrix.png     # Per-class confusion matrix on the test set
│   ├── training_history.csv     # Per-epoch metrics (loss, acc, precision, recall, F1)
│   └── training_log.txt         # Full training log
├── dictionary.txt               # Filtered vocabulary built from training data
├── stopword.txt                 # Vietnamese stopword list
├── requirements.txt
└── README.md
```

---

## 📊 Results

Training was run for **5 epochs** (early stopping triggered after epoch 5 due to no val-loss improvement for 3 consecutive epochs). Each epoch took approximately **21.6 minutes** on a CUDA GPU.

### 🧪 Training History

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc | Val F1 |
|-------|-----------|-----------|----------|---------|--------|
| 1 | 0.8319 | 79.46% | 0.3478 | 89.88% | 89.78% |
| 2 | 0.2518 | 92.41% | 0.2895 | 91.26% | 91.26% |
| 3 | 0.1728 | 94.86% | 0.3012 | 91.30% | 91.36% |
| 4 | 0.1269 | 96.31% | 0.3381 | 91.98% | 92.02% |
| 5 | 0.0969 | 97.46% | 0.3912 | 92.00% | 91.99% |

### 🧾 Test Set Performance (Best Checkpoint — Epoch 2)

| Metric | Score |
|--------|-------|
| Loss | 0.2868 |
| Accuracy | **91.41%** |
| Weighted Precision | **91.56%** |
| Weighted Recall | **91.41%** |
| Weighted F1 | **91.43%** |

> The best model checkpoint was saved at **epoch 2** (lowest val loss: 0.2895). The confusion matrix is saved at `vnct/confusion_matrix.png`.

---

## ⚙️ Requirements

- Python 3.10+
- CUDA-capable GPU (recommended; CPU inference is possible but slow)

Install all dependencies:

```bash
pip install -r requirements.txt
```

Key dependencies:

| Package | Version |
|---------|---------|
| torch | ≥ 2.0.0 |
| transformers | ≥ 4.40.0 |
| accelerate | ≥ 0.29.0 |
| sentencepiece | ≥ 0.1.99 |
| scikit-learn | 1.8.0 |
| pyvi | 0.1.1 |
| pandas | 3.0.1 |
| matplotlib | ≥ 3.7.0 |

---

## 🚀 Setup

```bash
git clone https://github.com/<your-username>/vietnamese-news-classification-transformer.git
cd vietnamese-news-classification-transformer

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

---

## 🛠 Usage

### 1. Data Preprocessing

Raw articles (one file per article, encoded UTF-16) should be placed inside category-named subdirectories under `Train_Full/` and `Test_Full/`. Running the preprocessing pipeline will:

- Normalise Unicode and Vietnamese tone marks
- Segment words via `pyvi`
- Remove stopwords
- Write one cleaned `.txt` per category under `clean-data/train/` and `clean-data/test/`

```bash
python src/preprocessing.py
```

### 2. Training

```bash
python src/train.py
```

This will:

1. Load and label data from `clean-data/test/` (training set, ~50 k articles) and split `clean-data/train/` 50/50 into validation and test sets (~16.9 k each).
2. Tokenise all splits using the PhoBERT tokenizer (downloaded automatically from the Hugging Face Hub on first run).
3. Fine-tune `vinai/phobert-base` with AdamW, a linear warmup scheduler, gradient clipping, and early stopping.
4. Save the best checkpoint to `vnct/best_model/`, the final model to `vnct/`, and the training history to `vnct/training_history.csv`.
5. Evaluate the best checkpoint on the test set and save the confusion matrix to `vnct/confusion_matrix.png`.

### 3. Training Arguments

All hyperparameters can be overridden via command-line flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--epochs` | `10` | Maximum number of training epochs |
| `--batch_size` | `32` | Training batch size |
| `--eval_batch_size` | `64` | Evaluation batch size |
| `--max_len` | `256` | Maximum token sequence length |
| `--lr` | `2e-5` | AdamW learning rate |
| `--warmup_ratio` | `0.1` | Fraction of total steps used for linear warmup |
| `--patience` | `3` | Early stopping patience (epochs without val-loss improvement) |
| `--val_ratio` | `0.10` | *(reserved)* Validation split ratio |

**Example — shorter run with a smaller batch:**

```bash
python src/train.py --epochs 3 --batch_size 16 --lr 3e-5
```

---

## 📦 Output Artifacts

| Path | Description |
|------|-------------|
| `vnct/best_model/` | Best checkpoint (by lowest validation loss), loadable with `AutoModelForSequenceClassification.from_pretrained()` |
| `vnct/model.safetensors` | Final model weights (after all epochs) |
| `vnct/label_map.json` | JSON mapping of category name → integer label |
| `vnct/training_history.csv` | CSV with per-epoch `loss`, `acc`, `precision`, `recall`, `f1` for train/val/test splits |
| `vnct/training_log.txt` | Full timestamped training log |
| `vnct/confusion_matrix.png` | Confusion matrix of the best model on the test set |

## ❤️ Acknowledgements
I would like to thank my former teammates who contributed to my earlier NLP projects during my Bachelor's journey. Your collaboration, discussions, and support helped lay the foundation for this work.
