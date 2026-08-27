# Customer Sentiment & Urgency Router

**Deep Learning Applications — ELA-1 | Track 2: Customer Experience Focus**

A deep learning pipeline that classifies raw customer-support email text into
5 operational categories, jointly capturing **sentiment** and **urgency**, to
automate inbox routing and prioritization.

## 🔗 Code

**Google Colab Notebook:** https://colab.research.google.com/drive/1_n3fzkqVwJbrzatjLJ081C61HnhxVSSg?usp=sharing

> Upload `customer_sentiment_urgency_router.ipynb` to Colab, click **Share → Anyone with the link → Viewer**, and paste that link above before submitting.

## Problem & Categories

| Label | Category | Meaning / Routing Action |
|---|---|---|
| 0 | `urgent_complaint` | Angry customer, time-critical → escalate immediately |
| 1 | `general_complaint` | Dissatisfied, not time-critical → standard support queue |
| 2 | `urgent_request` | Neutral/positive tone but time-sensitive → fast-track |
| 3 | `neutral_inquiry` | Routine question → standard queue / auto-FAQ |
| 4 | `positive_feedback` | Praise / thanks → low priority, optional follow-up |

## Dataset

- **1,800 synthetic emails** (5 balanced classes, 360 each) — exceeds the 1,500-email minimum
- Built from randomized templates with slot-filling (products, order IDs, amounts, names)
- Realism injected via: synonym substitution, cross-class filler sentences, ~20% neighbor-class sentence borrowing, light typo/word-drop noise, and ~4% label noise
- Split: 70% train / 15% validation / 15% test (1,260 / 270 / 270), stratified by class

## Model Architecture

```
Raw Email Text
  → Text Cleaning (lowercase, strip URLs, mask order IDs/amounts, remove punctuation)
  → Tokenization (Keras Tokenizer, vocab ≤ 8,000, <OOV> token)
  → Padding/Truncation (max length = 120 tokens)
  → Embedding(vocab_size, 64, mask_zero=True)
  → Bidirectional GRU(32 units/direction)
  → Dropout(0.5)
  → Dense(64, ReLU)
  → Dropout(0.5)
  → Dense(5, Softmax)
```

**Key hyperparameters:** Adam optimizer (lr=1e-3), categorical crossentropy loss, batch size 32, max 25 epochs with early stopping (patience=4 on val_loss).

## Results

Evaluated on the held-out test set (270 emails):

| Metric | Value |
|---|---|
| **Test Accuracy** | **96.30%** |
| Test Loss | 0.2670 |
| **Macro F1-Score** | **0.9629** |
| Weighted F1-Score | 0.9629 |
| Epochs trained (early stopped) | 8 |

**Per-class results:**

| Category | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| urgent_complaint | 0.9464 | 1.0000 | 0.9725 | 53 |
| general_complaint | 0.9630 | 0.9630 | 0.9630 | 54 |
| urgent_request | 0.9623 | 0.9444 | 0.9533 | 54 |
| neutral_inquiry | 0.9636 | 0.9636 | 0.9636 | 55 |
| positive_feedback | 0.9808 | 0.9444 | 0.9623 | 54 |

Training converged within 4–5 epochs; remaining misclassifications occur mostly between semantically adjacent classes (e.g. `urgent_request` vs. `neutral_inquiry`), consistent with the intentional class overlap built into the dataset.

## Files

| File | Description |
|---|---|
| `customer_sentiment_urgency_router.ipynb` | Full pipeline: dataset generation, preprocessing, model, training, evaluation |
| `Project_Report_Customer_Sentiment_Urgency_Router.pdf` | 3-page project report (abstract, architecture, metrics) |
| `README.md` | This file |

## How to Run

1. Open the notebook in Google Colab (GPU runtime optional — CPU trains in under a minute).
2. Run all cells top to bottom (`Runtime → Run all`).
3. Outputs are saved locally in the Colab session: `gru_email_router.keras`, `tokenizer.json`, `accuracy_loss_curves.png`, `confusion_matrix.png`, `metrics_summary.json`.

## Possible Extensions

- Fine-tune a DistilBERT encoder for greater robustness to novel phrasing
- Add a human-in-the-loop review queue for low-confidence predictions
- Replace synthetic data with real, anonymized support tickets for production deployment
