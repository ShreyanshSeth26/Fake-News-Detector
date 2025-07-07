# 📰 Fake News Detection with DistilBERT + LoRA

A robust and lightweight fake news detection system built using a fine-tuned DistilBERT model enhanced with PEFT (LoRA). This project leverages the power of transformers for binary classification—classifying news as either **real** or **fake**.

---

## 🚀 Features

- ✅ Fine-tuned `DistilBERT` for binary sequence classification
- ⚡ Lightweight LoRA-based parameter-efficient fine-tuning (via `peft`)
- 🔍 Probabilistic predictions using softmax scores
- 📦 Integrated inference pipeline for real-time use
- 📊 Rich evaluation metrics and error analysis (code-ready)
- 💾 Easily portable with support for saved checkpoints

---

## 📁 Project Structure

Fake-News-Detector/
│
├── outputs/
│ └── checkpoints/
│ └── BestModel/ # Final fine-tuned model checkpoint
│
├── main.py # Inference script using LoRA model
├── utils/ # Preprocessing, training, etc. (if any)
├── requirements.txt
└── README.md

---

## 🧠 Model Architecture

- **Base Model**: [`distilbert-base-uncased`](https://huggingface.co/distilbert-base-uncased)
- **Fine-Tuned** on: Custom Fake/Real news dataset
- **LoRA Adapter**: Efficient fine-tuning using low-rank adaptation via HuggingFace's `peft` library

---
## DataSets Used

- https://www.kaggle.com/datasets/emineyetm/fake-news-detection-datasets
- http://datasets.activeloop.ai/docs/ml/datasets/liar-dataset/
- https://www.kaggle.com/datasets/abhinavkrjha/fake-news-challenge?select=train_stances.csv\
  
---

## 📦 Requirements

```bash
pip install -r requirements.txt
