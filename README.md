# Pneumonia Detection using InceptionV3

This project fine-tunes the Inception-V3 model to classify chest X-ray images as either Normal or Pneumonia using the PneumoniaMNIST dataset.

## Dataset
- Source: [Kaggle - PneumoniaMNIST](https://www.kaggle.com/datasets/rijulshr/pneumoniamnist)
- Format: Pre-split into train, validation, and test sets

## Model
- Transfer Learning with InceptionV3 (ImageNet weights, frozen base)
- Custom classification head for binary classification

## Preprocessing
- Images normalized and resized to 299x299
- Labels flattened to shape (n,)
- Class imbalance handled using `class_weight`

## Hyperparameters
- Optimizer: Adam (lr=0.0001)
- Loss: Binary Crossentropy
- Batch size: 32
- Epochs: 15 (with EarlyStopping)

## Evaluation Metrics
- Accuracy, Precision, Recall, F1-score
- Confusion matrix and performance plots

## How to Run

```bash
pip install -r requirements.txt
python pnemonia_classifier.py
```