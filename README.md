# pneumonia-xray-classifier

> Binary classification tool to classify chest X-rays (Normal vs Pneumonia) using YOLOv8 and the Kaggle Chest X-Ray Images dataset.

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![YOLOv8](https://img.shields.io/badge/Model-YOLOv8-purple)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-In%20Progress-yellow)

<img src="https://github.com/caolanmaguire/calsickofthis/blob/main/classifier-project.png?raw=true" alt="Project Banner" width="100%"/>

---

## Table of Contents

- [About](#about)
- [Dataset](#dataset)
- [Model](#model)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Limitations & Future Work](#limitations--future-work)
- [License](#license)

---

## About

<img src="https://github.com/caolanmaguire/calsickofthis/blob/main/webpage.png?raw=true"/>

---

## Dataset

<!-- Describe the dataset used (Kaggle Chest X-Ray Images), size, split, and how to download it -->

---

## Model

<!-- Describe the YOLOv8 architecture choice, training config (epochs, image size, etc.) -->

---

## Results

### Kaggle Test Set

              precision    recall  f1-score   support

      NORMAL       0.96      0.42      0.59      4135
   PNEUMONIA       0.32      0.93      0.48      1203

    accuracy                           0.54      5338
   macro avg       0.64      0.68      0.53      5338
weighted avg       0.81      0.54      0.56      5338


              precision    recall  f1-score   support

      NORMAL       0.89      0.93      0.91      4135
   PNEUMONIA       0.70      0.59      0.64      1203

    accuracy                           0.85      5338
   macro avg       0.79      0.76      0.77      5338
weighted avg       0.84      0.85      0.85      5338

![Confusion Matrix](results/train_kaggle/confusion_matrix.png)
![ROC Curve](results/train_kaggle/roc_curve.png)

---

## Installation

```bash
# Clone the repo
git clone https://github.com/caolanmaguire/pneumonia-xray-classifier.git
cd pneumonia-xray-classifier

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

```bash
# Train the model
python main.py

# Run inference / demo mode
python api.py

# navigate to the index.html page in your browser
```

<!-- Expand with any flags, options, or environment variables -->

---

## Project Structure

```
pneumonia-xray-classifier/
├── main.py
├── index.html
├── api.py
├── README.md
├── chest_xray/
└── ...
```

---

## Limitations & Future Work

<!-- Honest notes on dataset bias, model constraints, and what you'd do next -->

---

## License

Distributed under the MIT License. See `LICENSE` for more information.
