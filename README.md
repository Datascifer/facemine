```markdown
# Facial Emotion Recognition (FaceMine)

An AI project that learns to identify eight different facial emotions—like happiness, anger, surprise, and sadness—by training on two large photo collections (FER-2013 and CK+) and using smart techniques to ensure even rare expressions are recognized accurately.

---

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Dataset](#dataset)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Models](#models)
- [Usage](#usage)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Repository Structure](#repository-structure)
- [Scripts](#scripts)
- [Contributing](#contributing)
- [License](#license)

---

## Features

- **Combined Dataset**: Merges FER-2013 (in-the-wild images) and CK+ (lab-controlled images) for 8 emotion classes.
- **Five Architectures**:  
  - **MyCNN** – A compact, custom CNN built from scratch.  
  - **MyCNNv2** – Expanded baseline CNN with three convolutional blocks and configurable dropout.  
  - **ResNet-18** – Fine-tuned on ImageNet, adapted to 1×48×48 grayscale.  
  - **VGG-16** (benchmark) – Pretrained VGG-16 backbone, classifier head replaced for emotion logits.  
  - **DenseNet-121** (benchmark) – Pretrained DenseNet-121 backbone, classifier head replaced.  
- **Imbalance Mitigation**: WeightedRandomSampler and data augmentation to handle rare classes.
- **Robust Validation**: 5-fold cross-validation with learning-rate scheduling.
- **End-to-End Scripts**: Download, train, and evaluate all models with a single CLI.

---

## Installation

```bash
git clone https://github.com/yourusername/fer-emotion-recognition.git
cd fer-emotion-recognition
pip install -r requirements.txt
```

---

## Dataset

> **Do Not Commit Data to GitHub**  
> - Add `data/` to `.gitignore`.  
> - Download at runtime via:
>   ```bash
>   python scripts/download_datasets.py
>   ```
> - After running, you should have:  
>   ```
>   data/
>   ├─ fer2013/
>   │  ├─ train/
>   │  └─ test/
>   └─ ckplus/
>      └─ ckextended.csv
>   ```  

If you rename your data folders, update paths in `train.py` and `scripts/evaluate_models.py`.

---

## Exploratory Data Analysis

See `notebooks/EDA.ipynb` for:

- Sample frames from each dataset  
- Class distribution histograms  
- Combined dataset summary  

---

## Models

| Model           | Description                                                                                       |
| --------------- | ------------------------------------------------------------------------------------------------- |
| **MyCNN**       | Two convolutional layers + pooling → two FC layers (baseline).                                    |
| **MyCNNv2**     | Three conv-batchnorm-ReLU blocks → one FC head; configurable dropout.                             |
| **ResNet-18**   | ImageNet-pretrained, first conv adapted to grayscale 48×48, fine-tuned on last block + classifier.|
| **VGG-16**      | ImageNet-pretrained VGG-16, first conv swapped for 1-channel, classifier rebuilt for 8 outputs.   |
| **DenseNet-121**| ImageNet-pretrained DenseNet-121, first conv swapped for 1-channel, classifier rebuilt.           |

> **Note on Kaggle code references**  
> We reviewed several Kaggle notebooks (e.g. VGG16, InceptionResNetV2, DenseNet121 tutorials) to guide our benchmark model definitions and evaluation metrics—only the model‐init and CI pipelines were adapted; the core PyTorch training loops and data handling remain our own.

---

## Usage

Train any model with:

```bash
python train.py \
  --model resnet18 \
  --batch_size 32 \
  --epochs 20 \
  --lr 5e-4
```

Replace `--model` with `mycnn`, `mycnnv2`, `branchcnn`, `vgg16` or `densenet121`.

---

## Training

Key strategies:

- **Pretraining**: Leveraging ImageNet weights for ResNet-18, VGG-16, DenseNet-121.  
- **Regularization**: Dropout (p=0.5), L2 weight decay, batch normalization.  
- **Sampling**: `WeightedRandomSampler` to counter severe class imbalance (e.g. “contempt” with only 14 samples).  
- **Validation**: 5-fold CV with `ReduceLROnPlateau` scheduling.

---

## Evaluation

We provide a comprehensive evaluation script:

```bash
python scripts/evaluate_models.py \
  --data_dir data \
  --ckpt_dir checkpoints \
  --batch_size 32
```

For each model (`mycnn`, `resnet18`, `branchcnn`, `vgg16`, `densenet121`), it reports:

- **Overall Accuracy** & **Macro F1**  
- **Per-class** precision, recall, F1  
- **Confusion Matrix**  

---

## Results

After running `evaluate_models.py`, you’ll get a table like:

| Model           | Accuracy | F1-macro |
| --------------- | -------- | -------- |
| MyCNN           | 0.51     | 0.40     |
| ResNet-18       | 0.52     | 0.42     |
| VGG-16          | 0.55     | 0.45     |
| DenseNet-121    | 0.54     | 0.44     |
| MultiBranchCNN  | 0.53     | 0.43     |

(Your exact numbers may vary with hyperparameter tuning.)

See full per-class reports and confusion matrices in the console output of `scripts/evaluate_models.py`.

---

## Repository Structure

```
├── data/                      # Datasets (ignored by Git)
├── notebooks/                 # EDA and analysis notebooks
│   └── EDA.ipynb
├── models/                    # Model definitions
│   ├── mycnn.py
│   ├── mycnnv2.py
│   ├── resnet18.py
│   ├── branchcnn.py
│   ├── vgg16.py
│   └── densenet121.py
├── scripts/
│   ├── download_datasets.py   # Download FER-2013 & CK+ via Kaggle API
│   └── evaluate_models.py     # Aggregates evaluation metrics across all models
├── train.py                   # Main training & CV entrypoint
├── requirements.txt           # Exact Python package versions
├── README.md                  # This file
└── LICENSE
```

---

## Scripts

- **download_datasets.py**  
  ```bash
  python scripts/download_datasets.py
  ```

- **evaluate_models.py**  
  ```bash
  python scripts/evaluate_models.py --data_dir data --ckpt_dir checkpoints
  ```

---

## Contributing

1. **Fork** & branch:  
   ```
   git checkout -b feature/YourFeature
   ```
2. **Implement** & test.
3. **Commit** with clear messages.  
4. **Push** & open a PR.

Also update docs/notebooks as needed.

---

## License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.
```
