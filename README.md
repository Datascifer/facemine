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
- [Contributing](#contributing)  
- [License](#license)  

---

## Features

- **Combined Dataset**: Merges FER-2013 (in-the-wild images) and CK+ (lab-controlled images) for 8 emotion classes.  
- **Six Architectures**:  
  1. **MyCNN** – Baseline CNN built from scratch (2025).  
  2. **MyCNNv2** – Expanded 3-block CNN with dynamic flattening (2025).  
  3. **EmotionResNet18** – Fine-tuned ResNet-18 for 1×48×48 grayscale (2025).  
  4. **Ma2024CNN** – “Convolutional Neural Networks-based Evaluation for the FER-2013 and Revised CK+ Datasets” architecture from Ma (2024).  
  5. **EmotionVGG16** – Fine-tuned VGG-16 adapted for grayscale (2025).  
  6. **EmotionDenseNet121** – Fine-tuned DenseNet-121 adapted for grayscale (2025).  
- **Imbalance Mitigation**: Weighted sampling and data augmentation to handle rare classes.  
- **Validation**: 5-fold cross-validation, hyper­parameter tuning, and learning-rate scheduling.  
- **Unified Evaluation**: Single script reports overall accuracy, macro F1, per-class metrics, and confusion matrices.  

---

## Installation

1. **Clone the repository**  
   ```bash
   git clone https://github.com/yourusername/fer-emotion-recognition.git
   cd fer-emotion-recognition
   ```

2. **Install dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

---

## Dataset

> **Do Not Commit Data to GitHub**  
>
> 1. **.gitignore** includes `data/`.  
> 2. **Download at runtime**:  
>    ```bash
>    python scripts/download_datasets.py
>    ```
> 3. **Expected structure** under `data/`:  
>    ```
>    data/
>    ├── fer2013/
>    │   ├── train/
>    │   └── test/
>    └── ckplus/
>        └── ckeextended.csv
>    ```
> 4. If your folders differ, update paths in `train.py` and `scripts/evaluate_models.py`.

---

## Exploratory Data Analysis

See `notebooks/EDA.ipynb` for:

- Sample frames from each dataset  
- Class distribution histograms  
- Combined dataset summary tables  

---

## Models

We compare six model architectures for facial emotion recognition:

| Model           | Description                                                                                             | Source                          |
|-----------------|---------------------------------------------------------------------------------------------------------|---------------------------------|
| **MyCNN**       | Two conv-pool blocks + two FC layers, built from scratch.                                               | `models/mycnn.py`               |
| **MyCNNv2**     | Three conv-pool blocks, dynamic flatten, configurable dropout, Kaiming/Xavier init.                     | `models/mycnnv2.py`             |
| **ResNet-18**   | Pretrained ResNet-18 stem modified for 1×48×48, freeze layers except last block & head.                | `models/resnet18.py`            |
| **Ma2024CNN**   | CNN from Ma (2024): data-driven evaluation on FER-2013 & CK+, uses normalization & augmentation.        | `models/ma2024cnn.py`           |
| **VGG-16**      | Pretrained VGG-16 modified for grayscale input, freeze early conv blocks, new classifier head.         | `models/vgg16.py`               |
| **DenseNet121** | Pretrained DenseNet-121 modified for grayscale, freeze all but last dense block & classifier.          | `models/densenet121.py`         |

**Techniques across all models**:

- **Data Augmentation**: flips, rotations, zooms  
- **Regularization**: Dropout (p=0.5), L2 weight decay, BatchNorm  
- **Optimization**: AdamW, learning-rate scheduling  

---

## Usage

### 1. Download datasets  
```bash
python scripts/download_datasets.py
```

### 2. Train a model  
```bash
python train.py \
  --model resnet18 \
  --batch_size 32 \
  --epochs 20 \
  --lr 1e-3
```
Replace `--model` with one of `mycnn`, `mycnnv2`, `resnet18`, `ma2024cnn`, `vgg16`, or `densenet121`.

---

## Training

- **Pretraining**: leverage ImageNet weights where applicable  
- **Sampler**: `WeightedRandomSampler` for class imbalance  
- **Validation**: 5-fold CV, `ReduceLROnPlateau` scheduler  

---

## Evaluation

After training, place your best checkpoints into `checkpoints/` (e.g. `best_resnet18.pth`, `best_ma2024cnn.pth`, etc.), then:

```bash
python scripts/evaluate_models.py \
  --data_dir data \
  --ckpt_dir checkpoints \
  --batch_size 32
```

This script will, for each model:

- Compute **Overall Accuracy** & **Macro F1**  
- Print **Per-class Precision / Recall / F1**  
- Display the **Confusion Matrix**

---

## Results

Here’s a sample of 5-fold cross-validation on **ResNet-18**:

| Fold | Accuracy | F1-macro |
|:----:|:--------:|:--------:|
| 1    | 0.52     | 0.42     |
| 2    | 0.50     | 0.39     |
| 3    | 0.51     | 0.40     |
| 4    | 0.53     | 0.42     |
| 5    | 0.51     | 0.40     |
| **Avg** | **0.51** | **0.41** |

Compare against **Ma 2024 CNN** and other benchmarks listed in [docs/LITERATURE_REVIEW.md].

---

## Repository Structure

```
├── data/                    # Downloaded datasets (ignored in Git)
├── docs/                    # Supplementary docs (e.g., literature review)
├── models/                  # All model definition modules
│   ├── mycnn.py
│   ├── mycnnv2.py
│   ├── resnet18.py
│   ├── ma2024cnn.py
│   ├── vgg16.py
│   └── densenet121.py
├── notebooks/               # Jupyter notebooks for EDA and analysis
│   └── EDA.ipynb
├── scripts/
│   ├── download_datasets.py # Downloads FER-2013 & CK+ via Kaggle API
│   └── evaluate_models.py   # Unified evaluation reporting metrics & CM
├── train.py                 # Main training & validation entrypoint
├── requirements.txt         # Python package dependencies
├── README.md                # Project overview and instructions
└── LICENSE                  # MIT license
```

---

## Contributing

We welcome contributions! Please:

1. **Fork** & create a feature branch:
   ```bash
   git checkout -b feature/your-feature
   ```
2. **Implement** & **test** your changes.  
3. **Commit** with descriptive messages:
   ```bash
   git commit -m "Add MultiBranchCNN improvements"
   ```
4. **Push** & open a **Pull Request**.

Remember to update docs or notebooks if functionality changes.

---

## License

This project is released under the **MIT License**. See [LICENSE](LICENSE) for details.
