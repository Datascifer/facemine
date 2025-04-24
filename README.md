# facemine
An AI project that learns to identify eight different facial emotions—like happiness, anger, surprise, and sadness—by training on two large photo collections (FER-2013 and CK+) and using smart techniques to make sure even rare expressions are recognized accurately.

# Facial Emotion Recognition

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
- [Results](#results)
- [Repository Structure](#repository-structure)
- [Contributing](#contributing)
- [License](#license)

---

## Features

- **Combined Dataset**: Merges FER-2013 (in-the-wild images) and CK+ (lab-controlled images) for 8 emotion classes.
- **Custom CNN & ResNet-18**: Implements a lightweight CNN and fine-tuned ResNet-18 adapted for grayscale 48×48 inputs.
- **Imbalance Mitigation**: Uses weighted sampling and data augmentation to handle rare classes.
- **Validation**: Employs 5-fold cross-validation and hyperparameter tuning for robust performance.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/fer-emotion-recognition.git
   cd fer-emotion-recognition
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Dataset

This project uses two datasets:

- **FER-2013**: Public benchmark of 48×48 grayscale images labeled with 7 emotions.
- **CK+**: Laboratory-controlled dataset with finely annotated facial expressions.

Datasets are automatically downloaded using `kagglehub`:
```python
import kagglehub
path_fer = kagglehub.dataset_download("msambare/fer2013")
path_ck  = kagglehub.dataset_download("davilsena/ckdataset")
```

## Exploratory Data Analysis

The `notebooks/EDA.ipynb` contains:
- Sample frames from each dataset.
- Class distribution histograms.
- Combined dataset summary tables.

## Models

- **MyCNN**: Two-convolutional-layer network with dropout and batch normalization.
- **ResNet-18**: Pretrained on ImageNet, modified stem for single-channel input, fine-tuned on combined dataset.

## Usage

To run training and evaluation:

```bash
python train.py \
  --model resnet18 \
  --batch_size 32 \
  --epochs 20 \
  --use_weighted_sampler
```

Replace `--model` with `mycnn` to train the custom CNN.

## Training

Key strategies:
- **Pretraining**: Leverage ImageNet weights.
- **Regularization**: Dropout (p=0.5), L2 weight decay.
- **Sampling**: WeightedRandomSampler for class imbalance.
- **Validation**: 5-fold cross-validation with learning-rate scheduling.

## Results

- **ResNet-18**: ~52% accuracy, ~0.42 F1 (5-fold average).
- **MyCNN**: ~51% accuracy, ~0.40 F1 (5-fold average).

Compare these to published baselines in the [Literature Review](docs/LITERATURE_REVIEW.md).

## Repository Structure

```
├── data/                   # Downloaded datasets
├── notebooks/              # EDA and analysis notebooks
│   └── EDA.ipynb
├── models/                 # Model definitions and architectures
│   ├── mycnn.py
│   └── resnet18.py
├── train.py                # Training and evaluation script
├── requirements.txt        # Python dependencies
├── README.md
└── LICENSE
```

## Contributing

Contributions are welcome! Please open issues or submit pull requests for bug fixes, new features, or improvements.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

