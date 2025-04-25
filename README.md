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

> **Do Not Commit Data to GitHub**
>
> 1. **Ignore the data folder.** Add `data/` to your `.gitignore`.
> 2. **Download datasets at runtime.** Use the provided script:
>    ```bash
>    python scripts/download_datasets.py
>    ```
> 3. **Verify your data directory.** After running the script, ensure you have:
>    - `data/fer2013/` containing the FER-2013 images (with `train/` and `test/` subfolders)
>    - `data/ckplus/` containing the CK+ dataset and `ckextended.csv`
>
> If your data directory is named differently, update paths in `train.py` and the notebooks accordingly.

## Exploratory Data Analysis

The `notebooks/EDA.ipynb` contains:
- Sample frames from each dataset.
- Class distribution histograms.
- Combined dataset summary tables.

## Models

This project implements and compares three model architectures for facial emotion recognition:

1. **Baseline CNN** – A simple convolutional network built from scratch (`MyCNN`), with two convolutional layers, pooling, and fully connected layers.
2. **Fine-Tuned CNN** – A pretrained ResNet-18 model (ImageNet weights) adapted for grayscale input and fine-tuned on the combined dataset.
3. **Multi-Branch CNN** – A custom architecture with specialized branches for extracting different feature types (e.g., texture vs. shape), merged before classification.

**Techniques Used Across Models:**

- **Data Augmentation:** Random flips, rotations, and scaling to simulate real-world variations.
- **Regularization:** Dropout (p=0.5), L2 weight decay, and batch normalization to reduce overfitting.
- **Optimization:** Experiments with AdamW and SGD optimizers, along with learning-rate tuning and scheduling.

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

The repository is organized as follows:

```
├── data/                   # Downloaded datasets (ignored in Git)
├── notebooks/              # Jupyter notebooks for EDA and analysis
│   └── EDA.ipynb           # Exploratory Data Analysis
├── models/                 # Model definition modules
│   ├── mycnn.py            # Implementation of the custom CNN architecture
│   └── resnet18.py         # Fine-tuned ResNet-18 model class
├── scripts/                # Utility scripts
│   └── download_datasets.py # Script to download FER-2013 and CK+ datasets
├── train.py                # Main training and evaluation entrypoint
├── requirements.txt        # List of Python package dependencies
├── README.md               # Project overview and instructions
└── LICENSE                 # License information
```

- **data/**: Contains raw dataset folders after running `download_datasets.py`. This directory is added to `.gitignore` to avoid committing large data files.
- **notebooks/**: Includes analysis notebooks for visualizing sample images and class distributions.
- **models/**: Holds the model architecture code, allowing easy imports and modifications.
- **scripts/**: Utility scripts to automate dataset downloading and setup.
- **train.py**: Orchestrates data loading, model instantiation, training loops, evaluation, and result logging based on CLI arguments.
- **requirements.txt**: Ensures reproducible environments by specifying exact package versions.
- **LICENSE**: Details the MIT licensing terms for this project.
## Scripts

Utility scripts are provided in the `scripts/` directory:

- **download_datasets.py**: Downloads the FER-2013 and CK+ datasets into `data/fer2013/` and `data/ckplus/`, respectively. Run:
  ```bash
  python scripts/download_datasets.py
  ```

Be sure to update any paths in your own code if you rename the `data/` folder.

## Contributing

Contributions are welcome! Please follow these steps:

1. **Fork** this repository and create a new branch:
   ```bash
   git checkout -b feature/YourFeature
   ```
2. **Develop** your feature or bugfix, ensuring code style consistency and adding tests where appropriate.
3. **Commit** your changes with clear, descriptive messages:
   ```bash
   git commit -m "Add <feature description>"
   ```
4. **Push** your branch to GitHub and open a **Pull Request**, describing your changes and referencing related issues.

Please also update documentation (this README or notebooks) to reflect any changes in functionality.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

