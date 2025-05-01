# FaceTime: Facial Emotion Recognition

FaceTime is a CNN-based facial expression recognition project designed to classify emotional states from static facial images. It leverages the FER-2013 and CK+ datasets and applies class balancing, data augmentation, and multiple CNN architectures to benchmark performance under realistic conditions.

---

## Folder Structure

```
├── checkpoints/          # Saved model weights (.pth)
├── models/               # Custom and pretrained CNN architectures (full code)
├── plots/                # Training and evaluation visualizations
├── runs/                 # Logs, SMOTE stats, and output artifacts
└── FaceTimev2.ipynb      # Main notebook for training and evaluation
```

---

## Models

Implemented architectures include:

- `MyCNNv1–v6`: Progressive CNN variants with increasing depth
- `ResNet18`: Pretrained ResNet adapted for grayscale FER input
- `VGG16`: Fine-tuned VGG model with customized head
- `DenseNet121`: Modified DenseNet for grayscale, unfrozen final block
- `Ma2024CNN`: Benchmark deep CNN based on Ma (2024)
- `Multi-Branch CNN`: Dual-path architecture for shape and texture features

All models reside in the `models/` directory and are designed to support both baseline and SMOTE-enhanced training.

---

## Training Workflow

Training is done via the `FaceTimev2.ipynb` notebook. The workflow includes:

- Preprocessing (grayscale, resize, normalization)
- Dataset merging (FER-2013 and CK+)
- Weighted sampling and SMOTE for imbalance
- Data augmentation (flips, rotations, scaling)
- Training with AdamW optimizer and early stopping

Settings:
- Batch size: 32  
- Max epochs: 10 (up to 50 for deep models)
- Learning rate: `1e-3`, weight decay: `1e-4`

---

## Evaluation

After training, model performance is measured by:

- Overall accuracy
- Macro-averaged F1 score
- Per-class precision, recall, and F1
- Confusion matrices
- Comparative plots for baseline vs. SMOTE (in `plots/`)

---

## Results

ResNet18 yielded the best overall performance:

- **Accuracy**: 59.00%  
- **F1 Score**: 53.23% (macro average)

Custom models (MyCNNv6, Ma2024CNN) showed strong performance post-SMOTE. Performance improved across all models with SMOTE-enhanced training.

---

## Notebook Usage

To run the pipeline:

1. Open `FaceTimev2.ipynb` in Jupyter or VS Code
2. Run all cells step-by-step
3. Modify model names or training parameters at the top as needed

---

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/yourusername/facetimev2.git
cd facetimev2
pip install -r requirements.txt
```

---

## requirements.txt

```txt
torch>=1.13
torchvision>=0.14
numpy
pandas
matplotlib
seaborn
scikit-learn
imblearn
opencv-python
tqdm
jupyter
```

> You can install via:  
> `pip install -r requirements.txt`

---

## License

This project is licensed under the MIT License. See `LICENSE` for details.
```
