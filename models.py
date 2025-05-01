# Install dependencies
## Dependencies
!pip install torch torchvision kagglehub

# Data load and preprocessing
import os
import pandas as pd
import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import kagglehub
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Define the common transformation for loading images
common_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),   # Convert to grayscale (if not already)
    transforms.Resize((48, 48)),                    # Resize to 48x48
    transforms.ToTensor(),                          # Convert to tensor (scales values to [0, 1])
    transforms.Normalize(mean=[0.5], std=[0.5]),    # Normalize to range [-1, 1] (optional, based on your dataset)
])

# Download datasets (using kagglehub in case datasets are not available locally)
path_fer = kagglehub.dataset_download("msambare/fer2013")
path_ck = kagglehub.dataset_download("davilsena/ckdataset")

print("FER-2013 path:", path_fer)
print("CK+ path:", path_ck)

# Load FER-2013 dataset
fer_train = datasets.ImageFolder(
    root=os.path.join(path_fer, "train"),
    transform=common_transform
)
fer_test = datasets.ImageFolder(
    root=os.path.join(path_fer, "test"),
    transform=common_transform
)

# Custom dataset class for CK+ dataset (CSV-based)
class CustomCKDataset(Dataset):
    def __init__(self, csv_file, usage, transform=None):
        df = pd.read_csv(csv_file)
        self.df = (df[df["Usage"] == usage].reset_index(drop=True)
                   if usage == "Training"
                   else df[df["Usage"] != "Training"].reset_index(drop=True))
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        label = int(row["emotion"])
        pixels = np.fromstring(row["pixels"], sep=" ", dtype=np.uint8).reshape(48, 48)
        img = Image.fromarray(pixels, mode="L")
        if self.transform:
            img = self.transform(img)
        return img, label

# Load CK+ dataset
csv_path = os.path.join(path_ck, "ckextended.csv")
ck_train = CustomCKDataset(csv_file=csv_path, usage="Training", transform=common_transform)
ck_test = CustomCKDataset(csv_file=csv_path, usage="PublicTest", transform=common_transform)

# Combine the FER-2013 and CK+ datasets
combined_train = torch.utils.data.ConcatDataset([fer_train, ck_train])
combined_test  = torch.utils.data.ConcatDataset([fer_test,  ck_test])

# Loaders
batch_size = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

fer_train_loader      = DataLoader(fer_train,      batch_size=batch_size, shuffle=True,  num_workers=1, pin_memory=device.type=='cuda')
fer_test_loader       = DataLoader(fer_test,       batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=device.type=='cuda')
ck_train_loader       = DataLoader(ck_train,       batch_size=batch_size, shuffle=True,  num_workers=1, pin_memory=device.type=='cuda')
ck_test_loader        = DataLoader(ck_test,        batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=device.type=='cuda')
combined_train_loader = DataLoader(combined_train, batch_size=batch_size, shuffle=True,  num_workers=1, pin_memory=device.type=='cuda')
combined_test_loader  = DataLoader(combined_test,  batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=device.type=='cuda')

# Print the shape of the images after preprocessing
# FER-2013 only
for inputs, labels in fer_train_loader:
    print("FER-2013 batch shape:", inputs.shape)
    break

# CK+ only
for inputs, labels in ck_train_loader:
    print("CK+    batch shape:", inputs.shape)
    break

# Combined
for inputs, labels in combined_train_loader:
    print("Combined batch shape:", inputs.shape)
    break

# Exploratory data analysis
# Pad FER‐2013 to length 8 (add zero count for 'contempt')
fer_labels    = fer_train.classes + ['contempt']
fer_counts7   = np.bincount([y for _,y in fer_train])             # length=7
fer_counts    = np.pad(fer_counts7, (0,1), constant_values=0)     # now length=8

# Inspect Class and Mappings
## FER-2013
print("FER classes:", fer_train.classes)
print("FER class→idx:", fer_train.class_to_idx)

## CK+ (numeric labels)
ck_df = pd.read_csv(csv_path)
print("CK+ emotion labels:", sorted(ck_df['emotion'].unique()))

# Single helper
def plot_distribution(counts, labels, title):
    plt.figure(figsize=(6, 4))
    plt.bar(labels, counts)
    plt.xticks(rotation=45)
    plt.title(title)
    plt.ylabel("Sample count")
    plt.tight_layout()
    plt.show()

# Define class names
class_names = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise", "contempt"]

# Pad FER‐2013 counts to length 8 (add zero for 'contempt')
fer_labels  = fer_train.classes + ['contempt']                      # 7→8 names
fer_counts7 = np.bincount([y for _, y in fer_train])               # length=7
fer_counts  = np.pad(fer_counts7, (0,1), constant_values=0)        # now length=8

# CK+ train/test counts (labels 0–7)
ck_df         = pd.read_csv(csv_path)
train_ck      = ck_df[ck_df["Usage"]=="Training"]["emotion"]
test_ck       = ck_df[ck_df["Usage"]!="Training"]["emotion"]
train_ck_cnts = train_ck.value_counts().sort_index().reindex(range(8), fill_value=0).values
test_ck_cnts  = test_ck.value_counts().sort_index().reindex(range(8), fill_value=0).values

# Combined = FER + CK+ train
combined_cnts = fer_counts + train_ck_cnts

# Plot them
plot_distribution(fer_counts,    fer_labels,  "FER-2013 Class Distribution")
plot_distribution(train_ck_cnts, class_names, "CK+ Training Distribution")
plot_distribution(test_ck_cnts,  class_names, "CK+ Test Distribution")
plot_distribution(combined_cnts, class_names, "Combined Train Distribution")

# Check Image per Class
import random
from torchvision.utils import make_grid

def show_samples(dataset, classes, n_samples=8):
    # pick one sample per class (if available)
    imgs, labs = [], []
    for cls_idx, cls_name in enumerate(classes):
        # find first n_samples of this class
        found = 0
        for img, lbl in dataset:
            if lbl==cls_idx and found<n_samples:
                imgs.append(img)
                labs.append(cls_name)
                found+=1
            if found>=n_samples: break
    grid = make_grid(imgs, nrow=n_samples, normalize=True, pad_value=1)
    plt.figure(figsize=(n_samples*1.5, len(classes)*1.5/4))
    plt.imshow(grid.permute(1,2,0).squeeze(), cmap='gray')
    plt.axis('off')
    plt.title("Examples per Class")
    plt.show()

# For FER-2013 only
show_samples(fer_train, fer_train.classes, n_samples=5)

# For CK+ only (has 8 classes 0–7)
show_samples(ck_train, class_names,    n_samples=5)

# For the combined dataset
show_samples(combined_train, class_names, n_samples=5)

from torchvision import datasets, transforms

# Define a “stat only” transform (no normalization)
stat_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48,48)),
    transforms.ToTensor(),  # [0,1] range
])

# FER-2013
fer_stat_ds = datasets.ImageFolder(
    root=os.path.join(path_fer, "train"),
    transform=stat_transform
)
all_fer = torch.cat([img.view(-1) for img,_ in fer_stat_ds])
print("FER-2013 → mean:", all_fer.mean().item(), " std:", all_fer.std().item())

# CK+ (Training split)
ck_stat_ds = CustomCKDataset(
    csv_file=csv_path,
    usage="Training",
    transform=stat_transform
)
all_ck = torch.cat([img.view(-1) for img,_ in ck_stat_ds])
print("CK+ Training → mean:", all_ck.mean().item(), " std:", all_ck.std().item())

# Combined (just concat the two stat datasets)
from torch.utils.data import ConcatDataset
combined_stat_ds = ConcatDataset([fer_stat_ds, ck_stat_ds])
all_combined = torch.cat([img.view(-1) for img,_ in combined_stat_ds])
print("Combined → mean:", all_combined.mean().item(), " std:", all_combined.std().item())


# Model Architecture
from torchvision import datasets, transforms

# Define a “stat only” transform (no normalization)
stat_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48,48)),
    transforms.ToTensor(),  # [0,1] range
])

# FER-2013
fer_stat_ds = datasets.ImageFolder(
    root=os.path.join(path_fer, "train"),
    transform=stat_transform
)
all_fer = torch.cat([img.view(-1) for img,_ in fer_stat_ds])
print("FER-2013 → mean:", all_fer.mean().item(), " std:", all_fer.std().item())

# CK+ (Training split)
ck_stat_ds = CustomCKDataset(
    csv_file=csv_path,
    usage="Training",
    transform=stat_transform
)
all_ck = torch.cat([img.view(-1) for img,_ in ck_stat_ds])
print("CK+ Training → mean:", all_ck.mean().item(), " std:", all_ck.std().item())

# Combined (just concat the two stat datasets)
from torch.utils.data import ConcatDataset
combined_stat_ds = ConcatDataset([fer_stat_ds, ck_stat_ds])
all_combined = torch.cat([img.view(-1) for img,_ in combined_stat_ds])
print("Combined → mean:", all_combined.mean().item(), " std:", all_combined.std().item())


# Utility Algorithms
!pip install tensorboard # install dependency

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from torch.utils.tensorboard import SummaryWriter

def save_model(model, name):
    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), f"models/{name}.pth")
    print(f"Saved {name}.pth")

# EarlyStopping
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0):
        self.patience,self.min_delta,self.counter=patience,min_delta,0
        self.best_loss=np.inf; self.early_stop=False
    def __call__(self,val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss,val_loss,self.counter = val_loss,val_loss,0
        else:
            self.counter+=1
        if self.counter>=self.patience: self.early_stop=True
        return self.early_stop

# Train & Validate
def train_and_evaluate(model, train_loader, val_loader, num_epochs, verbose, early_stopping, optimizer, scheduler=None):
    writer = SummaryWriter(log_dir='./runs')
    best_wts = model.state_dict(); best_loss=np.inf
    train_losses=[]; val_losses=[]

    for epoch in range(num_epochs):
        # TRAIN
        model.train(); epoch_loss=0
        for x,y in train_loader:
            x,y = x.to(device),y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss=nn.CrossEntropyLoss()(out,y)
            loss.backward(); optimizer.step()
            epoch_loss+=loss.item()
        epoch_loss/=len(train_loader); train_losses.append(epoch_loss)

        # VALID
        model.eval(); val_loss=0; preds=[]; labs=[]
        with torch.no_grad():
            for x,y in val_loader:
                x,y = x.to(device),y.to(device)
                out= model(x)
                val_loss+=nn.CrossEntropyLoss()(out,y).item()
                _,p = torch.max(out,1)
                preds+=p.cpu().tolist(); labs+=y.cpu().tolist()
        val_loss/=len(val_loader); val_losses.append(val_loss)

        acc = accuracy_score(labs,preds)
        f1  = f1_score(labs,preds,average='macro')
        writer.add_scalar('Loss/Train', epoch_loss, epoch)
        writer.add_scalar('Loss/Val',   val_loss,   epoch)
        writer.add_scalar('Acc/Val',    acc,        epoch)
        if verbose:
            print(f"Epoch {epoch+1}/{num_epochs}  Train:{epoch_loss:.4f}  Val:{val_loss:.4f}  Acc:{acc:.4f}  F1:{f1:.4f}")

        if early_stopping(val_loss):
            print("Early stopping"); break
        if scheduler: scheduler.step(val_loss)
        if val_loss<best_loss:
            best_loss=val_loss; best_wts=model.state_dict()

    model.load_state_dict(best_wts)
    writer.close()
    return model, (train_losses,val_losses)


# Model Evaluation
def evaluate_model(model, loader, all_names, test_type='combined'):
    model.eval()
    all_labels, all_preds = [], []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())

    # determine which labels appear in this split
    uniq = sorted(set(all_labels))
    names = [all_names[i] for i in uniq]

    acc = accuracy_score(all_labels, all_preds)
    f1  = f1_score(all_labels, all_preds, average='macro')

    print(f"\n--- {test_type.upper()} ---")
    print(classification_report(
        all_labels,
        all_preds,
        labels=uniq,
        target_names=names,
        zero_division=0
    ))

    cm = confusion_matrix(all_labels, all_preds, labels=uniq)
    plt.figure(figsize=(len(uniq), len(uniq)))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=names, yticklabels=names
    )
    plt.title(f"{model.__class__.__name__} ({test_type})")
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

    return acc, f1, cm


# Model optimization
model_classes = [
    ('MyCNN',       MyCNN),
    ('MyCNNv2',     MyCNNv2),
    ('MyCNNv3',     MyCNNv3),
    ('MyCNNv4',     MyCNNv4),
    ('MyCNNv5',     MyCNNv5),
    ('MyCNNv6',     MyCNNv6),
    ('ResNet18',    EmotionResNet18),
    ('VGG16',       EmotionVGG16),
    ('DenseNet121', EmotionDenseNet121),
    ('Ma2024CNN',   Ma2024CNN),
]

models_dict = {}
optimizers_dict = {}
schedulers    = {}

for name, cls in model_classes:
    m = cls(num_classes=8).to(device)
    lr = 1e-4 if 'v6' in name.lower() else 1e-3
    opt = optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', patience=3, factor=0.1)
    models_dict[name]     = m
    optimizers_dict[name] = opt
    schedulers[name]      = sch


# Train model
# Define split-specific label lists
fer_names  = fer_train.classes                  # ['angry','disgust','fear','happy','neutral','sad','surprise']
full_names = class_names                        # ['angry','disgust','fear','happy','neutral','sad','surprise','contempt']

# Training histories and final results
histories = {}
results    = {}

# Early stopping configuration
early_stopping = EarlyStopping(patience=3, min_delta=0.01)

for name, model in models_dict.items():
    print(f"\n### Training {name} ###")
    optimizer = optimizers_dict[name]
    scheduler = schedulers[name]

    # Train on combined dataset
    trained_model, (train_losses, val_losses) = train_and_evaluate(
        model,
        combined_train_loader,
        combined_test_loader,
        num_epochs=10,
        verbose=1,
        early_stopping=early_stopping,
        optimizer=optimizer,
        scheduler=scheduler
    )
    save_model(trained_model, name)

    # Store per-epoch losses
    histories[name] = (train_losses, val_losses)

    # Evaluate on FER-2013 (7 classes)
    acc_f, f1_f, _ = evaluate_model(
        trained_model,
        fer_test_loader,
        fer_names,
        test_type='fer'
    )
    # Evaluate on CK+ (8 classes)
    acc_c, f1_c, _ = evaluate_model(
        trained_model,
        ck_test_loader,
        full_names,
        test_type='ck'
    )
    # Evaluate on Combined (8 classes)
    acc_a, f1_a, _ = evaluate_model(
        trained_model,
        combined_test_loader,
        full_names,
        test_type='combined'
    )

    # Collect results
    results[name] = {
        'FER-2013': (acc_f, f1_f),
        'CK+':      (acc_c, f1_c),
        'Combined': (acc_a, f1_a)
    }

# Print final summary
print("\n=== FINAL RESULTS ===")
for name, res in results.items():
    print(f"\n{name}:")
    for split, (acc, f1) in res.items():
        print(f"  {split:9s} → Acc: {acc:.4f}, F1: {f1:.4f}")


# Post training SMOTE
!pip install imbalanced-learn

import imblearn

# SMOTE-BALANCED TRAIN+EVAL
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from imblearn.over_sampling import SMOTE

# assume fer_train, ck_train, combined_train, fer_test_loader, ck_test_loader,
# combined_test_loader, models_dict, optimizers_dict, schedulers, train_and_evaluate,
# evaluate_model, save_model, class_names, early_stopping, device, batch_size are already defined

def make_smote_loader(dataset, batch_size=32):
    # Flatten images & extract labels
    X = np.vstack([img.view(-1).cpu().numpy() for img, _ in dataset])
    y = np.array([lbl for _, lbl in dataset])
    # Apply SMOTE
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X, y)
    # Reconstruct a balanced DataLoader
    Xt = torch.from_numpy(X_res).float().view(-1, 1, 48, 48)
    yt = torch.from_numpy(y_res).long()
    ds = TensorDataset(Xt, yt)
    return DataLoader(ds,
                      batch_size=batch_size,
                      shuffle=True,
                      num_workers=1,
                      pin_memory=(device.type=='cuda'))

# Create SMOTE‐balanced loaders for each split
fer_smote_loader      = make_smote_loader(fer_train,      batch_size)
ck_smote_loader       = make_smote_loader(ck_train,       batch_size)
combined_smote_loader = make_smote_loader(combined_train, batch_size)

# Dictionary to collect results
smote_results = {'FER-2013': {}, 'CK+': {}, 'Combined': {}}

# Iterate over each split
for split, (train_loader, test_loader) in {
    'FER-2013':      (fer_smote_loader,      fer_test_loader),
    'CK+':           (ck_smote_loader,       ck_test_loader),
    'Combined':      (combined_smote_loader, combined_test_loader)
}.items():
    print(f"\n=== SMOTE on {split} ===")
    for name, model in models_dict.items():
        optimizer = optimizers_dict[name]
        scheduler = schedulers[name]
        # Train
        mdl_sm, _ = train_and_evaluate(
            model,
            train_loader,
            test_loader,
            num_epochs=10,
            verbose=1,
            early_stopping=early_stopping,
            optimizer=optimizer,
            scheduler=scheduler
        )
        # Save
        save_model(mdl_sm, f"{name}_{split.replace('+','plus')}_smote")
        # Evaluate
        acc, f1, _ = evaluate_model(
            mdl_sm,
            test_loader,
            class_names,
            test_type=f"{split} (SMOTE)"
        )
        smote_results[split][name] = (acc, f1)

# Print summary
print("\n=== SMOTE-BALANCED RESULTS BY SPLIT ===")
for split, res in smote_results.items():
    print(f"\n--- {split} ---")
    for name, (acc, f1) in res.items():
        print(f"{name:12s} → Acc: {acc:.4f}, F1: {f1:.4f}")


# Model Checkpoint and Reload
import torch, pickle
from torch import optim

# load your pickled artifacts
with open('checkpoints/train_artifacts.pkl','rb') as f:
    art = pickle.load(f)
histories      = art['histories']
results        = art['results']
smote_results  = art['smote_results']
class_names    = art['class_names']
batch_size     = art['batch_size']
device         = art['device']

# re‐define your model classes list exactly as before
model_classes = [
    ('MyCNN',       MyCNN),
    ('MyCNNv2',     MyCNNv2),
    # … all your other (name, class) pairs …
    ('Ma2024CNN',   Ma2024CNN),
]

# rebuild the dicts properly
models_dict     = {}
optimizers_dict = {}
schedulers      = {}

for name, cls in model_classes:
    m   = cls(num_classes=len(class_names)).to(device)
    lr  = 1e-4 if 'v6' in name.lower() else 1e-3
    opt = optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', patience=3, factor=0.1)

    models_dict[name]     = m
    optimizers_dict[name] = opt
    schedulers[name]      = sch

# load each checkpoint back
for name, model in models_dict.items():
    ckpt = torch.load(f'checkpoints/{name}_baseline.ckpt', map_location=device)
    model.load_state_dict(     ckpt['model_state_dict'])
    optimizers_dict[name].load_state_dict( ckpt['optimizer_state_dict'])
    schedulers[name].load_state_dict(      ckpt['scheduler_state_dict'])

print("✅ All models, optimizers and schedulers reloaded.")


# Final graphs
# ensure output dirs exist
os.makedirs("plots/confusion_matrices", exist_ok=True)
os.makedirs("plots", exist_ok=True)

# Confusion‐matrix PNGs
for tag, models in confusion_matrices.items():            # 'baseline' / 'smote'
    for model_name, splits in models.items():
        for split, cm in splits.items():
            if not (isinstance(cm, np.ndarray) and cm.ndim == 2):
                continue

            labels = fer_names if split == 'FER-2013' else class_names

            plt.figure(figsize=(6,6))
            sns.heatmap(
                cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels
            )
            plt.title(f"{model_name} [{tag}] — {split}")
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.tight_layout()
            fname = f"plots/confusion_matrices/{tag}_{model_name}_{split.replace('+','plus')}.png"
            plt.savefig(fname)
            plt.close()

# Baseline vs SMOTE bar chart (Combined)
x = np.arange(len(model_names))
baseline_acc = [results[name]['Combined'][0]     for name in model_names]
smote_acc    = [smote_results['Combined'][name][0] for name in model_names]

plt.figure(figsize=(10,5))
width = 0.35
plt.bar(x - width/2, baseline_acc, width, label="Baseline")
plt.bar(x + width/2, smote_acc,    width, label="SMOTE")
plt.xticks(x, model_names, rotation=45, ha='right')
plt.ylabel("Accuracy")
plt.title("Baseline vs SMOTE Accuracies (Combined)")
plt.legend()
plt.tight_layout()
plt.savefig("plots/baseline_vs_smote_combined.png")
plt.show()

# Baseline accuracy by split
splits = ['FER-2013','CK+','Combined']
plt.figure(figsize=(8,6))
for name in model_names:
    accs = [results[name][ds][0] for ds in splits]
    plt.plot(splits, accs, marker='o', label=name)
plt.xlabel("Test Split")
plt.ylabel("Accuracy")
plt.title("Baseline Accuracy by Split")
plt.legend(bbox_to_anchor=(1.02,1), loc='upper left', fontsize='small')
plt.tight_layout()
plt.savefig("plots/accuracy_by_split.png")
plt.show()

# Training vs Validation loss (zoomed)
# Gather min/max for tight y‐axis
all_losses = [l for name in model_names for seq in histories[name] for l in seq]
ymin, ymax = min(all_losses), max(all_losses)

plt.figure(figsize=(12, 7))
for name in model_names:
    train_l, val_l = histories[name]
    epochs = np.arange(1, len(train_l)+1)
    plt.plot(epochs, train_l, '--', alpha=0.7, label=f"{name} Train")
    plt.plot(epochs, val_l,  '-',  alpha=0.7, label=f"{name} Val")

plt.ylim(ymin*0.98, ymax*1.02)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss (Zoomed)")
plt.legend(fontsize='small', ncol=2, loc='upper center', bbox_to_anchor=(0.5,1.15))
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("plots/loss_curves.png")
plt.show()

# Baseline vs SMOTE correlation scatter (Combined)
plt.figure(figsize=(5,5))
plt.scatter(baseline_acc, smote_acc)
for xi, yi, name in zip(baseline_acc, smote_acc, model_names):
    plt.text(xi, yi, name, fontsize=8)
plt.plot([0,1],[0,1],'k--', alpha=0.5)
plt.xlabel("Baseline Accuracy")
plt.ylabel("SMOTE Accuracy")
plt.title("Baseline vs SMOTE Acc (Combined)")
plt.tight_layout()
plt.savefig("plots/acc_correlation.png")
plt.show()


























