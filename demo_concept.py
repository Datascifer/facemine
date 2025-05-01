```python
#!/usr/bin/env python3
"""
# demo_concept.py

Minimal benchmark comparing ResNet-18 vs MyCNNv6 on 48×48 grayscale data.

> **Proof-of-concept only** — no user interface included.  
> Future work: add a web frontend or Android app.

---

## Usage

```bash
python demo_concept.py --data_path /path/to/train

Repository:
    https://github.com/Datascifer/facemine/tree/main/demo_concept

Contact:
    mxg9xv@virginia.edu
"""

import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from your_model_file import MyCNNv6


def get_data_loader(path, batch_size=32, num_workers=4):
    transform = transforms.Compose([
        transforms.Grayscale(1),
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    dataset = datasets.ImageFolder(root=path, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)


def train_one_batch(model, loader, device, lr=1e-3):
    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    images, labels = next(iter(loader))
    images, labels = images.to(device), labels.to(device)
    optimizer.zero_grad()
    outputs = model(images)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    return loss.item()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def measure_inference_latency(model, device, input_size=(1, 48, 48), runs=100):
    model.eval()
    dummy = torch.randn(1, *input_size, device=device)
    for _ in range(10):
        _ = model(dummy)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    start = time.time()
    for _ in range(runs):
        _ = model(dummy)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    return (time.time() - start) / runs * 1000.0


def demo(models_dict, data_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = get_data_loader(data_path)
    for name, model in models_dict.items():
        model = model.to(device)
        loss = train_one_batch(model, loader, device)
        params = count_parameters(model)
        latency = measure_inference_latency(model, device)
        print(f"{name:8s}  loss {loss:.4f}  params {params:,}  latency {latency:.1f} ms")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ResNet18 vs MyCNNv6 demo")
    parser.add_argument("--data_path", required=True, help="Path to training data")
    args = parser.parse_args()

    models_to_demo = {
        "ResNet18": models.resnet18(pretrained=False, num_classes=8),
        "MyCNNv6":  MyCNNv6(num_classes=8),
    }

    demo(models_to_demo, data_path=args.data_path)
```

How to use:

1. Clone the repo:
   ```
   git clone https://github.com/Datascifer/facemine.git
   cd facemine/demo_concept
   ```
2. Install dependencies:
   ```
   pip install torch torchvision pytest
   ```
3. Prepare your data in ImageFolder format:
   ```
   /path/to/train/
       angry/
       happy/
       ...
   ```
4. Put your `MyCNNv6` class in `your_model_file.py`, ensuring it accepts `num_classes`.
5. Run the demo:
   ```
   python emotion_demo.py --data_path /path/to/train
   ```
6. (Optional) Run tests:
   ```
   pytest
   ```

This is an incomplete model—it needs a user interface to be usable. You may (or I may) build a web front end or an Android app in the future.
