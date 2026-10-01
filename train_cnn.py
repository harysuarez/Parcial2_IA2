# train_cnn.py
import json, os, random
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import models, transforms
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score
from PIL import Image

IMG = 224
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

def load_data(json_path="data/shipsnet.json"):
    with open(json_path) as f:
        d = json.load(f)
    raw = np.array(d["data"], dtype=np.uint8)
    images = raw.reshape(-1, 3, 80, 80).transpose(0, 2, 3, 1)   # N,80,80,3
    labels = np.array(d["labels"], dtype=np.int64)
    if "scene_ids" in d:
        groups = np.array(d["scene_ids"])
    else:
        print("⚠️ No hay scene_ids: el split tendrá leakage")
        groups = np.arange(len(labels))
    return images, labels, groups

class DS(Dataset):
    def __init__(self, X, y, tf): self.X, self.y, self.tf = X, y, tf
    def __len__(self): return len(self.X)
    def __getitem__(self, i):
        return self.tf(Image.fromarray(self.X[i])), int(self.y[i])

def rot90_random(im):
    return im.rotate(90 * random.randint(0, 3))

train_tf = transforms.Compose([
    transforms.Resize((IMG, IMG)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.Lambda(rot90_random),
    transforms.RandomResizedCrop(IMG, scale=(0.6, 1.0), ratio=(0.9, 1.1)),  # simula otras escalas
    transforms.ColorJitter(0.4, 0.4, 0.4, 0.05),
    transforms.RandomGrayscale(0.2),
    transforms.RandomApply([transforms.GaussianBlur(5, (0.1, 2.0))], p=0.3),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])
val_tf = transforms.Compose([
    transforms.Resize((IMG, IMG)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

def build_model():
    m = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    m.classifier = nn.Sequential(
        nn.Dropout(0.3), nn.Linear(m.last_channel, 64), nn.ReLU(), nn.Linear(64, 2)
    )
    return m

def evaluate(model, loader):
    model.eval(); P, T = [], []
    with torch.no_grad():
        for x, y in loader:
            P += model(x).argmax(1).tolist(); T += y.tolist()
    return accuracy_score(T, P), precision_score(T, P, zero_division=0), recall_score(T, P, zero_division=0)

def main():
    X, y, g = load_data()
    # Split POR ESCENA (sin leakage)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    tr, va = next(gss.split(X, y, g))
    print(f"Train: {len(tr)} | Val: {len(va)}")

    counts = np.bincount(y[tr])
    w = (1.0 / counts)[y[tr]]
    sampler = WeightedRandomSampler(w, len(w))
    train_loader = DataLoader(DS(X[tr], y[tr], train_tf), batch_size=32, sampler=sampler)
    val_loader = DataLoader(DS(X[va], y[va], val_tf), batch_size=64)

    model = build_model()
    EPOCHS = 8
    opt = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    crit = nn.CrossEntropyLoss(label_smoothing=0.05)

    best = 0
    os.makedirs("models", exist_ok=True)
    for ep in range(1, EPOCHS + 1):
        model.train(); loss_sum = n = 0
        for xb, yb in train_loader:
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward(); opt.step()
            loss_sum += loss.item() * len(xb); n += len(xb)
        sched.step()
        acc, prec, rec = evaluate(model, val_loader)
        print(f"Época {ep}/{EPOCHS} | loss {loss_sum/n:.4f} | VAL acc {acc*100:.2f}% prec {prec*100:.1f}% rec {rec*100:.1f}%")
        if acc > best:
            best = acc
            torch.save(model.state_dict(), "models/mobilenet_uav.pth")
    print(f"✅ Mejor accuracy de validación (por escenas): {best*100:.2f}%")

if __name__ == "__main__":
    main()