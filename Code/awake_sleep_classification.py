import os
import random
import numpy as np
import pandas as pd
import pywt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# ============================
# Configuration
# ============================
data_dir         = r"C:\research\EEG_Domain\combined\combined_new"
fatigue_csv_path = r"C:\codes\pytorchRun\Deep_learning_project\MDBF_Day1.csv"
sampling_rate    = 250
segment_secs     = 5
segment_length   = segment_secs * sampling_rate
percentile_sel   = 60
batch_size       = 32
num_epochs       = 100
learning_rate    = 1e-3
weight_decay     = 1e-4
device           = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================
# Load and preprocess labels
# ============================
labels_df = pd.read_csv(fatigue_csv_path)
labels_df['MDBF_Day1_WM_Scale'] = pd.to_numeric(labels_df['MDBF_Day1_WM_Scale'], errors='coerce')
labels_df = labels_df[(labels_df['MDBF_Day1_WM_Scale'] <= 30) | (labels_df['MDBF_Day1_WM_Scale'] >= 35)]
labels_df['label'] = (labels_df['MDBF_Day1_WM_Scale'] <= 28).astype(int)
fatigue_mapping = { str(r['ID']): int(r['label']) for _, r in labels_df.iterrows() }

# ============================
# DWT Feature Extraction
# ============================
def load_eeg_segments(file_path):
    df = pd.read_csv(file_path)
    return [df.iloc[i:i+segment_length] 
            for i in range(0, len(df), segment_length) 
            if len(df.iloc[i:i+segment_length]) == segment_length]

def extract_dwt(segment, wavelet='db4', level=6):
    feats = []
    for ch in segment.columns:
        coeffs = pywt.wavedec(segment[ch].values, wavelet, level=level)
        for c in coeffs:
            feats += [ np.std(c), np.mean(c), np.sqrt(np.mean(c**2)) ]
    return np.array(feats, dtype=np.float32)

# ============================
# Gather data
# ============================
all_feats, all_labels, all_sids = [], [], []
for fname in os.listdir(data_dir):
    if not fname.lower().endswith(".csv"):
        continue
    sid = fname.split('_')[0]
    if sid not in fatigue_mapping:
        continue
    lab = fatigue_mapping[sid]
    for seg in load_eeg_segments(os.path.join(data_dir, fname)):
        all_feats.append(extract_dwt(seg))
        all_labels.append(lab)
        all_sids.append(sid)

all_feats  = np.vstack(all_feats)
all_labels = np.array(all_labels)
all_sids   = np.array(all_sids)

# ============================
# Subject-level split
# ============================
unique_sids = np.unique(all_sids)
train_sids, temp_sids = train_test_split(unique_sids, test_size=0.3, random_state=42)
val_sids, test_sids   = train_test_split(temp_sids, test_size=0.5, random_state=42)

def mask(sids, subset): return np.isin(sids, subset)
X_train = all_feats[mask(all_sids, train_sids)]
y_train = all_labels[mask(all_sids, train_sids)]
X_val   = all_feats[mask(all_sids, val_sids)]
y_val   = all_labels[mask(all_sids, val_sids)]
X_test  = all_feats[mask(all_sids, test_sids)]
y_test  = all_labels[mask(all_sids, test_sids)]

# ============================
# Feature selection & scaling
# ============================
selector = SelectPercentile(f_classif, percentile=percentile_sel)
X_train = selector.fit_transform(X_train, y_train)
X_val   = selector.transform(X_val)
X_test  = selector.transform(X_test)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val   = scaler.transform(X_val)
X_test  = scaler.transform(X_test)

# ============================
# Dataset & DataLoader
# ============================
class DWTDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y).long()
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# sampler to balance classes
counts = np.bincount(y_train)
weights = [1.0/counts[c] for c in y_train]
sampler = WeightedRandomSampler(weights, len(weights), replacement=True)

train_loader = DataLoader(DWTDataset(X_train,y_train), batch_size=batch_size, sampler=sampler)
val_loader   = DataLoader(DWTDataset(X_val,  y_val),   batch_size=batch_size, shuffle=False)
test_loader  = DataLoader(DWTDataset(X_test, y_test),  batch_size=batch_size, shuffle=False)

# ============================
# Improved Model Architecture
# ============================
class DWTNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),

            nn.Linear(64, 2)
        )
    def forward(self, x):
        return self.net(x)

model = DWTNet(input_dim=X_train.shape[1]).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)

# ============================
# Training Loop (no early stopping)
# ============================
for epoch in range(1, num_epochs+1):
    model.train()
    for Xb, yb in train_loader:
        Xb, yb = Xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(Xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

    # validation
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for Xb, yb in val_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            preds = model(Xb).argmax(dim=1)
            correct += (preds == yb).sum().item()
            total   += yb.size(0)
    val_acc = correct/total
    scheduler.step(val_acc)

    print(f"Epoch {epoch}/{num_epochs} — Val Acc: {val_acc:.4f} — LR: {optimizer.param_groups[0]['lr']:.5f}")

# ============================
# Test Evaluation
# ============================
model.eval()
correct, total = 0, 0
with torch.no_grad():
    for Xb, yb in test_loader:
        Xb, yb = Xb.to(device), yb.to(device)
        preds = model(Xb).argmax(dim=1)
        correct += (preds == yb).sum().item()
        total   += yb.size(0)
test_acc = correct/total
print(f"Test Accuracy: {test_acc:.4f}")
