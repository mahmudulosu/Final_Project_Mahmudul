import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import clip                                  # pip install git+https://github.com/openai/CLIP.git
from tqdm import tqdm
from torch.utils.data import DataLoader, WeightedRandomSampler, Dataset
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# ============================
# Config
# ============================
scalogram_dir      = r"C:\research\EEG_Domain\scalograms_format"
fatigue_csv_path   = r"C:\codes\pytorchRun\Deep_learning_project\MDBF_Day1.csv"
batch_size         = 32
num_epochs         = 50
learning_rate      = 1e-4
device             = torch.device("cuda" if torch.cuda.is_available() else "cpu")
UNFREEZE_LAYERS    = ["transformer.resblocks.10", "transformer.resblocks.11"]  # last two blocks

# ============================
# Load Fatigue Labels
# ============================
labels_df = pd.read_csv(fatigue_csv_path)
labels_df['MDBF_Day1_WM_Scale'] = pd.to_numeric(labels_df['MDBF_Day1_WM_Scale'], errors='coerce')
labels_df = labels_df[(labels_df['MDBF_Day1_WM_Scale'] <= 30) | (labels_df['MDBF_Day1_WM_Scale'] >= 35)]
labels_df['label'] = (labels_df['MDBF_Day1_WM_Scale'] <= 28).astype(int)
fatigue_mapping = {str(r['ID']): r['label'] for _, r in labels_df.iterrows()}

# ============================
# Split participants
# ============================
all_files = [f for f in os.listdir(scalogram_dir) if f.endswith('.npy')
             and f.split('_')[0] in fatigue_mapping]
pids = list({f.split('_')[0] for f in all_files})
random.shuffle(pids)
n = len(pids)
train_pids = set(pids[:int(0.7*n)])
val_pids   = set(pids[int(0.7*n):int(0.85*n)])
test_pids  = set(pids[int(0.85*n):])

train_files = [f for f in all_files if f.split('_')[0] in train_pids]
val_files   = [f for f in all_files if f.split('_')[0] in val_pids]
test_files  = [f for f in all_files if f.split('_')[0] in test_pids]

# ============================
# Dataset
# ============================
class FatigueEEGDataset(Dataset):
    def __init__(self, files, mapping, root):
        self.files = files
        self.mapping = mapping
        self.root = root
    def __len__(self):
        return len(self.files)
    def __getitem__(self, idx):
        fn = self.files[idx]
        arr = np.load(os.path.join(self.root, fn))           # (224,224,17)
        x   = torch.tensor(arr, dtype=torch.float32).permute(2,0,1)
        y   = self.mapping[fn.split('_')[0]]
        return x, y

train_ds = FatigueEEGDataset(train_files, fatigue_mapping, scalogram_dir)
val_ds   = FatigueEEGDataset(val_files,   fatigue_mapping, scalogram_dir)
test_ds  = FatigueEEGDataset(test_files,  fatigue_mapping, scalogram_dir)

# compute weights for sampler & loss
train_labels = [y for _,y in train_ds]
counts       = np.bincount(train_labels)
weights_cls  = torch.tensor((counts.max()/counts).astype(float), device=device)
samp_w       = [weights_cls[y].item() for y in train_labels]
sampler      = WeightedRandomSampler(samp_w, len(samp_w), replacement=True)

train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler)
val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

# ============================
# CLIP‐based model
# ============================
class CLIPFatigue(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.clip_model, _ = clip.load("ViT-B/32", device=device, jit=False)
        # adapt patch embedding conv1 to 17 channels
        old = self.clip_model.visual.conv1
        self.clip_model.visual.conv1 = nn.Conv2d(
            17, old.out_channels, old.kernel_size, old.stride, old.padding, bias=False
        )
        with torch.no_grad():
            mean_w = old.weight.mean(dim=1, keepdim=True)
            self.clip_model.visual.conv1.weight.copy_(mean_w.repeat(1,17,1,1))
        # freeze all then unfreeze last blocks
        for name, p in self.clip_model.visual.named_parameters():
            p.requires_grad = False
            if any(name.startswith(l) for l in UNFREEZE_LAYERS):
                p.requires_grad = True
        # classification head
        embed_dim = self.clip_model.visual.output_dim
        self.head = nn.Linear(embed_dim, 2)

    def forward(self, x):
        # x: [B,17,224,224] → CLIP vision encoder → [B, embed_dim]
        v = self.clip_model.visual(x)  
        return self.head(v)

model     = CLIPFatigue(device).to(device)
criterion = nn.CrossEntropyLoss(weight=weights_cls)
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)

# ============================
# Train & validate
# ============================
train_losses, train_accs = [], []
val_losses, val_accs, val_epochs = [], [], []

for epoch in range(1, num_epochs+1):
    # train
    model.train()
    running_loss = correct = total = 0
    for x,y in tqdm(train_loader, desc=f"Train E{epoch}/{num_epochs}"):
        x,y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss   = criterion(logits, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * y.size(0)
        preds = logits.argmax(1)
        correct += (preds==y).sum().item()
        total   += y.size(0)
    train_losses.append(running_loss/total)
    train_accs.append(correct/total)

    # validate every 10 epochs
    if epoch % 10 == 0:
        model.eval()
        val_loss = v_correct = v_total = 0
        with torch.no_grad():
            for x,y in val_loader:
                x,y = x.to(device), y.to(device)
                out = model(x)
                l   = criterion(out,y)
                val_loss += l.item()*y.size(0)
                preds = out.argmax(1)
                v_correct += (preds==y).sum().item()
                v_total   += y.size(0)
        val_losses.append(val_loss/v_total)
        val_accs.append(v_correct/v_total)
        val_epochs.append(epoch)
        print(f"→ Val @E{epoch}: loss={val_loss/v_total:.4f}, acc={v_correct/v_total:.4f}")

# ============================
# Test evaluation
# ============================
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for x,y in test_loader:
        x,y = x.to(device), y.to(device)
        out = model(x)
        all_preds.extend(out.argmax(1).cpu().tolist())
        all_labels.extend(y.cpu().tolist())

print("\nTest Report:")
print(classification_report(all_labels, all_preds, target_names=["Awake","Tired"]))
cm = confusion_matrix(all_labels, all_preds)
sns.heatmap(cm, annot=True, fmt="d", xticklabels=["Awake","Tired"],
            yticklabels=["Awake","Tired"])
plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title("Confusion Matrix")
plt.show()
