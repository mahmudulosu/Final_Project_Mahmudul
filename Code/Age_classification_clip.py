# Dependencies: torch, torchvision, numpy, pandas, Pillow, nibabel, scipy, transformers, sklearn, seaborn, matplotlib
# Install via:
#   pip install transformers scikit-learn seaborn matplotlib

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
from PIL import Image
import nibabel as nib
from scipy.ndimage import zoom
from transformers import CLIPProcessor, CLIPModel
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

# ============================
# Config
# ============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

eeg_dir  = r"C:/research/EEG_Domain/scalograms_224x224x17"
mri_dir  = r"C:/research/MRI/structural_MRI"
csv_path = r"C:/research/MRI/participants_LSD_andLEMON.csv"
UNFREEZE_BLOCKS = 4

# ============================
# Load Age Labels
# ============================
data = pd.read_csv(csv_path)
valid_ages = ["20-25","25-30","60-65","65-70","70-75"]
age_group_mapping = {
    "20-25": 0,  # young
    "25-30": 0,
    "60-65": 1,  # old
    "65-70": 1,
    "70-75": 1
}
filtered = data[data["age"].isin(valid_ages)]
age_map = {
    row["participant_id"]: age_group_mapping[row["age"]]
    for _, row in filtered.iterrows()
}

# ============================
# CLIP Setup & Freeze/Unfreeze
# ============================
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)

for p in clip.parameters(): p.requires_grad = False
layers = clip.vision_model.encoder.layers
for block in layers[-UNFREEZE_BLOCKS:]:
    for p in block.parameters(): p.requires_grad = True
for name, p in clip.named_parameters():
    if "layer_norm" in name or "visual_projection" in name:
        p.requires_grad = True

embed_dim = clip.config.projection_dim

# ============================
# Dataset Classes
# ============================
class EEGDataset(Dataset):
    def __init__(self, npy_dir, labels, processor):
        self.items = []
        for fn in os.listdir(npy_dir):
            if not fn.endswith('.npy'): continue
            pid = fn.split('_')[0]
            if pid in labels:
                self.items.append((os.path.join(npy_dir,fn), labels[pid]))
        self.processor = processor

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]
        arr = np.load(path)
        if arr.shape[2] >= 3:
            img = arr[..., :3]
        else:
            m = arr.mean(axis=2)
            img = np.stack([m]*3, -1)
        img = ((img - img.min())/(img.max()-img.min())*255).astype(np.uint8)
        pix = self.processor(images=Image.fromarray(img), return_tensors='pt').pixel_values.squeeze(0)
        return pix, torch.tensor(label, dtype=torch.long)

class MRIDataset(Dataset):
    def __init__(self, nii_dir, labels, processor, target_shape=(128,128,128)):
        self.items = []
        for fn in os.listdir(nii_dir):
            if not fn.endswith(('.nii','.nii.gz')): continue
            pid = fn.split('_')[0]
            if pid in labels:
                self.items.append((os.path.join(nii_dir,fn), labels[pid]))
        self.processor, self.target_shape = processor, target_shape

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]
        vol = nib.load(path).get_fdata()
        vol = (vol - vol.min())/(vol.max()-vol.min()+1e-5)
        factors = [t/s for t,s in zip(self.target_shape, vol.shape)]
        vol = zoom(vol, factors, order=1)
        cx, cy, cz = [d//2 for d in vol.shape]
        rgb = np.stack([vol[cx,:,:], vol[:,cy,:], vol[:,:,cz]], -1)
        rgb = ((rgb - rgb.min())/(rgb.max()-rgb.min())*255).astype(np.uint8)
        pix = self.processor(images=Image.fromarray(rgb), return_tensors='pt').pixel_values.squeeze(0)
        return pix, torch.tensor(label, dtype=torch.long)

class FusionDataset(Dataset):
    def __init__(self, eeg_ds, mri_ds):
        self.eeg_map = {os.path.basename(p).split('_')[0]: i for i,(p,_) in enumerate(eeg_ds.items)}
        self.mri_map = {os.path.basename(p).split('_')[0]: i for i,(p,_) in enumerate(mri_ds.items)}
        self.eeg_ds, self.mri_ds = eeg_ds, mri_ds
        self.ids = list(set(self.eeg_map).intersection(self.mri_map))

    def __len__(self): return len(self.ids)

    def __getitem__(self, idx):
        pid = self.ids[idx]
        pix_e, lbl = self.eeg_ds[self.eeg_map[pid]]
        pix_m, _   = self.mri_ds[self.mri_map[pid]]
        return pix_e, pix_m, lbl

# ============================
# Heads
# ============================
class SimpleHead(nn.Module):
    def __init__(self,in_dim,hidden=256,num_classes=2):
        super().__init__()
        self.mlp=nn.Sequential(
            nn.Linear(in_dim,hidden), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(hidden,num_classes)
        )
    def forward(self,x): return self.mlp(x)

eeg_head = SimpleHead(embed_dim).to(device)
mri_head = SimpleHead(embed_dim).to(device)
fus_head = SimpleHead(embed_dim*2).to(device)

# ============================
# Loader utils
# ============================
def make_loaders(ds, batch_size):
    labels = [int(ds[i][-1]) for i in range(len(ds))]
    weights = 1.0/np.bincount(labels)
    sample_weights = [weights[l] for l in labels]
    n = len(ds); t = int(0.8*n); v = n-t
    train_ds, val_ds = random_split(ds, [t,v])
    train_w = [sample_weights[i] for i in train_ds.indices]
    sampler = WeightedRandomSampler(train_w, len(train_w), True)
    return (DataLoader(train_ds, batch_size=batch_size, sampler=sampler),
            DataLoader(val_ds, batch_size=batch_size, shuffle=False))

# Prepare datasets & loaders
eeg_ds = EEGDataset(eeg_dir, age_map, processor)
mri_ds = MRIDataset(mri_dir, age_map, processor)
fus_ds = FusionDataset(eeg_ds, mri_ds)

eeg_train, eeg_val = make_loaders(eeg_ds, 16)
mri_train, mri_val = make_loaders(mri_ds, 8)
fus_train, fus_val = make_loaders(fus_ds, 8)

# ============================
# Training & Evaluation
# ============================
def train_and_report(feat_model, head, train_loader, val_loader, name, epochs=20, lr=2e-5):
    params = [p for p in feat_model.parameters() if p.requires_grad] + list(head.parameters())
    opt = optim.AdamW(params, lr=lr, weight_decay=1e-4)
    sched = CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss()

    for ep in range(1, epochs+1):
        feat_model.train(); head.train()
        tl=tc=tn=0
        for batch in tqdm(train_loader, desc=f"{name} Train EP{ep}"):
            opt.zero_grad()
            if len(batch)==3:
                e,m,l = batch
                emb_e = feat_model.get_image_features(pixel_values=e.to(device)).float()
                emb_m = feat_model.get_image_features(pixel_values=m.to(device)).float()
                inp = torch.cat([emb_e,emb_m],1)
            else:
                px,l = batch
                inp = feat_model.get_image_features(pixel_values=px.to(device)).float()
            logits = head(inp); loss = crit(logits, l.to(device))
            loss.backward(); opt.step()
            preds = logits.argmax(1)
            tl += loss.item()*l.size(0)
            tc += (preds==l.to(device)).sum().item()
            tn += l.size(0)
        sched.step()

        # validation
        y_true=[]; y_pred=[]
        feat_model.eval(); head.eval()
        with torch.no_grad():
            for batch in val_loader:
                if len(batch)==3:
                    e,m,l = batch
                    emb_e = feat_model.get_image_features(pixel_values=e.to(device)).float()
                    emb_m = feat_model.get_image_features(pixel_values=m.to(device)).float()
                    inp = torch.cat([emb_e,emb_m],1)
                else:
                    px,l = batch
                    inp = feat_model.get_image_features(pixel_values=px.to(device)).float()
                logits = head(inp)
                preds = logits.argmax(1)
                y_true.extend(l.numpy()); y_pred.extend(preds.cpu().numpy())

        train_acc = tc/tn
        val_acc = np.mean(np.array(y_true)==np.array(y_pred))
        print(f"{name} EP{ep}: loss={tl/tn:.4f} train_acc={train_acc:.4f} val_acc={val_acc:.4f}")

    # final report
    cm = confusion_matrix(y_true, y_pred)
    print(f"\n{name} Classification Report:")
    print(classification_report(y_true, y_pred, target_names=['young','old']))
    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt='d',
                xticklabels=['young','old'],
                yticklabels=['young','old'])
    plt.title(f"{name} Confusion Matrix")
    plt.show()

# ============================
# Execute
# ============================
print("\n>> EEG-only Age Classifier")
train_and_report(clip, eeg_head, eeg_train, eeg_val, 'EEG Age')

print("\n>> MRI-only Age Classifier")
train_and_report(clip, mri_head, mri_train, mri_val, 'MRI Age')

print("\n>> Fusion Age Classifier")
train_and_report(clip, fus_head, fus_train, fus_val, 'Fusion Age')

print("All done!")
