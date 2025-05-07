import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
from torchvision import models, transforms
from PIL import Image
import nibabel as nib
from scipy.ndimage import zoom
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# ============================
# Set seed for reproducibility
# ============================
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
set_seed()

# ============================
# Config and device
# ============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ============================
# Paths and age mapping
# ============================
eo_dir   = r"C:\research\EEG_Domain\eye_openscalo"
ec_dir   = r"C:\research\EEG_Domain\eyeclose_scalo"
mri_dir  = r"C:\research\commonMRI"
csv_path = r"C:/research/MRI/participants_LSD_andLEMON.csv"

# Which ResNet stages to unfreeze
UNFREEZE_LAYERS = ["layer3", "layer4"]

# Load and filter age groups
age_df = pd.read_csv(csv_path)
valid_ages = ["20-25", "25-30", "60-65", "65-70", "70-75"]
# Map the five buckets into two classes: 0 = young, 1 = old
age_group_mapping = {
    "20-25": 0, "25-30": 0,
    "60-65": 1, "65-70": 1, "70-75": 1
}
filtered = age_df[age_df['age'].isin(valid_ages)]
age_map = {
    row['participant_id']: age_group_mapping[row['age']]
    for _, row in filtered.iterrows()
}

# ============================
# Dataset definitions
# ============================
class EODataset(Dataset):
    def __init__(self, npy_dir, labels):
        self.samples = []
        for fn in os.listdir(npy_dir):
            if not fn.endswith('.npy'): continue
            pid = fn.split('_')[0]
            if pid not in labels: continue
            self.samples.append((os.path.join(npy_dir, fn), labels[pid]))
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        arr = np.load(path)
        arr = (arr - arr.min())/(arr.max()-arr.min()+1e-5)
        pix = torch.tensor(arr, dtype=torch.float32).permute(2,0,1)
        return pix, torch.tensor(label, dtype=torch.long)

class ECDataset(EODataset):
    pass

class MRIDataset(Dataset):
    def __init__(self, nii_dir, labels):
        self.samples = []
        for fn in os.listdir(nii_dir):
            if not fn.endswith(('.nii','.nii.gz')): continue
            pid = fn.split('_')[0]
            if pid not in labels: continue
            self.samples.append((os.path.join(nii_dir, fn), labels[pid]))
        self.transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
        ])
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        vol = nib.load(path).get_fdata()
        vol = (vol - vol.min())/(vol.max()-vol.min()+1e-5)
        vol = zoom(vol, [128/s for s in vol.shape], order=1)
        cx, cy, cz = [d//2 for d in vol.shape]
        slices = [vol[cx,:,:], vol[:,cy,:], vol[:,:,cz]]
        rgb = np.stack(slices, axis=-1).astype(np.float32)
        rgb = (rgb - rgb.min())/(rgb.max()-rgb.min()+1e-5)
        pil = Image.fromarray((rgb*255).astype(np.uint8))
        pix = self.transform(pil)
        return pix, torch.tensor(label, dtype=torch.long)

class Fusion3Dataset(Dataset):
    def __init__(self, eo_ds, ec_ds, mri_ds):
        self.eo_map  = {os.path.basename(p).split('_')[0]: i for i,(p,_) in enumerate(eo_ds.samples)}
        self.ec_map  = {os.path.basename(p).split('_')[0]: i for i,(p,_) in enumerate(ec_ds.samples)}
        self.mri_map = {os.path.basename(p).split('_')[0]: i for i,(p,_) in enumerate(mri_ds.samples)}
        self.eo_ds, self.ec_ds, self.mri_ds = eo_ds, ec_ds, mri_ds
        # only keep IDs present in all three modalities
        self.ids = list(set(self.eo_map) & set(self.ec_map) & set(self.mri_map))
    def __len__(self): return len(self.ids)
    def __getitem__(self, idx):
        pid = self.ids[idx]
        eo, lbl = self.eo_ds[self.eo_map[pid]]
        ec, _   = self.ec_ds[self.ec_map[pid]]
        mr, _   = self.mri_ds[self.mri_map[pid]]
        return eo, ec, mr, lbl

# ============================
# DataLoaders with balanced sampling
# ============================
def make_loader(ds, batch_size):
    labels = [int(ds[i][-1]) for i in range(len(ds))]
    weights = 1.0 / np.bincount(labels)
    sampler_weights = [weights[l] for l in labels]
    train_size = int(0.8 * len(ds))
    val_size   = len(ds) - train_size
    train_ds, val_ds = random_split(ds, [train_size, val_size])
    sw = [sampler_weights[i] for i in train_ds.indices]
    sampler = WeightedRandomSampler(sw, len(sw), replacement=True)
    return (
      DataLoader(train_ds, batch_size=batch_size, sampler=sampler),
      DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
    )

# ============================
# Simple classification head
# ============================
class SimpleHead(nn.Module):
    def __init__(self, in_dim, hidden=256, num_classes=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden, num_classes)
        )
    def forward(self, x): return self.net(x)

# ============================
# Training / evaluation loops
# ============================
def train_single(backbone, head, loader, epochs=10, lr=2e-4, wd=1e-4):
    params = [p for p in backbone.parameters() if p.requires_grad] + list(head.parameters())
    opt = optim.AdamW(params, lr=lr, weight_decay=wd)
    crit = nn.CrossEntropyLoss()
    for epoch in range(1, epochs+1):
        backbone.train(); head.train()
        total_loss = correct = total = 0
        for x, y in loader[0]:
            opt.zero_grad()
            feats = backbone(x.to(device))
            out   = head(feats)
            loss  = crit(out, y.to(device))
            loss.backward(); opt.step()
            preds = out.argmax(dim=1)
            total_loss += loss.item() * y.size(0)
            correct    += (preds == y.to(device)).sum().item()
            total      += y.size(0)
        print(f"Epoch {epoch}/{epochs} "
              f"- Loss: {total_loss/total:.4f} "
              f"- Acc: {correct/total:.4f}")

def eval_single(backbone, head, loader):
    backbone.eval(); head.eval()
    all_feats, all_preds, all_labels = [], [], []
    with torch.no_grad():
        for x, y in loader[1]:
            f = backbone(x.to(device))
            all_feats.append(f.cpu().numpy())
            out = head(f)
            all_preds.extend(out.argmax(1).cpu().numpy())
            all_labels.extend(y.numpy())
    feats = np.concatenate(all_feats, axis=0)
    cm    = confusion_matrix(all_labels, all_preds)
    cr    = classification_report(all_labels, all_preds, target_names=['young','old'])
    return feats, cm, cr, np.array(all_labels)

def train_fusion(back_eeg, back_mri, head, loader, epochs=10, lr=2e-4, wd=1e-4):
    params = (
      [p for p in back_eeg.parameters() if p.requires_grad] +
      [p for p in back_mri.parameters() if p.requires_grad] +
      list(head.parameters())
    )
    opt = optim.AdamW(params, lr=lr, weight_decay=wd)
    crit = nn.CrossEntropyLoss()
    for epoch in range(1, epochs+1):
        back_eeg.train(); back_mri.train(); head.train()
        total_loss = correct = total = 0
        for eo, ec, mr, y in loader[0]:
            opt.zero_grad()
            f_eo = back_eeg(eo.to(device))
            f_ec = back_eeg(ec.to(device))
            f_mr = back_mri(mr.to(device))
            feats = torch.cat([f_eo, f_ec, f_mr], dim=1)
            out   = head(feats)
            loss  = crit(out, y.to(device))
            loss.backward(); opt.step()
            preds = out.argmax(1)
            total_loss += loss.item() * y.size(0)
            correct    += (preds == y.to(device)).sum().item()
            total      += y.size(0)
        print(f"Epoch {epoch}/{epochs} "
              f"- Loss: {total_loss/total:.4f} "
              f"- Acc: {correct/total:.4f}")

def eval_fusion(back_eeg, back_mri, head, loader):
    back_eeg.eval(); back_mri.eval(); head.eval()
    all_feats, all_preds, all_labels = [], [], []
    with torch.no_grad():
        for eo, ec, mr, y in loader[1]:
            f_eo = back_eeg(eo.to(device))
            f_ec = back_eeg(ec.to(device))
            f_mr = back_mri(mr.to(device))
            feats = torch.cat([f_eo, f_ec, f_mr], dim=1)
            all_feats.append(feats.cpu().numpy())
            out = head(feats)
            all_preds.extend(out.argmax(1).cpu().numpy())
            all_labels.extend(y.numpy())
    feats = np.concatenate(all_feats, axis=0)
    cm    = confusion_matrix(all_labels, all_preds)
    cr    = classification_report(all_labels, all_preds, target_names=['young','old'])
    return feats, cm, cr, np.array(all_labels)

# ============================
# Build datasets & loaders
# ============================
eo_ds   = EODataset(eo_dir, age_map)
ec_ds   = ECDataset(ec_dir, age_map)
mri_ds  = MRIDataset(mri_dir, age_map)
fus3_ds = Fusion3Dataset(eo_ds, ec_ds, mri_ds)

eo_loader   = make_loader(eo_ds, batch_size=16)
ec_loader   = make_loader(ec_ds, batch_size=16)
mri_loader  = make_loader(mri_ds, batch_size=8)
fus3_loader = make_loader(fus3_ds, batch_size=8)

# ============================
# Main experiment loop
# ============================
for variant in ['resnet50', 'resnet101', 'resnet152']:
    print(f"\n=== {variant} ===")
    # -------- EEG Backbone --------
    base_eeg = getattr(models, variant)(pretrained=True)
    # adapt first conv to 17-channel EEG input
    old_conv = base_eeg.conv1
    new_conv = nn.Conv2d(17, old_conv.out_channels,
                         old_conv.kernel_size,
                         old_conv.stride,
                         old_conv.padding,
                         bias=False)
    with torch.no_grad():
        # average pretrained weights over RGB -> copy to all 17 inputs
        mean_w = old_conv.weight.mean(dim=1, keepdim=True)
        new_conv.weight.copy_(mean_w.repeat(1,17,1,1))
    base_eeg.conv1 = new_conv
    # replace classifier with identity
    feat_dim = base_eeg.fc.in_features
    base_eeg.fc = nn.Identity()
    # freeze all but specified layers + batchnorm
    for name, p in base_eeg.named_parameters():
        p.requires_grad = any(name.startswith(l) for l in UNFREEZE_LAYERS + ["bn1","bn2"])
    back_eeg = base_eeg.to(device)
    head_eo  = SimpleHead(feat_dim, num_classes=2).to(device)

    # -------- MRI Backbone --------
    base_mri = getattr(models, variant)(pretrained=True)
    feat_mri_dim = base_mri.fc.in_features
    base_mri.fc = nn.Identity()
    for name, p in base_mri.named_parameters():
        p.requires_grad = any(name.startswith(l) for l in UNFREEZE_LAYERS + ["bn1","bn2"])
    back_mri = base_mri.to(device)
    head_mri = SimpleHead(feat_mri_dim, num_classes=2).to(device)

    # -------- Fusion Head --------
    head_fus = SimpleHead(feat_dim*3, num_classes=2).to(device)

    # --- Train & Evaluate EO-only ---
    print("\n>> EO-only Training")
    train_single(back_eeg, head_eo, eo_loader, epochs=10)
    feats_tr, cm_tr, cr_tr, lbls_tr = eval_single(back_eeg, head_eo, eo_loader)
    tsne_tr = TSNE(n_components=2, random_state=42).fit_transform(feats_tr)
    plt.figure(figsize=(6,5))
    plt.scatter(tsne_tr[:,0], tsne_tr[:,1], c=lbls_tr, alpha=0.7)
    plt.title(f'{variant} EO Training TSNE'); plt.show()
    feats, cm, cr, lbls = eval_single(back_eeg, head_eo, eo_loader)
    print("Confusion Matrix:\n", cm)
    print("Classification Report:\n", cr)

    # --- Train & Evaluate MRI-only ---
    print("\n>> MRI-only Training")
    train_single(back_mri, head_mri, mri_loader, epochs=10)
    feats_tr, cm_tr, cr_tr, lbls_tr = eval_single(back_mri, head_mri, mri_loader)
    tsne_tr = TSNE(n_components=2, random_state=42).fit_transform(feats_tr)
    plt.figure(figsize=(6,5))
    plt.scatter(tsne_tr[:,0], tsne_tr[:,1], c=lbls_tr, alpha=0.7)
    plt.title(f'{variant} MRI Training TSNE'); plt.show()
    feats, cm, cr, lbls = eval_single(back_mri, head_mri, mri_loader)
    print("Confusion Matrix:\n", cm)
    print("Classification Report:\n", cr)

    # --- Train & Evaluate Fusion (EO+EC+MRI) ---
    print("\n>> Fusion EO+EC+MRI Training")
    train_fusion(back_eeg, back_mri, head_fus, fus3_loader, epochs=10)
    feats_tr, cm_tr, cr_tr, lbls_tr = eval_fusion(back_eeg, back_mri, head_fus, fus3_loader)
    tsne_tr = TSNE(n_components=2, random_state=42).fit_transform(feats_tr)
    plt.figure(figsize=(6,5))
    plt.scatter(tsne_tr[:,0], tsne_tr[:,1], c=lbls_tr, alpha=0.7)
    plt.title(f'{variant} Fusion Training TSNE'); plt.show()
    feats, cm, cr, lbls = eval_fusion(back_eeg, back_mri, head_fus, fus3_loader)
    print("Confusion Matrix:\n", cm)
    print("Classification Report:\n", cr)

print("All age‐group experiments complete.")
