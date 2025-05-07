# Dependencies: torch, torchvision, numpy, pandas, Pillow, nibabel, scipy, transformers, scikit-learn, seaborn, matplotlib
# Install via:
#   pip install torch torchvision transformers scikit-learn seaborn matplotlib nibabel scipy

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
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

# ============================
# Config
# ============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

eeg_dir   = r"C:/research/EEG_Domain/scalograms_224x224x17"
mri_dir   = r"C:/research/MRI/structural_MRI"
csv_path  = r"C:/research/MRI/participants_LSD_andLEMON.csv"
UNFREEZE_BLOCKS = 4
BATCH_SIZES = {"EEG":16, "MRI":8, "Fusion":8}
EPOCHS   = 20
LR       = 2e-5
WD       = 1e-4

# ============================
# Load gender labels
# ============================
df = pd.read_csv(csv_path)
gender_map = {
    row["participant_id"]: 0 if row["gender"]=="M" else 1
    for _, row in df.iterrows()
}

# ============================
# CLIP setup for EEG + MRI
# ============================
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# EEG branch: expand to 17-channel patch embedding
clip_eeg = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
vcfg = clip_eeg.config.vision_config
old_pe = clip_eeg.vision_model.embeddings.patch_embedding
new_pe = nn.Conv2d(17, old_pe.out_channels, kernel_size=vcfg.patch_size, stride=vcfg.patch_size, bias=False)
with torch.no_grad():
    avg_w = old_pe.weight.mean(dim=1, keepdim=True)
    new_pe.weight.copy_(avg_w.repeat(1,17,1,1))
new_pe = new_pe.to(device)
clip_eeg.vision_model.embeddings.patch_embedding = new_pe

# MRI branch: standard CLIP
clip_mri = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)

# Freeze all, then unfreeze last UNFREEZE_BLOCKS + projection head
def freeze_unfreeze(model):
    for p in model.parameters(): p.requires_grad=False
    for blk in model.vision_model.encoder.layers[-UNFREEZE_BLOCKS:]:
        for p in blk.parameters(): p.requires_grad=True
    for name,p in model.named_parameters():
        if "layer_norm" in name or "visual_projection" in name:
            p.requires_grad=True

freeze_unfreeze(clip_eeg)
freeze_unfreeze(clip_mri)

embed_dim = clip_eeg.config.projection_dim

# ============================
# Dataset classes
# ============================
class EEGDataset(Dataset):
    def __init__(self, npy_dir, labels):
        self.items = []
        for fn in os.listdir(npy_dir):
            if not fn.endswith(".npy"): continue
            pid = fn.split("_")[0]
            if pid in labels:
                self.items.append((os.path.join(npy_dir,fn), labels[pid]))
    def __len__(self): return len(self.items)
    def __getitem__(self, idx):
        path,label = self.items[idx]
        arr = np.load(path)
        arr = (arr - arr.min())/(arr.max()-arr.min()+1e-5)
        pix = torch.tensor(arr, dtype=torch.float32).permute(2,0,1)  # (17,224,224)
        return pix, torch.tensor(label)

class MRIDataset(Dataset):
    def __init__(self, nii_dir, labels, processor, target=(128,128,128)):
        self.items = []
        for fn in os.listdir(nii_dir):
            if not fn.endswith((".nii",".nii.gz")): continue
            pid = fn.split("_")[0]
            if pid in labels:
                self.items.append((os.path.join(nii_dir,fn), labels[pid]))
        self.processor = processor
        self.target = target
    def __len__(self): return len(self.items)
    def __getitem__(self, idx):
        path,label = self.items[idx]
        vol = nib.load(path).get_fdata()
        vol = (vol-vol.min())/(vol.max()-vol.min()+1e-5)
        factors = [t/s for t,s in zip(self.target, vol.shape)]
        vol = zoom(vol, factors, order=1)
        cx,cy,cz = [d//2 for d in vol.shape]
        rgb = np.stack([vol[cx,:,:], vol[:,cy,:], vol[:,:,cz]], -1)
        rgb = ((rgb-rgb.min())/(rgb.max()-rgb.min())*255).astype(np.uint8)
        pix = self.processor(images=Image.fromarray(rgb), return_tensors="pt")\
                   .pixel_values.squeeze(0)
        return pix, torch.tensor(label)

class FusionDataset(Dataset):
    def __init__(self, eeg_ds, mri_ds):
        self.e_map = {os.path.basename(p).split("_")[0]:i
                      for i,(p,_) in enumerate(eeg_ds.items)}
        self.m_map = {os.path.basename(p).split("_")[0]:i
                      for i,(p,_) in enumerate(mri_ds.items)}
        self.eeg_ds, self.mri_ds = eeg_ds, mri_ds
        self.ids = list(set(self.e_map).intersection(self.m_map))
    def __len__(self): return len(self.ids)
    def __getitem__(self, idx):
        pid = self.ids[idx]
        e_pix, lbl = self.eeg_ds[self.e_map[pid]]
        m_pix, _   = self.mri_ds[self.m_map[pid]]
        return e_pix, m_pix, lbl

# ============================
# Classification head
# ============================
class SimpleHead(nn.Module):
    def __init__(self, in_dim, hidden=256, ncls=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden, ncls)
        )
    def forward(self,x): return self.net(x)

eeg_head    = SimpleHead(embed_dim).to(device)
mri_head    = SimpleHead(embed_dim).to(device)
fusion_head= SimpleHead(embed_dim*2).to(device)

# ============================
# DataLoader helper
# ============================
def make_loaders(ds, bs):
    labels = [int(ds[i][-1]) for i in range(len(ds))]
    w = 1.0/np.bincount(labels)
    sw = [w[l] for l in labels]
    n = len(ds); t = int(0.8*n)
    td, vd = random_split(ds, [t, n-t])
    tw = [sw[i] for i in td.indices]
    sampler = WeightedRandomSampler(tw, len(tw), True)
    return DataLoader(td, batch_size=bs, sampler=sampler), DataLoader(vd, batch_size=bs)

# prepare datasets
eeg_ds    = EEGDataset(eeg_dir,   gender_map)
mri_ds    = MRIDataset(mri_dir,   gender_map, processor)
fusion_ds = FusionDataset(eeg_ds, mri_ds)

eeg_train,eeg_val         = make_loaders(eeg_ds,    BATCH_SIZES["EEG"])
mri_train,mri_val         = make_loaders(mri_ds,    BATCH_SIZES["MRI"])
fusion_train,fusion_val   = make_loaders(fusion_ds, BATCH_SIZES["Fusion"])

# ============================
# Training & TSNE for single modality
# ============================
def train_single(model, head, tr, vl, name):
    params = [p for p in model.parameters() if p.requires_grad] + list(head.parameters())
    opt = optim.AdamW(params, lr=LR, weight_decay=WD)
    sched = CosineAnnealingLR(opt, T_max=EPOCHS)
    crit  = nn.CrossEntropyLoss()

    for ep in range(1, EPOCHS+1):
        model.train(); head.train()
        total_loss=total_corr=total_n=0
        for pix,lbl in tqdm(tr, desc=f"{name} EP{ep}"):
            opt.zero_grad()
            emb = model.get_image_features(pixel_values=pix.to(device)).float()
            logits = head(emb)
            loss   = crit(logits, lbl.to(device))
            loss.backward(); opt.step()
            preds  = logits.argmax(1)
            total_loss += loss.item()*lbl.size(0)
            total_corr += (preds==lbl.to(device)).sum().item()
            total_n    += lbl.size(0)
        sched.step()
        print(f"{name} EP{ep}: loss={total_loss/total_n:.4f} train_acc={total_corr/total_n:.4f}")

    # Validation
    y_t,y_p=[],[]
    model.eval(); head.eval()
    with torch.no_grad():
        for pix,lbl in vl:
            emb = model.get_image_features(pixel_values=pix.to(device)).float()
            preds = head(emb).argmax(1).cpu().numpy()
            y_p.extend(preds); y_t.extend(lbl.numpy())
    print(f"\n{name} Report:")
    print(classification_report(y_t,y_p,target_names=["Male","Female"]))

    # t-SNE plot
    feats,labs=[],[]
    with torch.no_grad():
        for pix,lbl in vl:
            emb = model.get_image_features(pixel_values=pix.to(device)).float().cpu().numpy()
            feats.append(emb); labs.append(lbl.numpy())
    feats = np.vstack(feats); labs = np.concatenate(labs)
    proj = TSNE(n_components=2, random_state=42).fit_transform(feats)
    plt.figure(figsize=(6,6))
    sns.scatterplot(x=proj[:,0], y=proj[:,1], hue=labs, palette={0:"C0",1:"C1"}, legend='full', alpha=0.7)
    plt.title(f"t-SNE of {name} Embeddings by Gender")
    plt.show()

# ============================
# Training & TSNE for fusion
# ============================
def train_fusion(eeg_model, mri_model, head, tr, vl, name):
    params = ([p for p in eeg_model.parameters() if p.requires_grad] +
              [p for p in mri_model.parameters() if p.requires_grad] +
              list(head.parameters()))
    opt = optim.AdamW(params, lr=LR, weight_decay=WD)
    sched = CosineAnnealingLR(opt, T_max=EPOCHS)
    crit  = nn.CrossEntropyLoss()

    for ep in range(1, EPOCHS+1):
        eeg_model.train(); mri_model.train(); head.train()
        total_loss=total_corr=total_n=0
        for e_pix,m_pix,lbl in tqdm(tr, desc=f"{name} EP{ep}"):
            opt.zero_grad()
            emb_e = eeg_model.get_image_features(pixel_values=e_pix.to(device)).float()
            emb_m = mri_model.get_image_features(pixel_values=m_pix.to(device)).float()
            inp   = torch.cat([emb_e,emb_m],1)
            logits= head(inp)
            loss  = crit(logits, lbl.to(device))
            loss.backward(); opt.step()
            preds = logits.argmax(1)
            total_loss += loss.item()*lbl.size(0)
            total_corr += (preds==lbl.to(device)).sum().item()
            total_n    += lbl.size(0)
        sched.step()
        print(f"{name} EP{ep}: loss={total_loss/total_n:.4f} train_acc={total_corr/total_n:.4f}")

    # Validation
    y_t,y_p=[],[]
    eeg_model.eval(); mri_model.eval(); head.eval()
    with torch.no_grad():
        for e_pix,m_pix,lbl in vl:
            emb_e = eeg_model.get_image_features(pixel_values=e_pix.to(device)).float()
            emb_m = mri_model.get_image_features(pixel_values=m_pix.to(device)).float()
            inp   = torch.cat([emb_e,emb_m],1)
            preds = head(inp).argmax(1).cpu().numpy()
            y_p.extend(preds); y_t.extend(lbl.numpy())
    print(f"\n{name} Report:")
    print(classification_report(y_t,y_p,target_names=["Male","Female"]))

    # t-SNE
    feats,labs=[],[]
    with torch.no_grad():
        for e_pix,m_pix,lbl in vl:
            emb_e = eeg_model.get_image_features(pixel_values=e_pix.to(device)).float().cpu().numpy()
            emb_m = mri_model.get_image_features(pixel_values=m_pix.to(device)).float().cpu().numpy()
            emb   = np.concatenate([emb_e,emb_m],axis=1)
            feats.append(emb); labs.append(lbl.numpy())
    feats = np.vstack(feats); labs = np.concatenate(labs)
    proj = TSNE(n_components=2, random_state=42).fit_transform(feats)
    plt.figure(figsize=(6,6))
    sns.scatterplot(x=proj[:,0], y=proj[:,1], hue=labs, palette={0:"C0",1:"C1"}, legend='full', alpha=0.7)
    plt.title(f"t-SNE of {name} Fusion Embeddings by Gender")
    plt.show()

# ============================
# Execute all
# ============================
print("\n>> EEG-only")
train_single(clip_eeg, eeg_head, eeg_train, eeg_val, "EEG")
print("\n>> MRI-only")
train_single(clip_mri, mri_head, mri_train, mri_val, "MRI")
print("\n>> Fusion")
train_fusion(clip_eeg, clip_mri, fusion_head, fusion_train, fusion_val, "Fusion")
print("All done!")
