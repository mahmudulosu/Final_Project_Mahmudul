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

# ============================
# Config and device
# ============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ============================
# Data paths and labels
# ============================
eo_dir  = r"C:\research\EEG_Domain\eye_openscalo"
ec_dir  = r"C:\research\EEG_Domain\eyeclose_scalo"
mri_dir = r"C:\research\commonMRI"
csv_path = r"C:/research/MRI/participants_LSD_andLEMON.csv"

gender_df = pd.read_csv(csv_path)
gender_map = {row['participant_id']: 0 if row['gender']=='M' else 1
              for _, row in gender_df.iterrows()}

# ============================
# Datasets
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
        self.eo_map = {os.path.basename(p).split('_')[0]: i for i,(p,_) in enumerate(eo_ds.samples)}
        self.ec_map = {os.path.basename(p).split('_')[0]: i for i,(p,_) in enumerate(ec_ds.samples)}
        self.mri_map= {os.path.basename(p).split('_')[0]: i for i,(p,_) in enumerate(mri_ds.samples)}
        self.eo_ds, self.ec_ds, self.mri_ds = eo_ds, ec_ds, mri_ds
        self.ids = list(set(self.eo_map).intersection(self.ec_map).intersection(self.mri_map))

    def __len__(self): return len(self.ids)

    def __getitem__(self, idx):
        pid = self.ids[idx]
        eo_pix, label = self.eo_ds[self.eo_map[pid]]
        ec_pix, _     = self.ec_ds[self.ec_map[pid]]
        mri_pix, _    = self.mri_ds[self.mri_map[pid]]
        return eo_pix, ec_pix, mri_pix, label

# ============================
# Helpers: loader, head, train/eval
# ============================
def make_loader(ds, batch_size):
    labels = [int(ds[i][-1]) for i in range(len(ds))]
    weights = 1.0/np.bincount(labels)
    sampler_weights = [weights[l] for l in labels]
    t = int(0.8*len(ds)); v = len(ds)-t
    t_ds, v_ds = random_split(ds,[t,v])
    sw = [sampler_weights[i] for i in t_ds.indices]
    sampler = WeightedRandomSampler(sw,len(sw),True)
    return DataLoader(t_ds, batch_size=batch_size, sampler=sampler), DataLoader(v_ds, batch_size=batch_size)

class SimpleHead(nn.Module):
    def __init__(self, in_dim, hidden=256, num_classes=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(0.5), nn.Linear(hidden, num_classes)
        )
    def forward(self, x): return self.net(x)

# Single-modality train/eval functions
def train_single(backbone, head, loader, epochs=20, lr=2e-4, wd=1e-4):
    params = [p for p in backbone.parameters() if p.requires_grad] + list(head.parameters())
    opt = optim.AdamW(params, lr=lr, weight_decay=wd)
    crit = nn.CrossEntropyLoss()
    for e in range(1, epochs+1):
        backbone.train(); head.train()
        total_loss = correct = total = 0
        for px, lbl in loader:
            opt.zero_grad()
            feats = backbone(px.to(device))
            out = head(feats)
            loss = crit(out, lbl.to(device))
            loss.backward(); opt.step()
            preds = out.argmax(1)
            total_loss += loss.item()*lbl.size(0)
            correct += (preds==lbl.to(device)).sum().item()
            total += lbl.size(0)
        print(f"Epoch {e}/{epochs} - Loss: {total_loss/total:.4f}, Acc: {correct/total:.4f}")

def eval_single(backbone, head, loader):
    backbone.eval(); head.eval()
    feats, preds, labels = [], [], []
    with torch.no_grad():
        for px, lbl in loader:
            f = backbone(px.to(device))
            feats.append(f.cpu().numpy())
            out = head(f)
            preds.extend(out.argmax(1).cpu().numpy()); labels.extend(lbl.numpy())
    return np.concatenate(feats,0), confusion_matrix(labels, preds), classification_report(labels, preds, target_names=['M','F']), np.array(labels)

# Fusion train/eval functions
def train_fusion(back_eeg, back_mri, head, loader, epochs=20, lr=2e-4, wd=1e-4):
    params = [p for p in back_eeg.parameters() if p.requires_grad] + [p for p in back_mri.parameters() if p.requires_grad] + list(head.parameters())
    opt = optim.AdamW(params, lr=lr, weight_decay=wd)
    crit = nn.CrossEntropyLoss()
    for e in range(1, epochs+1):
        back_eeg.train(); back_mri.train(); head.train()
        total_loss = correct = total = 0
        for eo, ec, mr, lbl in loader:
            opt.zero_grad()
            f_eo = back_eeg(eo.to(device))
            f_ec = back_eeg(ec.to(device))
            f_mr = back_mri(mr.to(device))
            f = torch.cat([f_eo, f_ec, f_mr],1)
            out = head(f)
            loss = crit(out, lbl.to(device)); loss.backward(); opt.step()
            preds = out.argmax(1)
            total_loss += loss.item()*lbl.size(0)
            correct += (preds==lbl.to(device)).sum().item()
            total += lbl.size(0)
        print(f"Epoch {e}/{epochs} - Loss: {total_loss/total:.4f}, Acc: {correct/total:.4f}")

def eval_fusion(back_eeg, back_mri, head, loader):
    back_eeg.eval(); back_mri.eval(); head.eval()
    feats, preds, labels = [], [], []
    with torch.no_grad():
        for eo, ec, mr, lbl in loader:
            f_eo = back_eeg(eo.to(device))
            f_ec = back_eeg(ec.to(device))
            f_mr = back_mri(mr.to(device))
            f = torch.cat([f_eo, f_ec, f_mr],1)
            feats.append(f.cpu().numpy())
            out = head(f)
            preds.extend(out.argmax(1).cpu().numpy()); labels.extend(lbl.numpy())
    return np.concatenate(feats,0), confusion_matrix(labels, preds), classification_report(labels, preds, target_names=['M','F']), np.array(labels)

# ============================
# Main experiments
# ============================
set_seed()

eo_ds = EODataset(eo_dir, gender_map)
ec_ds = ECDataset(ec_dir, gender_map)
mri_ds= MRIDataset(mri_dir, gender_map)
fus3_ds = Fusion3Dataset(eo_ds, ec_ds, mri_ds)

eo_train, eo_val   = make_loader(eo_ds, 16)
ec_train, ec_val   = make_loader(ec_ds, 16)
mri_train, mri_val = make_loader(mri_ds, 8)
fus3_train, fus3_val = make_loader(fus3_ds, 8)

for variant in ['resnet50','resnet101','resnet152']:
    print(f"\n=== Training with {variant} ===")
    # EEG backbone
    base_eeg = getattr(models, variant)(pretrained=True)
    old = base_eeg.conv1
    new = nn.Conv2d(17, old.out_channels, old.kernel_size, old.stride, old.padding, bias=False)
    with torch.no_grad():
        mean_w = old.weight.mean(dim=1, keepdim=True)
        new.weight.copy_(mean_w.repeat(1,17,1,1))
    base_eeg.conv1 = new
    feat_dim = base_eeg.fc.in_features
    base_eeg.fc = nn.Identity()
    for name, p in base_eeg.named_parameters():
        p.requires_grad = name.startswith(('layer3','layer4','bn1','bn2'))
    back_eeg = base_eeg.to(device)
    head_eo = SimpleHead(feat_dim).to(device)

    # MRI backbone
    base_mri = getattr(models, variant)(pretrained=True)
    feat_mri_dim = base_mri.fc.in_features
    base_mri.fc = nn.Identity()
    for name, p in base_mri.named_parameters():
        p.requires_grad = name.startswith(('layer3','layer4','bn1','bn2'))
    back_mri = base_mri.to(device)
    head_mri = SimpleHead(feat_mri_dim).to(device)

    # Fusion head
    head_fus = SimpleHead(feat_dim*3).to(device)

    # EO-only
    print("\n>> EO-only Training")
    train_single(back_eeg, head_eo, eo_train)
    feats_eo_tr, _, _, labels_eo_tr = eval_single(back_eeg, head_eo, eo_train)
    tsne_eo_tr = TSNE(n_components=2, random_state=42).fit_transform(feats_eo_tr)
    plt.figure(figsize=(6,5))
    plt.scatter(tsne_eo_tr[:,0], tsne_eo_tr[:,1], c=labels_eo_tr, alpha=0.7)
    plt.title(f'{variant} EO Training TSNE')
    plt.show()
    feats_eo, cm_eo, cr_eo, labels_eo = eval_single(back_eeg, head_eo, eo_val)
    tsne_eo = TSNE(n_components=2, random_state=42).fit_transform(feats_eo)
    plt.figure(figsize=(6,5))
    plt.scatter(tsne_eo[:,0], tsne_eo[:,1], c=labels_eo, alpha=0.7)
    plt.title(f'{variant} EO Validation TSNE')
    plt.show()
    print("Confusion Matrix:\n", cm_eo)
    print("Classification Report:\n", cr_eo)

    # MRI-only
    print("\n>> MRI-only Training")
    train_single(back_mri, head_mri, mri_train)
    feats_mri_tr, _, _, labels_mri_tr = eval_single(back_mri, head_mri, mri_train)
    tsne_mri_tr = TSNE(n_components=2, random_state=42).fit_transform(feats_mri_tr)
    plt.figure(figsize=(6,5))
    plt.scatter(tsne_mri_tr[:,0], tsne_mri_tr[:,1], c=labels_mri_tr, alpha=0.7)
    plt.title(f'{variant} MRI Training TSNE')
    plt.show()
    feats_mri, cm_mri, cr_mri, labels_mri = eval_single(back_mri, head_mri, mri_val)
    tsne_mri = TSNE(n_components=2, random_state=42).fit_transform(feats_mri)
    plt.figure(figsize=(6,5))
    plt.scatter(tsne_mri[:,0], tsne_mri[:,1], c=labels_mri, alpha=0.7)
    plt.title(f'{variant} MRI Validation TSNE')
    plt.show()
    print("Confusion Matrix:\n", cm_mri)
    print("Classification Report:\n", cr_mri)

    # Fusion EO+EC+MRI
    print("\n>> Fusion EO+EC+MRI Training")
    train_fusion(back_eeg, back_mri, head_fus, fus3_train)
    feats_fus_tr, _, _, labels_fus_tr = eval_fusion(back_eeg, back_mri, head_fus, fus3_train)
    tsne_fus_tr = TSNE(n_components=2, random_state=42).fit_transform(feats_fus_tr)
    plt.figure(figsize=(6,5))
    plt.scatter(tsne_fus_tr[:,0], tsne_fus_tr[:,1], c=labels_fus_tr, alpha=0.7)
    plt.title(f'{variant} Fusion Training TSNE')
    plt.show()
    feats_fus, cm_fus, cr_fus, labels_fus = eval_fusion(back_eeg, back_mri, head_fus, fus3_val)
    tsne_fus = TSNE(n_components=2, random_state=42).fit_transform(feats_fus)
    plt.figure(figsize=(6,5))
    plt.scatter(tsne_fus[:,0], tsne_fus[:,1], c=labels_fus, alpha=0.7)
    plt.title(f'{variant} Fusion Validation TSNE')
    plt.show()
    print("Confusion Matrix:\n", cm_fus)
    print("Classification Report:\n", cr_fus)

print("All experiments done")
