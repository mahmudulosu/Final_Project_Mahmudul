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
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report

# ============================
# Configuration
# ============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

eeg_dir  = r"C:\research\EEG_Domain\eyeclose_scalo"
mri_dir  = r"C:\research\commonMRI"
csv_path = r"C:/research/MRI/participants_LSD_andLEMON.csv"
UNFREEZE_BLOCKS = 4

# ============================
# Load gender labels
# ============================
gender_df = pd.read_csv(csv_path)
gender_map = {row['participant_id']: 0 if row['gender']=='M' else 1
              for _, row in gender_df.iterrows()}

# ============================
# Prepare CLIP for EEG (17-channel input)
clip_eeg = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
vision_cfg = clip_eeg.config.vision_config
patch_size = vision_cfg.patch_size
old_conv = clip_eeg.vision_model.embeddings.patch_embedding
new_conv = nn.Conv2d(
    in_channels=17,
    out_channels=old_conv.out_channels,
    kernel_size=patch_size,
    stride=patch_size,
    bias=False
)
with torch.no_grad():
    mean_w = old_conv.weight.mean(dim=1, keepdim=True)
    new_conv.weight.copy_(mean_w.repeat(1, 17, 1, 1))
# Move new_conv to the same device as the rest of the model
new_conv = new_conv.to(device)
# Replace patch_embedding with the new conv
clip_eeg.vision_model.embeddings.patch_embedding = new_conv
for p in clip_eeg.parameters(): p.requires_grad = False
for blk in clip_eeg.vision_model.encoder.layers[-UNFREEZE_BLOCKS:]:
    for p in blk.parameters(): p.requires_grad = True
for name, p in clip_eeg.named_parameters():
    if "layer_norm" in name or "visual_projection" in name:
        p.requires_grad = True

# ============================
# Prepare CLIP for MRI (3-channel input)
# ============================
processor_mri = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_mri = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
for p in clip_mri.parameters(): p.requires_grad = False
for blk in clip_mri.vision_model.encoder.layers[-UNFREEZE_BLOCKS:]:
    for p in blk.parameters(): p.requires_grad = True
for name, p in clip_mri.named_parameters():
    if "layer_norm" in name or "visual_projection" in name:
        p.requires_grad = True

embed_dim = clip_eeg.config.projection_dim

# ============================
# Dataset classes
# ============================
class EEGDataset(Dataset):
    def __init__(self, npy_dir, labels):
        self.samples = []
        for fn in os.listdir(npy_dir):
            if not fn.endswith(".npy"): continue
            pid = fn.split("_")[0]
            if pid not in labels: continue
            self.samples.append((os.path.join(npy_dir, fn), labels[pid]))

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        arr = np.load(path)
        arr = (arr - arr.min())/(arr.max()-arr.min()+1e-5)
        pix = torch.tensor(arr, dtype=torch.float32).permute(2,0,1)
        return pix, torch.tensor(label, dtype=torch.long)

class MRIDataset(Dataset):
    def __init__(self, nii_dir, labels, processor):
        self.samples = []
        for fn in os.listdir(nii_dir):
            if not fn.endswith((".nii",".nii.gz")): continue
            pid = fn.split("_")[0]
            if pid not in labels: continue
            self.samples.append((os.path.join(nii_dir, fn), labels[pid]))
        self.processor = processor

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
        pix = self.processor(images=pil, return_tensors="pt").pixel_values.squeeze(0)
        return pix, torch.tensor(label, dtype=torch.long)

class FusionDataset(Dataset):
    def __init__(self, eeg_ds, mri_ds):
        self.eeg_map = {os.path.basename(p).split("_")[0]: i for i,(p,_) in enumerate(eeg_ds.samples)}
        self.mri_map = {os.path.basename(p).split("_")[0]: i for i,(p,_) in enumerate(mri_ds.samples)}
        self.eeg_ds, self.mri_ds = eeg_ds, mri_ds
        self.ids = list(set(self.eeg_map).intersection(self.mri_map))

    def __len__(self): return len(self.ids)

    def __getitem__(self, idx):
        pid = self.ids[idx]
        eeg_pix, label = self.eeg_ds[self.eeg_map[pid]]
        mri_pix, _     = self.mri_ds[self.mri_map[pid]]
        return eeg_pix, mri_pix, label

# ============================
# Classification heads
# ============================
class SimpleHead(nn.Module):
    def __init__(self, in_dim, hidden=256, num_classes=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(0.5), nn.Linear(hidden, num_classes)
        )
    def forward(self, x): return self.net(x)

eeg_head = SimpleHead(embed_dim).to(device)
mlri_head = SimpleHead(embed_dim).to(device)
fusion_head = SimpleHead(embed_dim*2).to(device)

# ============================
# DataLoader helper
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

# Prepare datasets & loaders
eeg_ds = EEGDataset(eeg_dir, gender_map)
mri_ds = MRIDataset(mri_dir, gender_map, processor_mri)
fus_ds = FusionDataset(eeg_ds, mri_ds)

eeg_train,eeg_val=make_loader(eeg_ds,16)
mri_train,mri_val=make_loader(mri_ds,8)
fus_train,fus_val=make_loader(fus_ds,8)

# ============================
# Training & Evaluation
# ============================
def train_eval(model, head, tr, vl, epochs=20, lr=2e-4, wd=1e-4):
    params=[p for p in model.parameters() if p.requires_grad]+list(head.parameters())
    opt=optim.AdamW(params,lr=lr,weight_decay=wd)
    sch=CosineAnnealingLR(opt,T_max=epochs)
    crit=nn.CrossEntropyLoss()
    for e in range(1,epochs+1):
        model.train(); head.train();tl=tc=tt=0
        for b in tr:
            opt.zero_grad()
            if len(b)==3:
                ex, mx, lb=b
                fe=model.get_image_features(pixel_values=ex.to(device)).float()
                fm=clip_mri.get_image_features(pixel_values=mx.to(device)).float()
                inp=torch.cat([fe,fm],1)
            else:
                px,lb=b
                inp=model.get_image_features(pixel_values=px.to(device)).float()
            lo=head(inp);loss=crit(lo,lb.to(device))
            loss.backward();opt.step()
            p=lo.argmax(1)
            tl+=loss.item()*lb.size(0);tc+=(p==lb.to(device)).sum().item();tt+=lb.size(0)
        sch.step();print(f"Ep{e}/{epochs} loss {tl/tt:.4f} acc {tc/tt:.4f}")
    model.eval(); head.eval();ap,al=[],[]
    with torch.no_grad():
        for b in vl:
            if len(b)==3:
                ex,mx,lb=b
                fe=model.get_image_features(pixel_values=ex.to(device)).float()
                fm=clip_mri.get_image_features(pixel_values=mx.to(device)).float()
                inp=torch.cat([fe,fm],1)
            else:
                px,lb=b;inp=model.get_image_features(pixel_values=px.to(device)).float()
            pr=head(inp).argmax(1).cpu().numpy()
            ap.extend(pr);al.extend(lb.numpy())
    print(confusion_matrix(al,ap));print(classification_report(al,ap,target_names=['M','F']))

# ============================
# Run experiments
# ============================
print("\n>> EEG-only")
train_eval(clip_eeg,eeg_head,eeg_train,eeg_val)
print("\n>> MRI-only")
train_eval(clip_mri,mlri_head,mri_train,mri_val)
print("\n>> Fusion")
train_eval(clip_eeg,fusion_head,fus_train,fus_val)
print("Done")
