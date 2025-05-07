import os
import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models.video as models
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from scipy.ndimage import zoom
import pandas as pd

# ✅ Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔥 Using device: {device}")

# ===============================
# ✅ Load Age Data
# ===============================
age_csv_path = r"C:\research\MRI\participants_LSD_andLEMON.csv"
age_data = pd.read_csv(age_csv_path)

# ✅ Define age groups
valid_ages = ["20-25", "25-30", "60-65", "65-70", "70-75"]
age_group_mapping = {"20-25": "young", "25-30": "young",
                     "60-65": "old", "65-70": "old", "70-75": "old"}

# ✅ Filter valid ages and assign labels
age_data = age_data[age_data['age'].isin(valid_ages)]
age_data['age_group'] = age_data['age'].map(age_group_mapping)
age_data['label'] = age_data['age_group'].map({"young": 0, "old": 1})

# ✅ Fix Participant ID formatting
age_data['participant_id'] = age_data['participant_id'].apply(lambda x: x.replace("sub-sub-", "sub-"))

# ===============================
# ✅ Load MRI Data
# ===============================
mri_data_path = r"C:\research\MRI\structural_MRI"
mri_files = [f for f in os.listdir(mri_data_path) if f.endswith('.nii') or f.endswith('.nii.gz')]

# ✅ Extract participant IDs from MRI filenames
def extract_participant_id(mri_filename):
    return mri_filename.split('_')[0]

filtered_mri_files = [f for f in mri_files if extract_participant_id(f) in age_data['participant_id'].values]

# ✅ Function to load and preprocess 3D MRI scans
def load_mri(file_path, target_shape=(128, 128, 128)):
    try:
        img = nib.load(file_path).get_fdata()
        img = (img - np.min(img)) / (np.max(img) - np.min(img))  # Normalize
        
        # ✅ Rescale to target shape
        zoom_factors = (target_shape[0] / img.shape[0], 
                        target_shape[1] / img.shape[1], 
                        target_shape[2] / img.shape[2])
        img_resized = zoom(img, zoom_factors, order=1)  # Linear interpolation
        
        return img_resized.astype(np.float32)
    except Exception as e:
        print(f"❌ Error loading {file_path}: {e}")
        return None

# ===============================
# ✅ Custom Dataset Class
# ===============================
class MRIDataset(Dataset):
    def __init__(self, mri_files, labels, mri_dir):
        self.mri_files = mri_files
        self.labels = labels
        self.mri_dir = mri_dir
    
    def __len__(self):
        return len(self.mri_files)
    
    def __getitem__(self, idx):
        mri_file = self.mri_files[idx]
        participant_id = extract_participant_id(mri_file)
        label = self.labels[participant_id]

        file_path = os.path.join(self.mri_dir, mri_file)
        img = load_mri(file_path)
        if img is None:
            # If an image fails to load, return a tensor of zeros and label - you may adjust handling as needed.
            img = np.zeros((128, 128, 128), dtype=np.float32)

        # Convert to PyTorch tensor and repeat channel to match 3-channel ResNet input
        img_tensor = torch.tensor(img).unsqueeze(0)  # Shape: (1, 128, 128, 128)
        img_tensor = img_tensor.repeat(3, 1, 1, 1)  # Expand to (3, 128, 128, 128)

        return img_tensor, torch.tensor(label, dtype=torch.long)

# ✅ Prepare dataset & dataloader
mri_labels = {row['participant_id']: row['label'] for _, row in age_data.iterrows()}
dataset = MRIDataset(filtered_mri_files, mri_labels, mri_data_path)

# ✅ Train-Test Split
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=0)

# ===============================
# ✅ Define 3D ResNet Model
# ===============================
class ResNet3DClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super(ResNet3DClassifier, self).__init__()
        self.resnet3d = models.r3d_18(weights="DEFAULT")  
        self.resnet3d.fc = nn.Linear(self.resnet3d.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet3d(x)

# ✅ Initialize Model
model = ResNet3DClassifier().to(device)

# ✅ Define Loss, Optimizer & Scheduler
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-4)  
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)  

# ===============================
# ✅ Training Loop with Validation Metrics
# ===============================
num_epochs = 10

# Lists to track training and validation loss and accuracy
train_loss_list = []
train_acc_list = []
val_loss_list = []
val_acc_list = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct, total = 0, 0

    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training"):
        inputs, labels = batch
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        preds = torch.argmax(outputs, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    
    train_loss = running_loss / total
    train_acc = correct / total
    train_loss_list.append(train_loss)
    train_acc_list.append(train_acc)
    
    # -----------------------------
    # Validate on the test dataset
    # -----------------------------
    model.eval()
    val_running_loss = 0.0
    val_correct, val_total = 0, 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_running_loss += loss.item() * inputs.size(0)
            preds = torch.argmax(outputs, dim=1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)
    
    val_loss = val_running_loss / val_total
    val_acc = val_correct / val_total
    val_loss_list.append(val_loss)
    val_acc_list.append(val_acc)
    
    scheduler.step()  
    print(f"Epoch [{epoch+1}/{num_epochs}]: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

# ===============================
# ✅ Plot Training vs Validation Curves
# ===============================
epochs = range(1, num_epochs + 1)
plt.figure(figsize=(12, 5))

# Loss Curve
plt.subplot(1, 2, 1)
plt.plot(epochs, train_loss_list, label="Training Loss", marker="o")
plt.plot(epochs, val_loss_list, label="Validation Loss", marker="o")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs. Validation Loss")
plt.legend()
plt.grid(True)

# Accuracy Curve
plt.subplot(1, 2, 2)
plt.plot(epochs, train_acc_list, label="Training Accuracy", marker="o")
plt.plot(epochs, val_acc_list, label="Validation Accuracy", marker="o")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training vs. Validation Accuracy")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# ===============================
# ✅ Evaluation on Test Set & Confusion Matrix
# ===============================
model.eval()
correct, total = 0, 0
all_preds, all_labels = [], []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        preds = torch.argmax(outputs, dim=1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

test_acc = correct / total
print(f"✅ Test Accuracy: {test_acc:.4f}")

# ✅ Generate Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(6,6))
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=["Young", "Old"], yticklabels=["Young", "Old"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

# ✅ Print Classification Report
print(classification_report(all_labels, all_preds, target_names=["Young", "Old"]))

# ✅ Save Model
torch.save(model.state_dict(), "mri_resnet3d_age_classifier.pth")
print("✅ Model saved successfully!")
