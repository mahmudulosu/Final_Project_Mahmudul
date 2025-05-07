import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.utils.data as data
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import classification_report, confusion_matrix

# ============================
# Configuration
# ============================
scalogram_dir = r"C:\research\EEG_Domain\scalograms_format"  # Folder with (224, 224, 17) scalograms
fatigue_csv_path = r"C:\codes\pytorchRun\Deep_learning_project\MDBF_Day1.csv"  # CSV with fatigue info
batch_size = 32
num_epochs = 50
learning_rate = 0.0001
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available

# ============================
# Load Fatigue Labels
# ============================
# The CSV must include columns: "ID" and "MDBF_Day1_WM_Scale"
labels_df = pd.read_csv(fatigue_csv_path)

# Convert fatigue scores to numeric (errors coerced to NaN)
labels_df['MDBF_Day1_WM_Scale'] = pd.to_numeric(labels_df['MDBF_Day1_WM_Scale'], errors='coerce')

# Filter using the extreme group strategy (only keep scores ≤28 or ≥38)
labels_df = labels_df[(labels_df['MDBF_Day1_WM_Scale'] <= 30) | (labels_df['MDBF_Day1_WM_Scale'] >= 35)]

# Create binary fatigue label: label=1 if score ≤28 ("tired"), else label=0 ("awake")
labels_df['label'] = (labels_df['MDBF_Day1_WM_Scale'] <= 28).astype(int)

# Create dictionary mapping participant ID (as string) to fatigue label
fatigue_mapping = {str(row['ID']): row['label'] for _, row in labels_df.iterrows()}

# ============================
# Participant-Level Train/Validation/Test Split
# ============================
# List all .npy files in the scalogram directory
all_files = [f for f in os.listdir(scalogram_dir) if f.endswith('.npy')]

# Filter files to include only those whose participant ID is in fatigue_mapping.
# Assumes file naming convention: participantID_condition_segX.npy
all_files = [f for f in all_files if f.split('_')[0] in fatigue_mapping]

# Extract unique participant IDs and shuffle them
participant_ids = list({f.split('_')[0] for f in all_files})
random.shuffle(participant_ids)

# Split participants into train (70%), validation (15%), and test (15%)
num_total = len(participant_ids)
train_end = int(0.7 * num_total)
val_end = train_end + int(0.15 * num_total)

train_pids = set(participant_ids[:train_end])
val_pids   = set(participant_ids[train_end:val_end])
test_pids  = set(participant_ids[val_end:])

# Assign files to train, validation, and test sets based on participant ID
train_files = [f for f in all_files if f.split('_')[0] in train_pids]
val_files   = [f for f in all_files if f.split('_')[0] in val_pids]
test_files  = [f for f in all_files if f.split('_')[0] in test_pids]

print(f"Total participants: {len(participant_ids)}")
print(f"Train participants: {len(train_pids)}; Validation participants: {len(val_pids)}; Test participants: {len(test_pids)}")
print(f"Total files: {len(all_files)}; Train files: {len(train_files)}; Validation files: {len(val_files)}; Test files: {len(test_files)}")

# ============================
# Custom Dataset for EEG Scalograms (Fatigue)
# ============================
class FatigueEEGDataset(data.Dataset):
    def __init__(self, scalogram_dir, fatigue_mapping, file_list, transform=None):
        self.scalogram_dir = scalogram_dir
        self.fatigue_mapping = fatigue_mapping
        self.file_list = file_list
        self.transform = transform
    
    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_name = self.file_list[idx]
        file_path = os.path.join(self.scalogram_dir, file_name)
        # Load EEG scalogram; expected shape: (224, 224, 17)
        scalogram = np.load(file_path)
        # Convert to tensor and rearrange to (17, 224, 224)
        scalogram = torch.tensor(scalogram, dtype=torch.float32).permute(2, 0, 1)

        # Extract participant ID (assumes naming convention: participantID_condition_segX.npy)
        participant_id = file_name.split('_')[0]
        label = self.fatigue_mapping.get(participant_id)
        if self.transform:
            scalogram = self.transform(scalogram)

        return scalogram, label

# Create dataset instances for train, validation, and test sets
train_dataset = FatigueEEGDataset(scalogram_dir, fatigue_mapping, train_files)
val_dataset   = FatigueEEGDataset(scalogram_dir, fatigue_mapping, val_files)
test_dataset  = FatigueEEGDataset(scalogram_dir, fatigue_mapping, test_files)

# ============================
# Compute Class Weights for Loss and Sampler (from training set)
# ============================
train_labels = [train_dataset[i][1] for i in range(len(train_dataset))]
class_counts = np.bincount(train_labels)  # e.g., [num_awake, num_tired]
max_count = np.max(class_counts)
loss_class_weights = [max_count / count for count in class_counts]
loss_class_weights_tensor = torch.tensor(loss_class_weights, dtype=torch.float32).to(device)

# Create a weighted random sampler for the training set
sampler_weights = [loss_class_weights[label] for label in train_labels]
sampler = WeightedRandomSampler(sampler_weights, num_samples=len(sampler_weights), replacement=True)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# ============================
# Modify ResNet for Fatigue Classification (Unfreeze all layers)
# ============================
class ResNetEEG(nn.Module):
    def __init__(self, pretrained=True):
        super(ResNetEEG, self).__init__()
        self.resnet = models.resnet50(pretrained=pretrained)
        # Modify the first convolution to accept 17-channel input instead of 3-channel
        self.resnet.conv1 = nn.Conv2d(17, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Unfreeze all layers (set requires_grad=True for every parameter)
        for param in self.resnet.parameters():
            param.requires_grad = True

        # Modify the final fully connected layer for 2-class output (awake vs. tired)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, 2)

    def forward(self, x):
        return self.resnet(x)

# Initialize the model (note: using ResNetEEG for fatigue classification)
model = ResNetEEG(pretrained=True).to(device)
criterion = nn.CrossEntropyLoss(weight=loss_class_weights_tensor)
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)

# ============================
# Training and Validation Function (validate every 10 epochs)
# ============================
def train_model(model, train_loader, val_loader, num_epochs, criterion, optimizer, device):
    train_losses, train_accs = [], []
    val_losses, val_accs, val_epochs = [], [], []  # Store validation metrics only for epochs with validation

    for epoch in range(num_epochs):
        # Training Phase
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} (Training)"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
        train_loss = running_loss / len(train_loader)
        train_acc = correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # Perform validation only at every 10th epoch
        if (epoch + 1) % 10 == 0:
            model.eval()
            running_val_loss, correct_val, total_val = 0.0, 0, 0
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct_val += (predicted == labels).sum().item()
                total_val += labels.size(0)
            val_loss = running_val_loss / len(val_loader)
            val_acc = correct_val / total_val
            val_losses.append(val_loss)
            val_accs.append(val_acc)
            val_epochs.append(epoch + 1)
            print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        else:
            print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
    
    return train_losses, train_accs, val_losses, val_accs, val_epochs

# Train the model (with validation every 10 epochs)
train_losses, train_accs, val_losses, val_accs, val_epochs = train_model(
    model, train_loader, val_loader, num_epochs, criterion, optimizer, device
)

# ============================
# Evaluate on Test Set
# ============================
model.eval()
test_loss, correct, total = 0.0, 0, 0
all_labels, all_preds = [], []
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        test_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(predicted.cpu().numpy())
test_loss /= len(test_loader)
test_acc = correct / total

print(f"\nTest Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")
print("\nClassification Report:\n", classification_report(all_labels, all_preds))
conf_matrix = confusion_matrix(all_labels, all_preds)

# Plot Confusion Matrix
plt.figure(figsize=(6, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", 
            xticklabels=["Awake", "Tired"], yticklabels=["Awake", "Tired"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Test Confusion Matrix")
plt.show()

# ============================
# Plot Training vs. Validation Curves
# ============================
epochs = range(1, num_epochs + 1)
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs, train_losses, label="Train Loss")
plt.scatter(val_epochs, val_losses, color="red", label="Validation Loss (every 10 epochs)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs. Validation Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs, train_accs, label="Train Accuracy")
plt.scatter(val_epochs, val_accs, color="red", label="Validation Accuracy (every 10 epochs)")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training vs. Validation Accuracy")
plt.legend()
plt.show()
