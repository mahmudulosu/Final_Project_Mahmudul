import os
import numpy as np
import pandas as pd
import pywt
import cv2
from tqdm import tqdm

# ============================
# Configuration
# ============================
data_dir = r"C:\\research\\EEG_Domain\\combined\\combined_new"  # EEG data folder
save_dir = r"C:\\research\\EEG_Domain\\scalograms_format"  # Output folder for saving scalograms
os.makedirs(save_dir, exist_ok=True)

sampling_rate = 250  # Hz
segment_length_seconds = 10  # Each segment is 10s
segment_length = segment_length_seconds * sampling_rate  # 10s → 2500 samples
target_size = (224, 224)  # Resize scalograms to this size

# ============================
# Function: Generate CWT Scalogram
# ============================
def generate_scalogram(signal):
    """
    Compute the Continuous Wavelet Transform (CWT) scalogram.
    Returns a resized scalogram (224, 224).
    """
    scales = np.arange(1, 128)  # Define scale range (127 frequency bands)
    coeffs, freqs = pywt.cwt(signal, scales, 'morl')  # Compute CWT
    scalogram = np.abs(coeffs)  # Convert to magnitude scalogram

    # Resize to (224, 224)
    resized_scalogram = cv2.resize(scalogram, target_size, interpolation=cv2.INTER_CUBIC)
    
    return resized_scalogram  # Final shape: (224, 224)

# ============================
# Process All EEG Files
# ============================
all_subjects = [f for f in os.listdir(data_dir) if f.endswith('.csv')]

for file_name in tqdm(all_subjects, desc="Processing EEG Files"):
    # Extract subject ID and condition (EC = Eye Closed, EO = Eye Open)
    if "_EC" in file_name:
        eye_condition = "EC"
    elif "_EO" in file_name:
        eye_condition = "EO"
    else:
        print(f"Skipping unknown file: {file_name}")
        continue

    subject_id = file_name.replace(".csv", "").replace("_EC", "").replace("_EO", "")
    file_path = os.path.join(data_dir, file_name)

    print(f"\nProcessing Subject: {subject_id} - Condition: {eye_condition}")

    # Load EEG Data
    data = pd.read_csv(file_path)

    # Skip empty files
    if data.empty:
        print(f"Skipping empty file: {subject_id} - {eye_condition}")
        continue

    # Check available electrodes (should be 17)
    electrodes = list(data.columns)
    if len(electrodes) != 17:
        print(f"Warning: {subject_id} - {eye_condition} does not have 17 electrodes! Found: {len(electrodes)}")
        continue

    # Split EEG data into 10-second segments
    for i in range(0, len(data), segment_length):
        segment = data.iloc[i:i + segment_length]

        if len(segment) < segment_length:
            print(f"Skipping last incomplete segment: {len(segment)} samples")
            continue

        time_index = i // segment_length  # Track time segments

        # Generate scalograms for each of the 17 electrodes
        all_electrode_scalograms = []
        for electrode in electrodes:
            signal = segment[electrode].dropna().values  # Ensure no NaN values
            scalogram = generate_scalogram(signal)  # Compute CWT scalogram
            all_electrode_scalograms.append(scalogram)

        # Stack 17 electrodes to create (224, 224, 17) tensor
        segment_tensor = np.stack(all_electrode_scalograms, axis=-1)  # Shape: (224, 224, 17)

        # Save as .npy file with Subject ID + Eye Condition
        save_path = os.path.join(save_dir, f"{subject_id}_{eye_condition}_seg{time_index}.npy")
        np.save(save_path, segment_tensor)

    print(f"Finished processing {subject_id} - Condition: {eye_condition}\n")

print("✅ All EEG data has been converted to (224, 224, 17) scalograms and saved successfully.")
