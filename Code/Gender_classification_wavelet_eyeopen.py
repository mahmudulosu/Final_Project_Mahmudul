import os
import pandas as pd
import numpy as np
import pywt
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.svm import SVC

# Constants
sampling_rate = 250  # Assuming the sampling rate is 250 Hz
segment_length = 5 * sampling_rate  # 2-second segments

# Path to the folder containing EEG data
data_folder = r"C:\research\common_data_eeg_ec"
csv_path = r'C:\research\MRI\participants_LSD_andLEMON.csv'

def load_eeg_data(file_name):
    """
    Load EEG data from a CSV file and segment it.
    """
    data = pd.read_csv(file_name)
    segments = [data.iloc[i:i + segment_length] for i in range(0, len(data), segment_length) if len(data.iloc[i:i + segment_length]) == segment_length]
    return segments

def extract_dwt_features(segment):
    """
    Apply Discrete Wavelet Transform (DWT) and extract statistical features.
    """
    features = []
    for column in segment.columns:
        data = segment[column].values
        coeffs = pywt.wavedec(data, 'db4', level=5)
        for coeff in coeffs:
            features.extend([
                np.mean(coeff),
                np.var(coeff),
                np.sqrt(np.mean(coeff**2))  # RMS
            ])
    return features

# Load gender information
data = pd.read_csv(csv_path)

# Print the columns to verify their names
print("Columns in CSV:", data.columns)

# Create a mapping from participant ID to gender
gender_mapping = {row['participant_id']: row['gender'] for _, row in data.iterrows()}

# Lists to hold the training and testing data
X_train_all = []
X_test_all = []
y_train_all = []
y_test_all = []

# Iterate over all files in the specified folder
for file_name in os.listdir(data_folder):
    if file_name.endswith(".csv"):
        file_path = os.path.join(data_folder, file_name)
        subject_id = file_name.split('_')[0]  # Assuming the subject ID is part of the file name before the first underscore
        
        if subject_id in gender_mapping:
            segments = load_eeg_data(file_path)
            
            subject_features = []
            subject_labels = []
            
            for segment in segments:
                segment = segment.apply(pd.to_numeric, errors='coerce').fillna(0)  # Convert to numeric and fill NA
                features = extract_dwt_features(segment)
                subject_features.append(features)
                subject_labels.append(gender_mapping[subject_id])
            
            # Convert lists to numpy arrays
            subject_features = np.array(subject_features)
            subject_labels = np.array(subject_labels)
            
            # Perform 80% train and 20% test split for the current subject
            X_train, X_test, y_train, y_test = train_test_split(
                subject_features, subject_labels, test_size=0.2, random_state=42, stratify=subject_labels)
            
            # Append the splits to the corresponding lists
            X_train_all.extend(X_train)
            X_test_all.extend(X_test)
            y_train_all.extend(y_train)
            y_test_all.extend(y_test)

# Convert the lists to numpy arrays for further processing
X_train_all = np.array(X_train_all)
X_test_all = np.array(X_test_all)
y_train_all = np.array(y_train_all)
y_test_all = np.array(y_test_all)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_all)
X_test_scaled = scaler.transform(X_test_all)

# Feature selection using ANOVA F-value to retain top 60% features
selector = SelectPercentile(score_func=f_classif, percentile=60)
X_train_selected = selector.fit_transform(X_train_scaled, y_train_all)
X_test_selected = selector.transform(X_test_scaled)

# Train a Support Vector Machine (SVM) classifier
model = SVC(kernel='linear', random_state=42)
model.fit(X_train_selected, y_train_all)

# Predict and evaluate the model
y_pred = model.predict(X_test_selected)
accuracy = accuracy_score(y_test_all, y_pred)

print(f'Test Accuracy: {accuracy * 100:.2f}%')

# Display confusion matrix and classification report for overall evaluation
print("\nConfusion Matrix:")
print(confusion_matrix(y_test_all, y_pred))

print("\nClassification Report:")
print(classification_report(y_test_all, y_pred, target_names=['Female', 'Male']))
