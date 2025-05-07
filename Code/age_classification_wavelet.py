import os

import pandas as pd
import numpy as np
import pywt
from scipy.stats import skew, kurtosis
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split, KFold
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

data_dir = r"C:\research\Common_Data"
csv_path = r'C:\research\MRI\participants_LSD_andLEMON.csv'

def load_eeg_data(file_name):
    """
    Load EEG data from a CSV file.
    """
    data = pd.read_csv(file_name)
    return data

def extract_dwt_features(data):
    """
    Apply Discrete Wavelet Transform (DWT) and extract statistical features.
    """
    features = []
    for column in data:
        coeffs = pywt.wavedec(data[column], 'db4', level=5)
        for coeff in coeffs:
            features.extend([
                np.mean(coeff),
                np.var(coeff),
                np.sqrt(np.mean(coeff**2))
                # skew(coeff),
                # kurtosis(coeff)
            ])
    return features

# Load age and gender information
data = pd.read_csv(csv_path)

# Print the columns to verify their names
print("Columns in CSV:", data.columns)

# Define valid age ranges and groupings
valid_ages = ["20-25", "25-30", "60-65", "65-70", "70-75"]
age_group_mapping = {"20-25": "young", "25-30": "young",
                     "60-65": "old", "65-70": "old", "70-75": "old"}

# Filter the data to include only the specified age ranges
filtered_data = data[data['age'].isin(valid_ages)]

# Create a mapping from participant ID to age group
age_mapping = {row['participant_id']: age_group_mapping[row['age']] for _, row in filtered_data.iterrows()}

# Load data and extract features
all_features = []
all_labels = []
subject_ids = []

# Dictionary to hold EO and EC data separately for each subject
eeg_data = {}

for file_name in os.listdir(data_dir):
    if file_name.endswith('.csv'):
        file_path = os.path.join(data_dir, file_name)
        data = load_eeg_data(file_path)
        subject_id, condition = file_name.split('_')[0], file_name.split('_')[1].split('.')[0]
        
        if subject_id in age_mapping:
            if subject_id not in eeg_data:
                eeg_data[subject_id] = {}
            eeg_data[subject_id][condition] = data

# Process each subject's data
for subject_id in eeg_data:
    if "EO" in eeg_data[subject_id] and "EC" in eeg_data[subject_id]:
        # Merge EO and EC data by averaging corresponding columns
        eo_data = eeg_data[subject_id]["EO"]
        ec_data = eeg_data[subject_id]["EC"]
        merged_data = (eo_data + ec_data) 
        
        features = extract_dwt_features(merged_data)
        all_features.append(features)
        subject_ids.append(subject_id)
        all_labels.append(age_mapping[subject_id])

all_features = np.array(all_features)
all_labels = np.array(all_labels)
subject_ids = np.array(subject_ids)

# Encode age labels to numeric values
label_mapping = {"young": 0, "old": 1}
all_labels = np.array([label_mapping[label] for label in all_labels])

# Feature selection using ANOVA F-value
selector = SelectPercentile(score_func=f_classif, percentile=60)
selected_features = selector.fit_transform(all_features, all_labels)

# Get unique subject ids
unique_subject_ids = np.unique(subject_ids)

# Initialize KFold
kf = KFold(n_splits=10, shuffle=True, random_state=42)

# Store accuracy for each fold
fold_accuracies = []

for train_index, test_index in kf.split(unique_subject_ids):
    train_subject_ids = unique_subject_ids[train_index]
    test_subject_ids = unique_subject_ids[test_index]

    # Create masks for train and test sets
    train_mask = np.isin(subject_ids, train_subject_ids)
    test_mask = np.isin(subject_ids, test_subject_ids)

    # Create train and test sets using the masks
    X_train, y_train = selected_features[train_mask], all_labels[train_mask]
    X_test, y_test = selected_features[test_mask], all_labels[test_mask]

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
                                                                                                                                                                                                                        
    # Build and train a neural network model
    model = Sequential()
    model.add(Dense(128, input_dim=X_train_scaled.shape[1], activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, validation_data=(X_test_scaled, y_test), verbose=0)

    # Evaluate the model
    y_pred = np.argmax(model.predict(X_test_scaled), axis=1)
    fold_accuracy = accuracy_score(y_test, y_pred)
    fold_accuracies.append(fold_accuracy)

# Print all fold accuracies
print("Fold Accuracies:", fold_accuracies)

# Print average accuracy
average_accuracy = np.mean(fold_accuracies)
print(f"Average Accuracy: {average_accuracy * 100:.2f}%")