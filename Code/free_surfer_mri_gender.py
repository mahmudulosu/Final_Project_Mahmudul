import os
import numpy as np
import pandas as pd
import nibabel as nib
from scipy.ndimage import binary_erosion
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt

# For handling class imbalance
from imblearn.over_sampling import SMOTE

###############################################
# FreeSurfer-like Feature Extraction Functions
###############################################
def load_and_extract_fs_features(file_path):
    """
    Simulate extraction of FreeSurfer-like features from an MRI scan.
    
    Features:
    - Cortical Thickness Proxy: Proportion of voxels in the cortical shell.
    - Left Subcortical Volume: Count of voxels in the left half of the eroded (inner) brain mask.
    - Right Subcortical Volume: Count of voxels in the right half of the eroded brain mask.
    - Total Brain Volume: Count of nonzero voxels in the brain mask.
    
    The function assumes that nonzero voxels belong to the brain.
    """
    img = nib.load(file_path)
    data = img.get_fdata()
    
    # Create a brain mask (assume nonzero voxels belong to the brain)
    brain_mask = data > 0
    
    # Compute total brain volume (number of voxels)
    total_brain_vol = np.sum(brain_mask)
    
    # Compute an eroded mask to approximate the inner (subcortical) brain
    eroded_mask = binary_erosion(brain_mask, iterations=2)
    
    # Cortical mask: voxels that were in the brain but removed by erosion
    cortical_mask = brain_mask & ~eroded_mask
    cortical_prop = np.sum(cortical_mask) / total_brain_vol if total_brain_vol > 0 else 0
    
    # For subcortical volumes, split the eroded mask into left and right halves along the x-axis
    x, y, z = brain_mask.shape
    left_mask = eroded_mask[:x//2, :, :]
    right_mask = eroded_mask[x//2:, :, :]
    left_subcortical_vol = np.sum(left_mask)
    right_subcortical_vol = np.sum(right_mask)
    
    # Feature vector: [cortical thickness proxy, left subcortical vol, right subcortical vol, total brain vol]
    features = np.array([cortical_prop, left_subcortical_vol, right_subcortical_vol, total_brain_vol])
    return features

def apply_fs_features_to_dataset(dataset_folder):
    """
    Load all MRI files in a folder and extract FreeSurfer-like features.
    """
    from os import listdir
    from os.path import isfile, join

    file_paths = [join(dataset_folder, f) for f in listdir(dataset_folder) if isfile(join(dataset_folder, f))]
    all_features = [load_and_extract_fs_features(fp) for fp in file_paths]
    X_fs = np.stack(all_features, axis=0)
    print("FreeSurfer-like Feature matrix shape:", X_fs.shape)
    return X_fs, file_paths

###############################################
# Visualization Functions
###############################################
def plot_feature_values(X_features, participant_index, file_paths):
    """
    Plot the extracted features for a specific participant as a bar chart and a scatter plot for the first two features.
    """
    # Bar plot for the selected participant
    plt.figure(figsize=(10, 6))
    features = X_features[participant_index]
    plt.bar(range(len(features)), features)
    plt.title(f'Extracted Features for {file_paths[participant_index]}')
    plt.xlabel('Feature Index')
    plt.ylabel('Feature Value')
    plt.grid(True)
    plt.show()

    # Scatter plot for the first two features
    if X_features.shape[1] >= 2:
        plt.figure(figsize=(8, 6))
        plt.scatter(X_features[:, 0], X_features[:, 1], c='blue', label='Participants')
        plt.scatter(X_features[participant_index, 0], X_features[participant_index, 1], 
                    color='red', label='Selected Participant')
        plt.title('Scatter Plot of the First Two Extracted Features')
        plt.xlabel('Feature 1 (Cortical Thickness Proxy)')
        plt.ylabel('Feature 2 (Left Subcortical Volume)')
        plt.legend()
        plt.grid(True)
        plt.show()

def plot_confusion_matrix(cm, classes, title='Confusion Matrix'):
    """
    Plot the confusion matrix as a heatmap.
    """
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    thresh = cm.max() / 2.0
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], 'd'),
                 ha="center", va="center",
                 color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

def plot_roc_curve(y_true, y_scores, title='ROC Curve'):
    """
    Plot the ROC curve and compute the AUC.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()
    return roc_auc

def plot_feature_importance(coef, n_top=4):
    """
    Plot the top n feature importances based on the absolute SVM coefficients.
    (Here, we only have 4 features.)
    """
    importances = np.abs(coef).flatten()
    indices = np.argsort(importances)[::-1]
    top_indices = indices[:n_top]
    plt.figure(figsize=(10, 6))
    plt.title("Feature Importances (Top {} Features)".format(n_top))
    plt.bar(range(n_top), importances[top_indices], align="center")
    plt.xticks(range(n_top), [f'Feature {i+1}' for i in top_indices])
    plt.xlabel("Feature Index")
    plt.ylabel("Absolute Coefficient Value")
    plt.grid(True)
    plt.show()

###############################################
# Main Pipeline for Gender Classification
###############################################
if __name__ == "__main__":
    # ----------------------------------
    # 1. Load Participant Gender Data
    # ----------------------------------
    csv_path = r'C:\research\MRI\participants_LSD_andLEMON.csv'
    data = pd.read_csv(csv_path)
    print("Columns in CSV:", data.columns)
    
    # Create a mapping from participant ID to gender (assumes 'gender' column contains "M" and "F")
    gender_mapping = {row['participant_id']: row['gender'] for _, row in data.iterrows() if row['gender'] in ['M', 'F']}
    
    # Define label mapping for gender: Female = 0, Male = 1
    label_mapping = {'F': 0, 'M': 1}
    
    # ----------------------------------
    # 2. Process MRI Data with FS-like Feature Extraction
    # ----------------------------------
    data_folder = r'C:\research\MRI\structural_MRI'
    fs_features, file_paths = apply_fs_features_to_dataset(data_folder)
    
    # Match extracted features with gender labels based on participant IDs in filenames
    labels = []
    matched_file_paths = []
    valid_fs_features = []
    for i, fp in enumerate(file_paths):
        subject_id = os.path.basename(fp).split('_')[0]
        if subject_id in gender_mapping:
            labels.append(gender_mapping[subject_id])
            matched_file_paths.append(fp)
            valid_fs_features.append(fs_features[i])
    labels = np.array(labels)
    valid_fs_features = np.array(valid_fs_features)
    
    # Map gender to binary values (Female = 0, Male = 1)
    y = np.array([label_mapping[label] for label in labels])
    
    # ----------------------------------
    # 3. Split the Data and Handle Imbalance with SMOTE
    # ----------------------------------
    X_train, X_test, y_train, y_test = train_test_split(valid_fs_features, y, 
                                                        test_size=0.2, random_state=42, stratify=y)
    
    # Apply SMOTE to balance the training data
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    print("After SMOTE, training set class distribution:", np.bincount(y_train_res))
    
    # ----------------------------------
    # 4. Scale the Data
    # ----------------------------------
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_res)
    X_test_scaled = scaler.transform(X_test)
    
    # ----------------------------------
    # 5. Hyperparameter Tuning with Grid Search (SVM)
    # ----------------------------------
    param_grid = {
        'C': [0.01, 0.1, 1, 10, 100],
        'kernel': ['linear', 'rbf']
    }
    svm_clf = SVC(probability=True, class_weight='balanced', random_state=42)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(svm_clf, param_grid, cv=skf, scoring='accuracy')
    grid_search.fit(X_train_scaled, y_train_res)
    
    print("Best Parameters:", grid_search.best_params_)
    best_model = grid_search.best_estimator_
    
    # ----------------------------------
    # 6. Model Evaluation
    # ----------------------------------
    y_pred = best_model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    print("Classification Report:\n", report)
    
    # Compute and plot the Confusion Matrix (labels: Female, Male)
    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, classes=['Female', 'Male'], title='Confusion Matrix for Gender Classification')
    
    # ----------------------------------
    # 7. ROC Curve and Decision Threshold Adjustment
    # ----------------------------------
    # Here, we use the probability estimates for the "Male" class (label 1)
    y_scores = best_model.predict_proba(X_test_scaled)[:, 1]
    roc_auc = plot_roc_curve(y_test, y_scores, title='ROC Curve for Gender Classification')
    print("ROC AUC:", roc_auc)
    
    # Optionally adjust decision threshold (example: threshold = 0.3)
    threshold = 0.3
    y_pred_adjusted = (y_scores >= threshold).astype(int)
    print("Adjusted Threshold Classification Report:")
    print(classification_report(y_test, y_pred_adjusted))
    cm_adjusted = confusion_matrix(y_test, y_pred_adjusted)
    plot_confusion_matrix(cm_adjusted, classes=['Female', 'Male'], title='Confusion Matrix (Threshold Adjusted)')
    
    # ----------------------------------
    # 8. Feature Importance (if using a linear kernel)
    # ----------------------------------
    if grid_search.best_params_['kernel'] == 'linear':
        plot_feature_importance(best_model.coef_, n_top=4)
    else:
        print("Feature importance based on coefficients is not available for the RBF kernel.")
    
    # ----------------------------------
    # 9. Visualization: Scatter Plot of Extracted Features
    # ----------------------------------
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(valid_fs_features[:, 0], valid_fs_features[:, 1],
                          c=y, cmap='bwr', edgecolor='k', s=60)
    plt.title('Scatter Plot of First Two Extracted Features')
    plt.xlabel('Feature 1 (Cortical Thickness Proxy)')
    plt.ylabel('Feature 2 (Left Subcortical Volume)')
    plt.legend(*scatter.legend_elements(), title="Gender")
    plt.grid(True)
    plt.show()
    
    # ----------------------------------
    # 10. Visualization: Feature Values for a Selected Participant
    # ----------------------------------
    selected_index = 0  # adjust as needed
    plot_feature_values(valid_fs_features, selected_index, matched_file_paths)
