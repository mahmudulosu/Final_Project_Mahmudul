import os
import numpy as np
import pandas as pd
import nibabel as nib
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt

# For handling class imbalance
from imblearn.over_sampling import SMOTE

###############################################
# ROI-Based Feature Extraction Functions
###############################################
def load_and_extract_roi_features(file_path):
    """
    Simulated ROI-based feature extraction:
    Divide the MRI volume into 8 regions (by splitting each dimension in half) and compute:
      - Mean intensity per ROI
      - Volume (count of nonzero voxels) per ROI
    This results in 16 features per subject.
    """
    img = nib.load(file_path)
    data = img.get_fdata()

    # Get image dimensions and split each dimension in half
    x, y, z = data.shape
    x_mid, y_mid, z_mid = x // 2, y // 2, z // 2

    # Define 8 ROIs
    rois = [
        (slice(0, x_mid), slice(0, y_mid), slice(0, z_mid)),
        (slice(0, x_mid), slice(0, y_mid), slice(z_mid, z)),
        (slice(0, x_mid), slice(y_mid, y), slice(0, z_mid)),
        (slice(0, x_mid), slice(y_mid, y), slice(z_mid, z)),
        (slice(x_mid, x), slice(0, y_mid), slice(0, z_mid)),
        (slice(x_mid, x), slice(0, y_mid), slice(z_mid, z)),
        (slice(x_mid, x), slice(y_mid, y), slice(0, z_mid)),
        (slice(x_mid, x), slice(y_mid, y), slice(z_mid, z))
    ]

    features = []
    for roi in rois:
        roi_data = data[roi]
        mean_intensity = np.mean(roi_data)
        volume = np.sum(roi_data > 0)  # count nonzero voxels as proxy for volume
        features.extend([mean_intensity, volume])
    
    return np.array(features)

def apply_roi_features_to_dataset(dataset_folder):
    """
    Load all MRI files in a folder and extract ROI-based features.
    """
    from os import listdir
    from os.path import isfile, join

    file_paths = [join(dataset_folder, f) for f in listdir(dataset_folder) if isfile(join(dataset_folder, f))]
    all_features = [load_and_extract_roi_features(fp) for fp in file_paths]
    X_roi = np.stack(all_features, axis=0)
    print("ROI Feature matrix shape:", X_roi.shape)
    return X_roi, file_paths

###############################################
# Visualization Functions
###############################################
def plot_roi_features(X_roi, participant_index, file_paths):
    """
    Plot the ROI features for a specific participant as a bar chart and a scatter plot for the first two features.
    """
    # Bar plot for selected participant
    plt.figure(figsize=(10, 6))
    features = X_roi[participant_index]
    plt.bar(range(len(features)), features)
    plt.title(f'ROI Features for {file_paths[participant_index]}')
    plt.xlabel('Feature Index')
    plt.ylabel('Feature Value')
    plt.grid(True)
    plt.show()

    # Scatter plot for the first two ROI features
    if X_roi.shape[1] >= 2:
        plt.figure(figsize=(8, 6))
        plt.scatter(X_roi[:, 0], X_roi[:, 1], c='blue', label='Participants')
        plt.scatter(X_roi[participant_index, 0], X_roi[participant_index, 1], color='red', label='Selected Participant')
        plt.title('Scatter Plot of the First Two ROI Features')
        plt.xlabel('ROI Feature 1')
        plt.ylabel('ROI Feature 2')
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
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

def plot_roc_curve(y_true, y_scores, title='ROC Curve'):
    """
    Plot the ROC curve and compute the AUC.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
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

def plot_feature_importance(coef, n_top=10):
    """
    Plot the top n feature importances based on the absolute SVM coefficients.
    """
    importances = np.abs(coef).flatten()
    indices = np.argsort(importances)[::-1]
    top_indices = indices[:n_top]
    plt.figure(figsize=(10, 6))
    plt.title("Feature Importances (Top {} ROI Features)".format(n_top))
    plt.bar(range(n_top), importances[top_indices], align="center")
    plt.xticks(range(n_top), [f'Feature {i+1}' for i in top_indices])
    plt.xlabel("ROI Feature Index")
    plt.ylabel("Absolute Coefficient Value")
    plt.grid(True)
    plt.show()

###############################################
# Main Pipeline
###############################################
if __name__ == "__main__":
    # ----------------------------------
    # 1. Load Participant Age Data
    # ----------------------------------
    csv_path = r'C:\research\MRI\participants_LSD_andLEMON.csv'
    age_data = pd.read_csv(csv_path)
    print("Columns in CSV:", age_data.columns)

    # Define valid age ranges and map to age groups
    valid_ages = ["20-25", "25-30", "60-65", "65-70", "70-75"]
    age_group_mapping = {"20-25": "young", "25-30": "young",
                         "60-65": "old", "65-70": "old", "70-75": "old"}
    filtered_data = age_data[age_data['age'].isin(valid_ages)]
    age_mapping = {row['participant_id']: age_group_mapping[row['age']] for _, row in filtered_data.iterrows()}

    # ----------------------------------
    # 2. Process MRI Data with ROI-Based Feature Extraction
    # ----------------------------------
    data_folder = r'C:\research\MRI\structural_MRI'
    roi_features, file_paths = apply_roi_features_to_dataset(data_folder)

    # Match ROI features with age group labels based on participant IDs extracted from filenames
    labels = []
    matched_file_paths = []
    valid_roi_features = []
    for i, fp in enumerate(file_paths):
        subject_id = os.path.basename(fp).split('_')[0]
        if subject_id in age_mapping:
            labels.append(age_mapping[subject_id])
            matched_file_paths.append(fp)
            valid_roi_features.append(roi_features[i])
    labels = np.array(labels)
    valid_roi_features = np.array(valid_roi_features)
    
    # Map labels to binary values: young=0, old=1
    label_mapping = {'young': 0, 'old': 1}
    y = np.array([label_mapping[label] for label in labels])

    # ----------------------------------
    # 3. Split the Data and Handle Imbalance with SMOTE
    # ----------------------------------
    X_train, X_test, y_train, y_test = train_test_split(valid_roi_features, y, 
                                                        test_size=0.2, random_state=42, stratify=y)
    
    # Use SMOTE to oversample the minority class in the training set
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
    # Use StratifiedKFold to maintain class balance in folds
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

    # Compute and plot the Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, classes=['young', 'old'], title='Confusion Matrix')

    # ----------------------------------
    # 7. ROC Curve and Decision Threshold Adjustment
    # ----------------------------------
    # Get probability estimates for the positive class ("old")
    y_scores = best_model.predict_proba(X_test_scaled)[:, 1]
    roc_auc = plot_roc_curve(y_test, y_scores, title='ROC Curve for Age Classification')
    print("ROC AUC:", roc_auc)

    # Optionally adjust the decision threshold from 0.5 to improve recall on the "old" class.
    # For example, set threshold to 0.3:
    threshold = 0.3
    y_pred_adjusted = (y_scores >= threshold).astype(int)
    print("Adjusted Threshold Classification Report:")
    print(classification_report(y_test, y_pred_adjusted))
    cm_adjusted = confusion_matrix(y_test, y_pred_adjusted)
    plot_confusion_matrix(cm_adjusted, classes=['young', 'old'], title='Confusion Matrix (Threshold Adjusted)')

    # ----------------------------------
    # 8. Feature Importance (if using linear kernel)
    # ----------------------------------
    if grid_search.best_params_['kernel'] == 'linear':
        plot_feature_importance(best_model.coef_, n_top=10)
    else:
        print("Feature importance based on coefficients is not available for the RBF kernel.")

    # ----------------------------------
    # 9. Visualization: Scatter Plot of ROI Features
    # ----------------------------------
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(valid_roi_features[:, 0], valid_roi_features[:, 1],
                          c=y, cmap='bwr', edgecolor='k', s=60)
    plt.title('Scatter Plot of First Two ROI Features')
    plt.xlabel('ROI Feature 1')
    plt.ylabel('ROI Feature 2')
    plt.legend(*scatter.legend_elements(), title="Classes")
    plt.grid(True)
    plt.show()

    # ----------------------------------
    # 10. Visualization: ROI Features for a Selected Participant
    # ----------------------------------
    selected_index = 0  # adjust index as needed
    plot_roi_features(valid_roi_features, selected_index, matched_file_paths)
