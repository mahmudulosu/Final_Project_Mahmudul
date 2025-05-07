import os
import numpy as np
import pandas as pd
import nibabel as nib
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from nilearn.image import resample_img
import matplotlib.pyplot as plt

def load_and_flatten_mri(file_path):
    """
    Load an MRI file, resample it to 2mm isotropic resolution,
    and flatten the 3D data into a 1D array.
    """
    img = nib.load(file_path)
    target_affine = np.eye(3) * 2.0  # Rescale voxels to 2mm isotropic
    img_resampled = resample_img(img, target_affine=target_affine)
    data = img_resampled.get_fdata().flatten()
    return data

def apply_pca_to_dataset(dataset_folder, n_components=150):
    """
    Load all MRI files in a folder, flatten them, and apply PCA for dimensionality reduction.
    """
    from os import listdir
    from os.path import isfile, join
    
    file_paths = [join(dataset_folder, f) for f in listdir(dataset_folder) if isfile(join(dataset_folder, f))]
    # Load and flatten each MRI scan
    all_data = [load_and_flatten_mri(fp) for fp in file_paths]
    X = np.stack(all_data, axis=0)
    
    # Apply PCA to reduce dimensionality
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    print("PCA Components shape:", X_pca.shape)
    return X_pca, pca, file_paths

def plot_pca_components(X_pca, pca, participant_index, file_paths):
    """
    Plot the PCA components for a specific participant as a line plot,
    and also display a scatter plot of the first two PCA components for all subjects.
    """
    # Line plot for the selected participant
    plt.figure(figsize=(10, 6))
    plt.plot(X_pca[participant_index], marker='o')
    plt.title(f'PCA Components for {file_paths[participant_index]}')
    plt.xlabel('Component Number')
    plt.ylabel('PCA Value')
    plt.grid(True)
    plt.show()

    # Scatter plot for the first two PCA components
    if X_pca.shape[1] >= 2:
        plt.figure(figsize=(8, 6))
        plt.scatter(X_pca[:, 0], X_pca[:, 1], c='blue', label='Participants')
        plt.scatter(X_pca[participant_index, 0], X_pca[participant_index, 1], color='red', label='Selected Participant')
        plt.title('Scatter Plot of the First Two PCA Components')
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
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
    Plot the ROC curve and calculate the AUC.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line for random classifier
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
    Plot the top n feature importances based on the absolute value of the SVM coefficients.
    """
    importances = np.abs(coef).flatten()
    indices = np.argsort(importances)[::-1]
    top_indices = indices[:n_top]
    plt.figure(figsize=(10, 6))
    plt.title("Feature Importances (Top %d PCA Components)" % n_top)
    plt.bar(range(n_top), importances[top_indices], align="center")
    plt.xticks(range(n_top), [f'Component {i+1}' for i in top_indices])
    plt.xlabel("PCA Component")
    plt.ylabel("Absolute Coefficient Value")
    plt.grid(True)
    plt.show()

# ===============================
# Main Pipeline
# ===============================
if __name__ == "__main__":
    # Load age information from CSV
    csv_path = r'C:\research\MRI\participants_LSD_andLEMON.csv'
    age_data = pd.read_csv(csv_path)
    print("Columns in CSV:", age_data.columns)

    # Define valid age ranges and create age group mappings
    valid_ages = ["20-25", "25-30", "60-65", "65-70", "70-75"]
    age_group_mapping = {"20-25": "young", "25-30": "young",
                         "60-65": "old", "65-70": "old", "70-75": "old"}
    filtered_data = age_data[age_data['age'].isin(valid_ages)]
    # Create a mapping from participant ID to age group
    age_mapping = {row['participant_id']: age_group_mapping[row['age']] for _, row in filtered_data.iterrows()}

    # Apply PCA to the MRI dataset
    data_folder = r'C:\research\MRI\structural_MRI'
    n_components = 50  # You can adjust the number of PCA components as needed
    pca_components, pca, file_paths = apply_pca_to_dataset(data_folder, n_components=n_components)

    # Match PCA components with age group labels based on participant IDs extracted from filenames
    labels = []
    matched_file_paths = []
    valid_pca_components = []
    for i, fp in enumerate(file_paths):
        subject_id = os.path.basename(fp).split('_')[0]
        if subject_id in age_mapping:
            labels.append(age_mapping[subject_id])
            matched_file_paths.append(fp)
            valid_pca_components.append(pca_components[i])
    labels = np.array(labels)
    valid_pca_components = np.array(valid_pca_components)
    
    # Map age groups to binary values (e.g., 'young': 0, 'old': 1)
    label_mapping = {'young': 0, 'old': 1}
    y = np.array([label_mapping[label] for label in labels])

    # Split data into training and testing sets (80/20 split)
    X_train, X_test, y_train, y_test = train_test_split(valid_pca_components, y, test_size=0.2, random_state=42)

    # Scale the features using StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train an SVM classifier with a linear kernel and probability estimates enabled
    svm_clf = SVC(kernel='linear', probability=True, random_state=42)
    svm_clf.fit(X_train_scaled, y_train)

    # Make predictions on the test set and evaluate the model
    y_pred = svm_clf.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    print("Classification Report:\n", report)

    # Compute and plot the Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, classes=['young', 'old'], title='Confusion Matrix')

    # Plot the ROC Curve and calculate the AUC using probabilities for the "old" class
    y_scores = svm_clf.predict_proba(X_test_scaled)[:, 1]
    roc_auc = plot_roc_curve(y_test, y_scores, title='ROC Curve for Age Classification')
    print("ROC AUC:", roc_auc)

    # Plot the feature importances based on the SVM coefficients (only available for linear SVM)
    plot_feature_importance(svm_clf.coef_, n_top=10)

    # Scatter plot of the first two PCA components for all subjects,
    # coloring points by their label (young=0, old=1)
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(valid_pca_components[:, 0], valid_pca_components[:, 1],
                          c=y, cmap='bwr', edgecolor='k', s=60)
    plt.title('Scatter Plot of First Two PCA Components')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.legend(*scatter.legend_elements(), title="Classes")
    plt.grid(True)
    plt.show()

    # Plot PCA components for the first participant in the test set
    selected_index = 0
    plot_pca_components(valid_pca_components, pca, selected_index, matched_file_paths)
