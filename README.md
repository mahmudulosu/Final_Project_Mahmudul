# Multimodal Age‑Group Classifier (EEG + MRI + CLIP)

This repository trains three binary age‑group classifiers  
(EEG‑only, MRI‑only, and EEG + MRI fusion) using a partially fine‑tuned
[`openai/clip‑vit‑base‑patch32`](https://huggingface.co/openai/clip-vit-base-patch32) backbone.
eeg_dir  = r"C:/research/EEG_Domain/scalograms_224x224x17"
replace:"C:\Users\mdmahha\OneDrive - Oklahoma A and M System\Dataset_Deep_learning_project\scalograms_224x224x17"

mri_dir  = r"C:/research/MRI/structural_MRI"
replace:"C:\Users\mdmahha\OneDrive - Oklahoma A and M System\Dataset_Deep_learning_project\structural_MRI"

csv_path = r"C:/research/MRI/participants_LSD_andLEMON.csv"
replace:"C:\Users\mdmahha\OneDrive - Oklahoma A and M System\Dataset_Deep_learning_project\participants_LSD_andLEMON.csv"

Run:Age_classification_clip.py

# MRI Age‑Group Classifier (FreeSurfer‑like Features + SVM)

This project trains a binary classifier (young vs old) on structural MRI
volumes using a **very light, FreeSurfer‑inspired feature set** and a
support‑vector machine with class‑imbalance handling (SMOTE).

python mri_age_svm.py \
    --csv  "C:/research/MRI/participants_LSD_andLEMON.csv" \
    --mri  "C:/research/MRI/structural_MRI"

Run:Age_classification_cortical_thickness.py
# Multimodal Age‑Group Classification  
EEG (EO & EC scalograms) • Structural MRI • ResNet‑50/101/152

This project trains three flavours of binary classifiers:
Run:Age_classification_resnet.py

# EEG Age‑Group Classifier  
Discrete‑Wavelet Transform • Statistical Features • MLP • 10‑fold Subject CV

This project classifies participants into two age groups (young vs old) using
eyes‑open (EO) and eyes‑closed (EC) resting‑state EEG recordings.

Run:age_classification_wavelet.py

# EEG Fatigue Classifier  
Discrete‑Wavelet Features • 5‑Second Segments • Balanced MLP

This project predicts acute mental‑fatigue (fatigued vs alert) from resting‑state
EEG recordings using Discrete Wavelet Transform features and a fully connected
neural network.

Run:awake_sleep_classification.py

# Multimodal Gender Classifier  
Eyes‑Closed EEG (17‑channel scalograms) • Structural MRI • Fine‑tuned CLIP
Run:eyecloasedata_datafusion_gender.py
# MRI Gender Classifier  
FreeSurfer‑Like 4‑Feature Vector • SMOTE • SVM + Grid Search

This lightweight pipeline predicts participant gender from structural MRI
volumes using four brain‑volume proxies inspired by FreeSurfer outputs.

Run:free_surfer_mri_gender.py

# MRI Age‑Group Classifier  
Simple 8‑Region ROI Features • SMOTE • SVM Grid Search

This pipeline classifies participants into two age groups (young vs old)
using **16 handcrafted features** extracted from eight coarse brain ROIs.

Run:Age_classification_age_surfer.py

# Multimodal Gender Classifier  
17‑Channel EEG Scalograms • Structural MRI • Fine‑Tuned CLIP ViT‑B/32

This project trains three binary classifiers to predict participant gender:

Run:Gender_classification_Clip.py

# Multimodal Gender Classifier  
Eyes‑Open & Eyes‑Closed EEG Scalograms • Structural MRI • Fine‑Tuned ResNet‑50/101/152

This project trains three binary classifiers to predict participant gender:
Run:Gender_Classification_Resnet.py

# EEG Gender Classifier  
Eyes‑Closed (or Eyes‑Open) EEG • Discrete Wavelet Transform • SVM

This pipeline predicts participant gender (male vs female) from resting‑state
EEG recordings using **DWT features** and a linear Support‑Vector Machine.
Run:Gender_classification_wavelet_eyeopen.py

# MRI Age‑Group Classifier  
3‑D ResNet‑18 (Video Model Zoo) • Young vs Old

This project predicts whether a participant belongs to the **young** (20 – 30 y)
or **old** (60 – 75 y) age group using a single structural MRI volume.
The network fine‑tunes the *r3d_18* architecture from
`torchvision.models.video` on resampled 128³ voxels.
![output (30)](https://github.com/user-attachments/assets/5a17a674-ea13-4696-8cc9-76506cc37547)

Run:only_mri_age_resnet.py

# MRI Age‑Group Classifier  
Whole‑Brain PCA • Linear SVM • Young vs Old

This project classifies structural MRI scans into **young** (20‑30 y) versus
**old** (60‑75 y) using a very light pipeline:

Run:PCA_age_classification.py

# EEG Fatigue Classifier  
17‑Channel 224×224 Scalograms • Fine‑Tuned ResNet‑50 • Extreme‑Groups Strategy

This project predicts **acute mental fatigue** (“tired” vs “awake”) from
EEG scalograms.  Each scalogram is a 224 × 224 × 17 tensor
(17 frequency bins).  
run:sleep_awake_30sec_scalogram_resnet.py

# EEG Fatigue Classifier  
17‑Channel 224 × 224 Scalograms • Fine‑Tuned CLIP (ViT‑B/32) • Extreme‑Group MDBF Labels

This project predicts **acute mental fatigue** (“tired” vs “awake”) from
EEG scalograms by fine‑tuning the vision branch of **CLIP ViT‑B/32**:
Run:sleep_awake_clip.py
