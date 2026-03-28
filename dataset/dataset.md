# About Dataset
## Context
BraTS has always been focusing on the evaluation of state-of-the-art methods for the segmentation of brain tumors in multimodal magnetic resonance imaging (MRI) scans. BraTS 2020 utilizes multi-institutional pre-operative MRI scans and primarily focuses on the segmentation (Task 1) of intrinsically heterogeneous (in appearance, shape, and histology) brain tumors, namely gliomas. Furthemore, to pinpoint the clinical relevance of this segmentation task, BraTS’20 also focuses on the prediction of patient overall survival (Task 2), and the distinction between pseudoprogression and true tumor recurrence (Task 3), via integrative analyses of radiomic features and machine learning algorithms. Finally, BraTS'20 intends to evaluate the algorithmic uncertainty in tumor segmentation (Task 4).

## Tasks' Description and Evaluation Framework
In this year's challenge, 4 reference standards are used for the 4 tasks of the challenge:

## Manual segmentation labels of tumor sub-regions,
1. Clinical data of overall survival,
2. Clinical evaluation of progression status,
3. Uncertainty estimation for the predicted tumor sub-regions.
4. Imaging Data Description
All BraTS multimodal scans are available as NIfTI files (.nii.gz) and describe a) native (T1) and b) post-contrast T1-weighted (T1Gd), c) T2-weighted (T2), and d) T2 Fluid Attenuated Inversion Recovery (T2-FLAIR) volumes, and were acquired with different clinical protocols and various scanners from multiple (n=19) institutions, mentioned as data contributors here.

All the imaging datasets have been segmented manually, by one to four raters, following the same annotation protocol, and their annotations were approved by experienced neuro-radiologists. Annotations comprise the GD-enhancing tumor (ET — label 4), the peritumoral edema (ED — label 2), and the necrotic and non-enhancing tumor core (NCR/NET — label 1), as described both in the BraTS 2012-2013 TMI paper and in the latest BraTS summarizing paper. The provided data are distributed after their pre-processing, i.e., co-registered to the same anatomical template, interpolated to the same resolution (1 mm^3) and skull-stripped.


## Details
Stored in D:\Datasets\BraTS2020_training_data\content\data
Volumes filenames: volume_1_slice_0.h5 All are stored within one data folder
Survival Info "D:\Datasets\BraTS2020_training_data\content\data\survival_info.csv"
Name Mapping "D:\Datasets\BraTS2020_training_data\content\data\name_mapping.csv"
Meta Data "D:\Datasets\BraTS2020_training_data\content\data\meta_data.csv"