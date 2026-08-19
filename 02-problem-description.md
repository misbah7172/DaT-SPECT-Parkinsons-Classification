# DaT Parkinson's Challenge — Problem Description

Source: https://www.drivendata.org/competitions/311/dat-parkinsons-challenge/page/990/

Your goal is to develop machine learning models that predict the probability that a dopamine transporter (DaT) scan examination is **abnormal** rather than **normal**.

In this challenge, you will build models that distinguish normal from abnormal DaT scans using a large multicenter dataset collected and annotated by French experts. Successful solutions should generalize across institutions, scanners, and patient populations, and could help improve the consistency, accessibility, and scalability of DaT scan interpretation.

## Dataset

The imaging data are provided as three-dimensional DaT scan reconstructions in compressed Neuroimaging Informatics Technology Initiative (NIfTI) format (`.nii.gz`). Each examination is a single 3D volume, and each patient is represented by one file. The filename (minus the `.nii.gz` extension) is the `uid` that links each image to its label.

In addition, participants are welcome to incorporate external datasets for training; however, use of external data is subject to important exceptions and caveats (see External Data section on the Home page, and the full Rules).

Participants are not allowed to share the competition data or use it for any purpose other than this competition. Participants cannot send the data to any third-party service or API, including, but not limited to, OpenAI's ChatGPT, Google's Gemini, or similar tools. For complete details, review the competition rules.

### Files

```
data/
├── niftis/
│   ├── xaji0y6d.nii.gz
│   ├── pbhsahxt.nii.gz
│   └── ...
└── train\_labels.csv
```

### Images

Each `.nii.gz` file contains a single 3D reconstructed volume stored as 16-bit unsigned integer voxel intensities. Note that acquisition and reconstruction parameters vary across the dataset, so images are **not** all the same size or resolution:

* **Volume dimensions** differ from scan to scan (for example, `142 × 142 × 112` or `128 × 128 × 128`). Your pipeline should not assume a fixed input shape.
* **Voxel spacing** also varies (e.g., 2.46 mm and 3.895 mm isotropic), reflecting differences in scanners and acquisition protocols across contributing centers. The spacing is recorded in each file's NIfTI header and may be useful for resampling images to a common resolution.

These differences are a normal consequence of pooling data from multiple institutions, and building models that generalize across them is part of the challenge.

### Image example

Example DaT scan examination image (several axial slices moving through the volume):
https://drivendata-public-assets.s3.amazonaws.com/dat\_example.png

Note: Each file is a single 3D reconstruction, and dopamine transporter uptake is concentrated in the striatum near the center of the brain. Examinations vary in size and voxel spacing across scanners and centers, so the number and appearance of slices will differ from scan to scan.

### Labels

`train\_labels.csv` contains one row per DaT scan examination in the training set, with the following columns:

* `uid` (str) — unique identifier for each DaT scan examination; matches the image filename without the `.nii.gz` extension
* `is\_pathologic` (float) — classification of the DaT scan examination, where `0.0 = Normal` and `1.0 = abnormal`

Label example — first five rows of `train\_labels.csv`:

|uid|is\_pathologic|
|-|-|
|xaji0y6d|0.0|
|pbhsahxt|0.0|
|hv3a3zmf|0.0|
|8mdd4v30|0.0|
|t9nt3w5u|1.0|

### Test set

The test set is withheld and is not available for download. Its images are only accessible from within the runtime container, where they are mounted alongside the training data. Because this is a code execution challenge, you will not see the test examinations directly — your submitted code reads them at inference time and generates predictions for each one.

The test examinations are the same NIfTI format as the training data and follow the same conventions (one 3D volume per file, named `<uid>.nii.gz`, with varying dimensions and voxel spacing).

## Performance metric

**Leaderboard performance is evaluated according to log loss.** Log loss (a.k.a. logistic loss or cross-entropy loss) penalizes confident but incorrect predictions. It also rewards confidence scores that are well-calibrated probabilities, meaning that they accurately reflect the long-run probability of being correct. This is an error metric, so a lower value is better.

Log loss for a single observation is calculated as follows:

$$L\_{\\log}(y, p) = -(y \\log (p) + (1 - y) \\log (1 - p))$$

where $y$ is a binary variable indicating whether the examination is abnormal (1) or normal (0), and $p$ is the user-predicted probability that the examination is abnormal. The loss for the entire dataset is the average loss across all observations.

Note: log loss can often be improved with calibration (https://scikit-learn.org/stable/modules/calibration.html). A well-calibrated model outputs predictions that are directly interpretable as probabilities.

The leaderboard also displays Area Under the Receiver Operating Characteristic (AUROC) for reference (https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc\_auc\_score.html), but this metric does *not* affect leaderboard ranking or prizes.

## Submission format

This is a code execution challenge. Rather than submitting predictions directly, participants will package their trained model and inference code for containerized execution.

Your code must read the test examinations and produce a `submission.csv` with one row per examination:

* `uid` (str) — unique identifier for each DaT scan examination
* `is\_pathologic` (float) — your predicted probability that the examination is abnormal, as a value between 0 and 1

Because performance is evaluated with log loss, submit calibrated probabilities rather than discrete `0`/`1` labels.

Additional information about the runtime environment, package structure, and submission requirements can be found on the Code Submission Format page.

