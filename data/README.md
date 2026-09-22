# Data

The project uses the mass cases of CBIS-DDSM, the Curated Breast Imaging Subset of the
Digital Database for Screening Mammography, published by The Cancer Imaging Archive
(DOI 10.7937/K9/TCIA.2016.7O02S9CY). The images are not redistributed here: download them
from The Cancer Imaging Archive and place them as described below.

The code reads everything from `data/raw/`; the paths are set in `src/config.py`:

```
data/raw/
  mass_case_description_train_set.csv
  mass_case_description_test_set.csv
  mass_train/images/        full mammograms of the training cases
  mass_test/images/         full mammograms of the test cases
  mass_train_roi/images/    lesion masks and cropped patches of the training cases
  mass_test_roi/images/     lesion masks and cropped patches of the test cases
```

The two CSVs are the dataset's case-description files for the mass training and test
cases, and their file-path columns decide where the loaders look. Each full mammogram has
its own folder under `images/`, named as the first component of the image file path in the
CSV (for example `Mass-Training_P_00001_LEFT_CC`), with the subfolders that path names
below it; its one DICOM sits in the innermost of those subfolders, not directly in the top
folder. Each lesion has its own folder under the ROI `images/` directory, named as the
first component of its ROI file paths (for example `Mass-Training_P_00001_LEFT_CC_1`),
holding two DICOMs: the binary mask and the cropped patch. The loader tells those two
apart by image size, the mask being the larger. A lesion folder holding only one DICOM
gives only the cropped patch; the default crop path also needs the mask, so it skips that
lesion. File names are never relied on, so the DICOMs can keep the names they are
downloaded with.

The dataset's training and test cases are pooled before the project draws its own
patient-level partition, so neither `mass_test` folder is the project's test set.

Everything under `data/` except this README is ignored by git.
