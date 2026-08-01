RSPECT_dataset_conversion --> Contains scripts to convert the RSPECT annotations to training datasets

patch_extraction --> scripts to convert images and masks to a patch dataset for classification models training

preprocessing --> preprocessing scripts

Install as a library:

python -m pip install -e . 

To launch training:

python -m cteph-lesion-typing.main config.yaml train


How to use:
- Download RSPECT dataset and augmented-RSPECT annotations
- Convert dataset to segmentation
- Extract dataset fingerprint (if normalization is wanted)
- Convert segmentation to patches (check if normalization is set as True if normalization is wanted !)
- Preprocess dataset (normalization)
- Train 


TO DO:
- Make a preprocessing script based on the nnUNet one and check if it works properly
