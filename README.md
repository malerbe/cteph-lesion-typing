RSPECT_dataset_conversion --> Contains scripts to convert the RSPECT annotations to training datasets

patch_extraction --> scripts to convert images and masks to a patch dataset for classification models training

preprocessing --> preprocessing scripts

Install as a library:

python -m pip install -e . 

To launch training:

python -m cteph-lesion-typing.main config.yaml train


How to use:
- Download RSPECT dataset and augmented-RSPECT annotations
- Convert dataset to segmentation (RSPECT_to_seg.py)
- Extract dataset fingerprint (if normalization is wanted) (extract_fingerprint.py)
- Convert segmentation to patches (check if normalization is set as True if normalization is wanted !) (lesion_patches_extraction.py)
- Create splitting folds (split_patch_dataset.py)
- Write a configuration file (config.yaml)
- Train 


TO DO:
- Make a preprocessing script based on the nnUNet one and check if it works properly
- Check what normalization is used on VoCo and replicate it. 
- Make heart_patches_extraction.py script that will use TotalSegmentator's output and get a bbox from it + apply the right Normalization for VoCo. 
