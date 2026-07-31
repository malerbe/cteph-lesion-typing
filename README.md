RSPECT_dataset_conversion --> Contains scripts to convert the RSPECT annotations to training datasets

patch_extraction --> scripts to convert images and masks to a patch dataset for classification models training

Install as a library:

python -m pip install -e . 

To launch training:

python -m cteph-lesion-typing.main config.yaml train
