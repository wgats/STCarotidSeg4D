# STCarotidSeg4D

Spatio-temporal carotid artery segmentation for 4D Flow MRI.

## Usage

### Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the packages in `requirements.txt`.

```bash
pip install -r requirements.txt
```
### Configurations

The following models are available in `models.py`:
- 2D U-Net - `unet2d`
- 3D U-Net - `unet3d`
- UNETR `unetr`
- Spatio-Temporal Transformer  `spatio_temporal_transformer`

The following temporal dimension processing strategies (transforms) are available:
- Interpolation - `interpolate`
- Padding with cropping - `pad-crop`
- Padding with averaging - `pad-average`

Image sizes, keys for transforms and training parameters can be changed in `constants.py`. Images can either have 1 (magnitude) or 3 channels (magnitude + velocities).

### Training

To train individual folds for cross-validation

```bash
python train.py -dp <DATASET FOLDER> -o <OUTPUT FOLDER> -m <MODEL> -t <TRANSFORM> -c <CHANNELS> -d <DEPTH> -f <FOLD> 
```
A `splits_final.json` file specifying splits in the nnU-Net convention [[1]](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/dataset_format.md) is required for cross-validation, and should be placed in the dataset folder.

### Validation

To evalate trained models on all folds of the validation data:

```bash
python validation.py -m <MODEL FOLDER> -n <MODEL> -d <DATASET FOLDER> -o <OUTPUT FOLDER> -c <CHANNELS> -i <DEPTH> -t <TRANSFORM> -p
```

A `splits_final.json` file specifying splits in the nnU-Net convention [[1]](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/dataset_format.md) is required, and should be placed in the dataset folder.

### Inference

To perform ensemble inference with majority voting:

```bash
python ensemble.py -m <MODEL FOLDER> -o <OUTPUT FOLDER> -n <MODEL> -i <IMAGE FOLDER> -c <CHANNELS> -d <DEPTH> -t <TRANSFORM> -p
```

A `splits_final.json` file specifying splits in the nnU-Net convention [[1]](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/dataset_format.md) is required, and should be placed in the dataset folder.

To perform inference with a single model:

```bash
python inference.py -i <IMAGE FOLDER> -o <OUTPUT FOLDER> -m <MODEL CHECKPOINT PATH> -c <CHANNELS> -d <DEPTH> -t <TRANSFORM> -p
```

To save predictions without temporal post-processing discard the `-p`.

### Evaluation

To evaluate predictions:

```bash
python evaluation.py -i <IMAGE FOLDER> -gt<GROUNT TRUTH LABELS> -s <SAVE FOLDER> -p <PREDICTED LABELS> -v <VELOCITY IMAGE FOLDER> -p
```

If evaluating predictions that have not been post-processed, omit `-p`. Velocity images are required for velocity based metrics.

## References

[1] Isensee F. nnU-Net dataset format https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/dataset_format.md (Accessed 19 June 2025)

## License

The code in this repository is licensed under the GNU-GPL license. 