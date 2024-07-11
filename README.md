# LDDMM meets GANs: Generative Adversarial Networks for diffeomorphic registration

## Prerequisites
- `Python 3.6`
- `Tensorflow 2.6.2 (GPU)`
- `NumPy`
- `scipy`
- `NiBabel`

# Instructions

To use the VoxelMorph library, either clone this repository and install the requirements listed in `setup.py` or install directly with pip.

```
pip install voxelmorph
```

## Pre-trained models

See list of pre-trained models available [here](data/readme.md#models).

## Training

If you would like to train your own model, you will likely need to customize some of the data-loading code in `voxelmorph/generators.py` for your own datasets and data formats. However, it is possible to run many of the example scripts out-of-the-box, assuming that you provide a list of filenames in the training dataset. Training data can be in the NIfTI, MGZ, or npz (numpy) format, and it's assumed that each npz file in your data list has a `vol` parameter, which points to the image data to be registered, and an optional `seg` variable, which points to a corresponding discrete segmentation (for semi-supervised learning). It's also assumed that the shape of all training image data is consistent, but this, of course, can be handled in a customized generator if desired.

For a given image list file `/images/list.txt` and output directory `/models/output`, the following script will train an image-to-image registration network (described in MICCAI 2018 by default) with an unsupervised loss. Model weights will be saved to a path specified by the `--model-dir` flag.

```
./scripts/tf/train.py --img-list /images/list.txt --model-dir /models/output --gpu 0
```

The `--img-prefix` and `--img-suffix` flags can be used to provide a consistent prefix or suffix to each path specified in the image list. Image-to-atlas registration can be enabled by providing an atlas file, e.g. `--atlas atlas.npz`. If you'd like to train using the original dense CVPR network (no diffeomorphism), use the `--int-steps 0` flag to specify no flow integration steps. Use the `--help` flag to inspect all of the command line options that can be used to fine-tune network architecture and training.


## Registration

If you simply want to register two images, you can use the `register.py` script with the desired model file. For example, if we have a model `model.h5` trained to register a subject (moving) to an atlas (fixed), we could run:

```
./scripts/tf/register.py --moving moving.nii.gz --fixed atlas.nii.gz --moved warped.nii.gz --model model.h5 --gpu 0
```

This will save the moved image to `warped.nii.gz`. To also save the predicted deformation field, use the `--save-warp` flag. Both npz or nifty files can be used as input/output in this script.

# Reference Paper

If you use...
# Notes
- **keywords**: machine learning, convolutional neural networks, alignment, mapping, registration  
- **data in papers**: 
In our initial papers, we used publicly available data, but unfortunately we cannot redistribute it (due to the constraints of those datasets). We do a certain amount of pre-processing for the brain images we work with, to eliminate sources of variation and be able to compare algorithms on a level playing field. In particular, we perform FreeSurfer `recon-all` steps up to skull stripping and affine normalization to Talairach space, and crop the images via `((48, 48), (31, 33), (3, 29))`. 

We encourage users to download and process their own data. See [a list of medical imaging datasets here](https://github.com/adalca/medical-datasets). Note that you likely do not need to perform all of the preprocessing steps, and indeed VoxelMorph has been used in other work with other data.


# Data
While we cannot release most of the data used in the VoxelMorph papers as they prohibit redistribution, we thorough processed and [re-released OASIS1](https://github.com/adalca/medical-datasets/blob/master/neurite-oasis.md) while developing [HyperMorph](http://hypermorph.voxelmorph.net/). We now include a quick [VoxelMorph tutorial](https://colab.research.google.com/drive/1ZefmWXBupRNsnIbBbGquhVDsk-7R7L1S?usp=sharing) to train VoxelMorph on neurite-oasis data.

# Contact
For any code-related problems or questions please [open an issue](https://github.com/voxelmorph/voxelmorph/issues/new?labels=voxelmorph) or [start a discussion](https://github.com/voxelmorph/voxelmorph/discussions) of general registration/VoxelMorph topics.
