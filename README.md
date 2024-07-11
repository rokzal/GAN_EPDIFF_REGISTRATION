# LDDMM meets GANs: Generative Adversarial Networks for diffeomorphic registration

## Prerequisites
- `Python 3.6`
- `Tensorflow 2.6.2 (GPU)`
- `NumPy`
- `scipy`
- `NiBabel`

## Inference

Run "src/register_arb.py" with the desired parameters.

For example: 

python ./register_arb.py --moving 0395_aligned_norm.nii --fixed 0394_aligned_norm.nii --results_path="results"  --operator='lo' --model_path="../saved_models/2024_OASIS_GAN_STA.h5" --cn_alpha=0.0025 --s=2 --no_standarize --parameterization="sta"

## Pre-trained models

See "saved_models/". Both stationary and EPDiff-constrained models reported on the original publication are available there.

## Training

If you want to train your own model, you can use "src/3dtrain_diff_main.py", with the aproppiate parameters.

For example: 

python ./3dtrain_diff_main.py --energy_loss --reg_weight=50 --id="2024_OASIS_TDE_NCC" --data_path="Data_Path/*"  --use_disk --epochs=150 --lr=1E-4 --cn_alpha=0.0025  --operator='lo' --s=2 --parameterization="tde" --batch_size=1 --gd_size=3 --no_standarize --Sim_Loss="NCC"


You might need to modify our code to fit your particular data or other requirements. You can find all relevant code in "src/lib/modules/".

# Reference Paper

(Pending)