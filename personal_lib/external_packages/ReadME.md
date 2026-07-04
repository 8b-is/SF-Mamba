# Install External Package
If you want to use following packages, please type the command bellow.

## Install
```
# --user is needed not to create .so file in the mamba-1p1p1 directory. The .so file is harmful to the different GPUs if the home directory is shared among different GPU architectures.
pip install personal_lib/external_packages/mamba_bfold --user
pip install personal_lib/external_packages/non_causal_conv1d --user  
```

#### For GAIA
You need to install custom CUDA packages in GAIA environment with GPU, otherwise Triton internally used in mamba will fail.  
The procedure is as follows. (You only need to install once. Then, it will be saved in ~/.local directory and you can use it without installing it.)  
```
# start interactive job
salloc --account=project44 --partition=project44-a100-v2 --ntasks=1 --ntasks-per-node=1 --gpus-per-node=1 --cpus-per-task=32 --time=60
source /etc/profile.d/modules.sh
module load singularity/3.5.3
# dive into container
singularity run PATH_TO_YOUR_SINGULARITY_CONTAINER
# install 
pip install personal_lib/external_packages/mamba_bfold --user  
pip install personal_lib/external_packages/non_causal_conv1d --user  
```