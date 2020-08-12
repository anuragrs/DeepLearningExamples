set -e

conda_path=/shared/conda
source $conda_path/etc/profile.d/conda.sh
conda activate base

XLA_FLAGS="--xla_gpu_cuda_data_dir=/shared/conda" eval ${@}
