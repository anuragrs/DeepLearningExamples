set -ex
conda_path=/shared/conda2
source $conda_path/etc/profile.d/conda.sh
conda activate base

eval ${@}
