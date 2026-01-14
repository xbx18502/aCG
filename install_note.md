source /etc/profile.d/modules.sh
module use /home/app/nvhpc/24.11/modulefiles
module load nvhpc-hpcx-cuda12/24.11

export NCCL_HOME="/home/app/nvhpc/24.11/Linux_x86_64/24.11/comm_libs/nccl"
export NVSHMEM_HOME="/home/app/nvhpc/24.11/Linux_x86_64/24.11/comm_libs/12.6/nvshmem"

cmake ../cuda -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=90 -DACG_ENABLE_PROFILING=ON -DNCCL_HOME="/home/app/nvhpc/24.11/Linux_x86_64/24.11/comm_libs/nccl" -DNVSHMEM_DIR="/home/app/nvhpc/24.11/Linux_x86_64/24.11/comm_libs/12.6/nvshmem" -DMETIS_DIR=${HOME}/lib/metis -DGKLIB_DIR=${HOME}/lib/gklib -DACG_ENABLE_NVTX=ON