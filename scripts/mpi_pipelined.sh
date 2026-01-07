#!/bin/bash
#PJM -L rscgrp=b-batch
#PJM -L node=2
#PJM -L elapse=00:05:00
#PJM -j
#PJM -S


module purge
source /etc/profile.d/modules.sh
module use /home/app/nvhpc/24.11/modulefiles
module load nvhpc-hpcx-cuda12/24.11

export NVCC_GENCODE="arch=compute_90,code=sm_90"

export NVSHMEM_HOME="/home/app/nvhpc/24.11/Linux_x86_64/24.11/comm_libs/12.6/nvshmem"

export HYDRA_HOME="$NVSHMEM_HOME"

export CUDA_HOME="/home/app/nvhpc/24.11/Linux_x86_64/24.11/cuda/12.6"

export MPI_HOME="/home/app/nvhpc/24.11/Linux_x86_64/24.11/comm_libs/12.6/hpcx/hpcx-2.20/ompi"
export NCCL_HOME="/home/app/nvhpc/24.11/Linux_x86_64/24.11/comm_libs/nccl"

export OMPI_MCA_plm_rsh_agent="/usr/bin/pjrsh"

export UCX_IB_GPU_DIRECT_RDMA=no
export UCX_MEMTYPE_REG_METHODS=rcache
export UCX_TLS=rc,self,sm,cuda_copy,gdr_copy
# 设置求解器和通信库选项
# 求解器: acg, acg-pipelined, acg-device, acg-device-pipelined
# 通信库: mpi, nccl, nvshmem
SOLVER="acg-pipelined"
COMM="mpi"

# 输入矩阵文件 (需要替换为实际的矩阵文件路径)
MATRIX_FILE="../matrices_generator/poisson2d.mtx"

# 确保矩阵文件存在
if [[ ! -f "${MATRIX_FILE}" ]]; then
	echo "[ERROR] MATRIX_FILE 不存在: ${MATRIX_FILE}" >&2
	exit 1
fi

task=" \
mpirun -v --display-allocation --display-map -hostfile ${PJM_O_NODEINF} \
-np 8 --map-by ppr:4:node \
--bind-to numa \
../build/acg-cuda --solver ${SOLVER} --comm ${COMM} \
--max-iterations 1000 --residual-rtol 1e-6 \
--manufactured-solution --verbose ${MATRIX_FILE}"

echo "command: ${task}"

eval ${task}
