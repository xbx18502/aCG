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
# NCCL settings
export NCCL_HOME="/home/app/nvhpc/24.11/Linux_x86_64/24.11/comm_libs/nccl"

export OMPI_MCA_plm_rsh_agent="/usr/bin/pjrsh"

export NVSHMEM_BOOTSTRAP=MPI
export NVSHMEMTEST_MPI_SUPPORT=1

# -x NVSHMEMTEST_USE_MPI_LAUNCHER=1 
# --map-by socket --bind-to socket

# 设置求解器和通信库选项
# 求解器: acg, acg-pipelined, acg-device, acg-device-pipelined
# 通信库: mpi, nccl, nvshmem


# 输入矩阵文件 (需要替换为实际的矩阵文件路径)
MATRIX_FILE="../matrices_generator/poisson2d_n2048.mtx"

# 确保矩阵文件存在
if [[ ! -f "${MATRIX_FILE}" ]]; then
	echo "[ERROR] MATRIX_FILE 不存在: ${MATRIX_FILE}" >&2
	exit 1
fi
SOLVER0="acg"
COMM0="nvshmem"
task00=" \
mpirun -v --display-allocation --display-map -hostfile ${PJM_O_NODEINF} \
-np 1 --map-by ppr:1:node \
--bind-to numa \
../build/acg-cuda --quiet --solver ${SOLVER0} --comm ${COMM0} \
--max-iterations 1000 --residual-rtol 1e-6 \
--manufactured-solution ${MATRIX_FILE}"

task01=" \
mpirun -v --display-allocation --display-map -hostfile ${PJM_O_NODEINF} \
-np 2 --map-by ppr:2:node \
--bind-to numa \
../build/acg-cuda --quiet --solver ${SOLVER0} --comm ${COMM0} \
--max-iterations 1000 --residual-rtol 1e-6 \
--manufactured-solution ${MATRIX_FILE}"

task02=" \
mpirun -v --display-allocation --display-map -hostfile ${PJM_O_NODEINF} \
-np 4 --map-by ppr:4:node \
--bind-to numa \
../build/acg-cuda --quiet --solver ${SOLVER0} --comm ${COMM0} \
--max-iterations 1000 --residual-rtol 1e-6 \
--manufactured-solution ${MATRIX_FILE}"

task03=" \
mpirun -v --display-allocation --display-map -hostfile ${PJM_O_NODEINF} \
-np 8 --map-by ppr:4:node \
--bind-to numa \
../build/acg-cuda --quiet --solver ${SOLVER0} --comm ${COMM0} \
--max-iterations 1000 --residual-rtol 1e-6 \
--manufactured-solution ${MATRIX_FILE}"

SOLVER1="acg-pipelined"
COMM1="nvshmem"
task10=" \
mpirun -v --display-allocation --display-map -hostfile ${PJM_O_NODEINF} \
-np 1 --map-by ppr:1:node \
--bind-to numa \
../build/acg-cuda --quiet --solver ${SOLVER1} --comm ${COMM1} \
--max-iterations 1000 --residual-rtol 1e-6 \
--manufactured-solution ${MATRIX_FILE}"

task11=" \
mpirun -v --display-allocation --display-map -hostfile ${PJM_O_NODEINF} \
-np 2 --map-by ppr:2:node \
--bind-to numa \
../build/acg-cuda --quiet --solver ${SOLVER1} --comm ${COMM1} \
--max-iterations 1000 --residual-rtol 1e-6 \
--manufactured-solution ${MATRIX_FILE}"

task12=" \
mpirun -v --display-allocation --display-map -hostfile ${PJM_O_NODEINF} \
-np 4 --map-by ppr:4:node \
--bind-to numa \
../build/acg-cuda --quiet --solver ${SOLVER1} --comm ${COMM1} \
--max-iterations 1000 --residual-rtol 1e-6 \
--manufactured-solution ${MATRIX_FILE}"

task13=" \
mpirun -v --display-allocation --display-map -hostfile ${PJM_O_NODEINF} \
-np 8 --map-by ppr:4:node \
--bind-to numa \
../build/acg-cuda --quiet --solver ${SOLVER1} --comm ${COMM1} \
--max-iterations 1000 --residual-rtol 1e-6 \
--manufactured-solution ${MATRIX_FILE}"


SOLVER2="acg-device-pipelined"
COMM2="nvshmem"
task20=" \
mpirun -v --display-allocation --display-map -hostfile ${PJM_O_NODEINF} \
-np 1 --map-by ppr:1:node \
--bind-to numa \
../build/acg-cuda --quiet --solver ${SOLVER2} --comm ${COMM2} \
--max-iterations 1000 --residual-rtol 1e-6 \
--manufactured-solution ${MATRIX_FILE}"

task21=" \
mpirun -v --display-allocation --display-map -hostfile ${PJM_O_NODEINF} \
-np 2 --map-by ppr:2:node \
--bind-to numa \
../build/acg-cuda --quiet --solver ${SOLVER2} --comm ${COMM2} \
--max-iterations 1000 --residual-rtol 1e-6 \
--manufactured-solution ${MATRIX_FILE}"

task22=" \
mpirun -v --display-allocation --display-map -hostfile ${PJM_O_NODEINF} \
-np 4 --map-by ppr:4:node \
--bind-to numa \
../build/acg-cuda --quiet --solver ${SOLVER2} --comm ${COMM2} \
--max-iterations 1000 --residual-rtol 1e-6 \
--manufactured-solution ${MATRIX_FILE}"

task23=" \
mpirun -v --display-allocation --display-map -hostfile ${PJM_O_NODEINF} \
-np 8 --map-by ppr:4:node \
--bind-to numa \
../build/acg-cuda --quiet --solver ${SOLVER2} --comm ${COMM2} \
--max-iterations 1000 --residual-rtol 1e-6 \
--manufactured-solution ${MATRIX_FILE}"


# echo "task00: ${task00}"
# echo "task01: ${task01}"
# echo "task02: ${task02}"
# echo "task03: ${task03}"
echo "SOLVER0: ${SOLVER0}, COMM0: ${COMM0}"
eval ${task00} 2>&1 | grep -E "total solver time"
eval ${task01} 2>&1 | grep -E "total solver time"
eval ${task02} 2>&1 | grep -E "total solver time"
eval ${task03} 2>&1 | grep -E "total solver time"


echo "SOLVER1: ${SOLVER1}, COMM1: ${COMM1}"
eval ${task10} 2>&1 | grep -E "total solver time"
eval ${task11} 2>&1 | grep -E "total solver time"
eval ${task12} 2>&1 | grep -E "total solver time"
eval ${task13} 2>&1 | grep -E "total solver time"


echo "SOLVER2: ${SOLVER2}, COMM2: ${COMM2}"
eval ${task20} 2>&1 | grep -E "total solver time"
eval ${task21} 2>&1 | grep -E "total solver time"
eval ${task22} 2>&1 | grep -E "total solver time"
eval ${task23} 2>&1 | grep -E "total solver time"