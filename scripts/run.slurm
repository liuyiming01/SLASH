#!/bin/bash
 
#SBATCH --job-name=select4_2      # 任务名称，记得修改
#SBATCH --partition=RTX3090               # 用sinfo命令可以看到所有队列，默认为debug队列
#SBATCH --nodes=1                       # 申请的节点数量，单机训练无需调整
#SBATCH --ntasks-per-node=2             # 若多卡或多进程，请调整此参数
#SBATCH --cpus-per-task=16               # 每个进程的CPU数量
#SBATCH --mem=100GB                      # 申请的内存大小
#SBATCH --gres=gpu:2                    # 若使用2块卡，则gres=gpu:2
#SBATCH --output=slurm_logs/%j.out                 # 设置标准输出的文件路径
#SBATCH --error=slurm_logs/%j.err                  # 设置标准错误的文件路径

bash run_select.sh