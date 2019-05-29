#!/bin/bash
#
#SBATCH -p gpu
#SBATCH --job-name=python
#SBATCH --cpus-per-task=32
#SBATCH --output=/data/vision/polina/projects/nn-visualization/code/training/semisupervised/slurm/semi_dkl_10000.txt
#SBATCH --gres=gpu:1
#SBATCH --mem=50000M
#

. /data/vision/polina/shared_software/anaconda3-4.3.1/etc/profile.d/conda.sh
export LD_LIBRARY_PATH=/data/vision/polina/shared_software/cuda-9.0/lib64:$LD_LIBRARY_PATH
conda activate grace

python /data/vision/polina/projects/nn-visualization/code/training/semisupervised/train.py