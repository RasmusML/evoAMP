#!/bin/sh
#BSUB -q gpua100
#BSUB -J rmlsJob
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 1:00
#BSUB -R "rusage[mem=3GB]"
#BSUB -o logs/%J.out
#BSUB -e logs/%J.err
source ../evoamp_env/bin/activate
module swap cudnn/v8.9.1.23-prod-cuda-11.X
echo "Running script..."
python scripts/train_model.py