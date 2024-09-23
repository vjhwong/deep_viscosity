#!/bin/bash
#SBATCH --job-name=ml_training
#SBATCH --output=output_%j.log
#SBATCH --error=error_%j.log
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --mail-type=END,FAIL            
#SBATCH --mail-user=victor.wong@student.uu.se

module load python/3.9
module load cuda/12.3

# Activate conda environment
source ~/miniconda3/bin/activate my_env

# Set PYTHONPATH to include the directory with custom modules
export PYTHONPATH=$PYTHONPATH:/home/your_username/my_project

# Run the training script with data path as argument
python /home/your_username/my_project/train_model.py --data-path /path/to/data --num_epochs 50 --batch_size 32 --learning_rate 0.01
