#!/usr/bin/env bash
#SBATCH -A NAISS2024-22-1219         
#SBATCH -p alvis                     
#SBATCH --gpus-per-node=A100:1       
#SBATCH -t 0-10:00:00                
#SBATCH --mail-type=END,FAIL         
#SBATCH --mail-user=victor.ju.hin.wong@gmail.com

# Load necessary modules
module load OpenCV/4.6.0-foss-2022a-CUDA-11.7.0-contrib scikit-learn/1.1.2-foss-2022a PyTorch-bundle/1.13.1-foss-2022a-CUDA-11.7.0 matplotlib/3.5.2-foss-2022a

# Activate the Python environment
source venv/bin/activate

# Export environment variables (without spaces around '=')
export DATA_PATH="/mimer/NOBACKUP/groups/naiss2024-22-1219/processed"
export NUM_EPOCHS=50
export X_DIM=198
export Y_DIM=195
export T_DIM=55
export BATCH_SIZE=32
export LEARNING_RATE=0.001
export NUM_WORKERS=4
export RANDOM_SEED=8

# Start the training script
echo "Starting training..."
python deep_viscosity/train_model_shell.py --random_seed $RANDOM_SEED --data_path $DATA_PATH --x_dim $X_DIM --y_dim $Y_DIM --t_dim $T_DIM --num_epochs $NUM_EPOCHS --batch_size $BATCH_SIZE --learning_rate $LEARNING_RATE --val_size 0.15 --test_size 0.15 --num_workers $NUM_WORKERS
echo "Training finished."
