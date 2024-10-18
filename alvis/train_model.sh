#!/usr/bin/env bash
#SBATCH -A NAISS2024-22-1219         
#SBATCH -p alvis                     
#SBATCH --gpus-per-node=A100:1       
#SBATCH -t 0-10:00:00                
#SBATCH --mail-type=END,FAIL         
#SBATCH --mail-user=victor.ju.hin.wong@gmail.com

# Load necessary modules
module load OpenCV/4.6.0-foss-2022a-CUDA-11.7.0-contrib scikit-learn/1.1.2-foss-2022a

# Activate the Python environment
source venv/bin/activate

# Export environment variables (without spaces around '=')
export DATA_PATH="/mimer/NOBACKUP/groups/naiss2024-22-1219/data/processed"
export NUM_EPOCHS=50
export X_DIM=198
export Y_DIM=195
export T_DIM=55
export BATCH_SIZE=32
export LEARNING_RATE=0.01
export NUM_WORKERS=4

# Start the training script
echo "Starting training..."
python alvis/train_model_shell.py --data-path $DATA_PATH --X_DIM $X_DIM --Y_DIM $Y_DIM --T_DIM $T_DIM --num_epochs $NUM_EPOCHS --batch_size $BATCH_SIZE --learning_rate $LEARNING_RATE --num_workers $NUM_WORKERS
echo "Training finished."