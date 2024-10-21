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
export MODEL_PATH="models/eager-vortex-142/eager-vortex-142.pth"
export DATA_PATH="/mimer/NOBACKUP/groups/naiss2024-22-1219/processed"
export X_DIM=198
export Y_DIM=195
export T_DIM=55
export BATCH_SIZE=32
export NUM_WORKERS=4

# Start the training script
echo "Starting testing..."
python deep_viscosity/test_model_shell.py --model_path $MODEL_PATH --data_path $DATA_PATH --x_dim $X_DIM --y_dim $Y_DIM --t_dim $T_DIM --batch_size $BATCH_SIZE --val_size 0.15 --test_size 0.15 --num_workers $NUM_WORKERS
echo "Testing finished."
