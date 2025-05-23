#!/bin/bash

# Default values
ALGORITHM="A2C"
ENV_PAIRS="Hopper-v5,HopperMuJoCoEnv-v0"
N_SAMPLES=500
TRAIN_STEPS=100000
EVAL_EPISODES=20
TARGET="generalization_gap"
LOG_DIR="rl_cross_env_analysis_results"
DEVICE="cuda:0"
RUN_ALL_PAIRS=true

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --algorithm=*)
      ALGORITHM="${1#*=}"
      ;;
    --env_pairs=*)
      ENV_PAIRS="${1#*=}"
      ;;
    --n_samples=*)
      N_SAMPLES="${1#*=}"
      ;;
    --train_steps=*)
      TRAIN_STEPS="${1#*=}"
      ;;
    --eval_episodes=*)
      EVAL_EPISODES="${1#*=}"
      ;;
    --target=*)
      TARGET="${1#*=}"
      ;;
    --log_dir=*)
      LOG_DIR="${1#*=}"
      ;;
    --device=*)
      DEVICE="${1#*=}"
      ;;
    --run_all_pairs)
      RUN_ALL_PAIRS=true
      ;;
    *)
      echo "Unknown parameter: $1"
      exit 1
      ;;
  esac
  shift
done

# Set job name and log directory if not provided
if $RUN_ALL_PAIRS; then
  JOB_NAME="${ALGORITHM}_all_pairs"
  if [ -z "$LOG_DIR" ]; then
    LOG_DIR="rl_cross_env_analysis_results"
  fi
else
  JOB_NAME="${ALGORITHM}_${ENV_PAIRS}"
  if [ -z "$LOG_DIR" ]; then
    LOG_DIR="rl_cross_env_analysis_results/${ENV_PAIRS}"
  fi
fi

# Create log directory
mkdir -p ${LOG_DIR}

# Create sbatch script
cat << EOF > job_${JOB_NAME}.sbatch
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --output=${LOG_DIR}/slurm_%j.out
#SBATCH --error=${LOG_DIR}/slurm_%j.err
#SBATCH --time=336:00:00  # This is already 14 days, which is good
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8  # Increase from 4 to 8
#SBATCH --mem=32G  # Increase from 16G to 32G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

# Create log directory
mkdir -p ${LOG_DIR}

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/home/lin30127/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/home/lin30127/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/home/lin30127/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="/home/lin30127/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/lin30127/.mujoco/mujoco210/bin

# Load required modules (adjust according to your system)
conda activate rlrobo

# Activate virtual environment if needed
# source /path/to/your/venv/bin/activate

cd /home/lin30127/workspace/SHAP-RLROBO

echo "Starting Cross-Environment RL hyperparameter analysis job at \$(date)"
echo "Algorithm: ${ALGORITHM}"
if $RUN_ALL_PAIRS; then
  echo "Running all environment pairs"
  
  # Run the analysis script with all pairs
  python CrossEnvRLHyperparameter.py \
    --algorithms ${ALGORITHM} \
    --n_samples ${N_SAMPLES} \
    --train_steps ${TRAIN_STEPS} \
    --eval_episodes ${EVAL_EPISODES} \
    --log_dir ${LOG_DIR} \
    --device ${DEVICE}
else
  echo "Run environment pairs: ${ENV_PAIRS}"
  
  # Run the analysis script for a single pair
  python CrossEnvRLHyperparameter.py \
    --algorithms ${ALGORITHM} \
    --env_pairs ${ENV_PAIRS} \
    --n_samples ${N_SAMPLES} \
    --train_steps ${TRAIN_STEPS} \
    --eval_episodes ${EVAL_EPISODES} \
    --log_dir ${LOG_DIR} \
    --device ${DEVICE}
fi

echo "Job completed at \$(date)"
EOF

# Submit the job
sbatch job_${JOB_NAME}.sbatch
if $RUN_ALL_PAIRS; then
  echo "Submitted job for ${ALGORITHM} on all environment pairs"
else
  echo "Submitted job for ${ALGORITHM} on ${ENV_PAIRS}"
fi
echo "Log directory: ${LOG_DIR}"
echo "To monitor training status: cat ${LOG_DIR}/status.txt"
