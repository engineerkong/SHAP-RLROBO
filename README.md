# SHAP-RLROBO
Repository for manuscript "Optimizing RL Generalizability in Robotics through SHAP Analysis of Algorithms and Hyperparameters"

### Installation
```bash
# MUJOCO
https://github.com/openai/mujoco-py?tab=readme-ov-file#install-mujoco
pip install gymnasium[mujoco]
```

```bash
# PYBULLET
pip install gym==0.22.0
git clone https://github.com/benelot/pybullet-gym.git
cd pybullet-gym
pip install -e .
```

```bash
# STABLE-BASELINES3, SHAP and SEABORN
pip install stable-baselines3[extra]
pip install shap
pip install seaborn
```

```bash
# Avoid potential errors when installation
pip install numpy==1.24.0
pip install 'shimmy>=2.0'
pip install imageio-ffmpeg
```

### Workflow

```bash
# 1. Sample configurations, train models and evaluate generalization (GPU required)
python run_experiments.py --help
```

```bash
# 2. Process results by needs
python process_results.py
```

```bash
# 3. SHAP process and analysis
python shap_analysis_pipeline.py
```

## Analysis Components

The SHAP analysis pipeline includes:
- **Main Impacts** - Compare configuration impacts across PPO, A2C, DDPG, and SAC
- **Interaction Impacts** - Examine hyperparameter interactions within algorithms
- **Task and Environment Insights** - Analyze performance on multiple tasks across MuJoCo and PyBullet
- **Optimal Configuration Selection** - Identify top-performing hyperparameter configurations

## Output

Results are saved to:
- `./results_experiments` 
- `./results_SHAP`