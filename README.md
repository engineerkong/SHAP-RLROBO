# SHAP-RLROBO
Repository for manuscript "Optimizing RL Generalizability in Robotics through SHAP Analysis of Algorithms and Hyperparameters"

### Installation
```
# MUJOCO
https://github.com/openai/mujoco-py?tab=readme-ov-file#install-mujoco
pip install gymnasium[mujoco]
```

```
# PYBULLET
pip install gym==0.22.0
git clone https://github.com/benelot/pybullet-gym.git
cd pybullet-gym
pip install -e .
```

```
# STABLE-BASELINES3, SHAP and SEABORN
pip install stable-baselines3[extra]
pip install shap
pip install seaborn
```

```
# Avoid potential errors when installation
pip install numpy==1.24.0
pip install 'shimmy>=2.0'
pip install imageio-ffmpeg
```

### Running
```
# batch train models
batch submit_cross_env_analysis.sh (GPU required)

# preprocess results
python epilen_ratio.py
python csv_concatenator.py

# SHAP process and analysis
python SHAPAnalysis.py --process="process"
python SHAPAnalysis.py --process="all"
```