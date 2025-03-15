# SHAP-RLROBO
SHAP Analysis of RL Generalization across Robotic Environments

### Installation
```
# MUJOCO python3.9
https://github.com/openai/mujoco-py?tab=readme-ov-file#install-mujoco
pip install gymnasium[mujoco]
```

```
# PYBULLET python3.9
pip install gym==0.22.0
git clone https://github.com/benelot/pybullet-gym.git
cd pybullet-gym
pip install -e .
pip install numpy==1.24.0 # change numpy version to avoid error
```

```
# STABLE-BASELINES3
pip install stable-baselines3[extra]
```

```
# SHAP
pip install shap
```

```
# Avoid potential errors when installation
pip install numpy==1.24.0
pip install 'shimmy>=2.0'
pip install imageio-ffmpeg
```