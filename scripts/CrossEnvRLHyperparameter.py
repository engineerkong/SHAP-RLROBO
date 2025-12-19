# Features: cross-env, earlystopping, random seed, argparse, direction
import numpy as np
import pandas as pd
import gymnasium
import gym
import pybulletgym
import os
from glob import glob
from stable_baselines3 import PPO, A2C, DDPG, SAC, TD3
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from gym_monitor import Monitor as GymMonitor
from RLHyperparameter import RLHyperparameter

# Custom callback for early stopping based on training plateau
class EarlyStoppingCallback(BaseCallback):
    def __init__(self, check_freq=1000, window_size=5, min_delta=0.001, patience=3, verbose=1):
        """
        Early stopping callback that monitors the mean reward and stops training when it plateaus.
        
        Parameters:
        -----------
        check_freq : int
            Frequency to check for improvement (in timesteps)
        window_size : int
            Number of evaluations to consider for determining plateau
        min_delta : float
            Minimum change in mean reward to qualify as an improvement
        patience : int
            Number of checks with no improvement after which training will stop
        verbose : int
            Verbosity level
        """
        super().__init__(verbose)
        self.check_freq = check_freq
        self.window_size = window_size
        self.min_delta = min_delta
        self.patience = patience
        self.best_mean_reward = -float('inf')
        self.no_improvement_count = 0
        self.last_mean_rewards = []
        
    def _on_step(self):
        # Check if it's time to evaluate
        if self.n_calls % self.check_freq == 0:
            # Get current episode rewards directly from the model's environment
            if hasattr(self.model, 'ep_info_buffer') and len(self.model.ep_info_buffer) > 0:
                # Use the episode info buffer from the model
                rewards = [ep_info['r'] for ep_info in self.model.ep_info_buffer]
                if len(rewards) > 0:
                    # Mean episode reward over last window_size episodes
                    current_reward = np.mean(rewards[-self.window_size:]) if len(rewards) >= self.window_size else np.mean(rewards)
                    self.last_mean_rewards.append(current_reward)
                    
                    # Keep only the last window_size rewards
                    if len(self.last_mean_rewards) > self.window_size:
                        self.last_mean_rewards.pop(0)
                        
                    # Check for improvement
                    if len(self.last_mean_rewards) == self.window_size:
                        # Check if the reward is plateauing
                        is_plateauing = True
                        for i in range(1, len(self.last_mean_rewards)):
                            if self.last_mean_rewards[i] - self.last_mean_rewards[i-1] > self.min_delta:
                                is_plateauing = False
                                break
                        
                        if is_plateauing:
                            self.no_improvement_count += 1
                        else:
                            self.no_improvement_count = 0
                            
                        # Check if we should stop training
                        if self.no_improvement_count >= self.patience:
                            if self.verbose > 0:
                                print(f"Stopping training because reward has plateaued for {self.patience} checks.")
                            return False
                            
                    # Also stop if we've reached an exceptionally high reward (optional)
                    if current_reward > self.best_mean_reward:
                        self.best_mean_reward = current_reward
                    
        return True

# Helper functions to extract data from Monitor
def ts2xy(timesteps, xaxis):
    """
    Convert timesteps to x,y data for plotting
    """
    if xaxis == 'timesteps':
        x = np.cumsum(timesteps.l.values)
        y = timesteps.r.values
    elif xaxis == 'episodes':
        x = np.arange(len(timesteps))
        y = timesteps.r.values
    else:
        raise NotImplementedError
    return x, y

def load_results(path):
    """
    Load results from Monitor logs
    """
    if os.path.isdir(path):
        monitor_files = glob(os.path.join(path, "*.monitor.csv"))
        if len(monitor_files) == 0:
            return None
        path = monitor_files[0]
    
    monitor_data = pd.read_csv(path, skiprows=1)
    return monitor_data

# Analysis class for training and testing with different environments
class CrossEnvRLHyperparameter(RLHyperparameter):
    def __init__(self, train_env_name, test_env_name, algorithm, param_grid, n_samples=50, seed=None, 
                 train_steps=100000, eval_episodes=10, log_dir=None, device="cuda:0", direction="forward"):
        """
        Initialize cross-environment RL hyperparameter SHAP analysis
        
        Parameters:
        -----------
        train_env_name : str
            Environment name for training
        test_env_name : str
            Environment name for testing
        algorithm : str
            Name of the RL algorithm (e.g., 'PPO', 'A2C', 'DDPG', 'SAC', 'TD3')
        param_grid : dict
            Hyperparameter ranges for the RL algorithm
        n_samples : int
            Number of hyperparameter combinations to sample
        seed : int
            Random seed for reproducibility
        train_steps : int
            Total timesteps to train the RL algorithm
        eval_episodes : int
            Number of episodes to use for evaluation
        log_dir : str, optional
            Path to the log directory
        device : str
            Computing device ('cpu' or 'cuda:0')
        direction : str
            Direction of environment transition: 'forward' (Gymnasium→Gym) or 'reverse' (Gym→Gymnasium)
        """
        super().__init__(
            env_name=None,  # Do not use the base class's env_name
            algorithm=algorithm,
            param_grid=param_grid,
            n_samples=n_samples,
            seed=seed,
            train_steps=train_steps,
            eval_episodes=eval_episodes,
            log_dir=log_dir if log_dir else "./rl_cross_env_hyperparams_logs",
            device=device
        )
        
        self.train_env_name = train_env_name
        self.test_env_name = test_env_name
        self.direction = direction

    def evaluate_generalization(self, params, run_id):
        """
        Train the RL algorithm with given hyperparameters and evaluate generalization performance 
        across different environments
        
        Parameters:
        -----------
        params : dict
            Hyperparameters for the RL algorithm
        run_id : int
            Unique identifier for this run
            
        Returns:
        --------
        dict
            Dictionary containing training reward, testing reward, and generalization gap
        """
        # Create a separate log directory for each run
        run_log_dir = os.path.join(self.log_dir, f"run_{run_id}")
        os.makedirs(run_log_dir, exist_ok=True)
        
        # Create environments based on direction
        if self.direction == "reverse":
            # Reverse: Train on Gym (PyBullet), Test on Gymnasium (MuJoCo)
            train_env = gym.make(self.train_env_name)
            train_env.seed(seed=self.seed)
            train_env.reset()
            train_env = GymMonitor(train_env, os.path.join(run_log_dir, "train_monitor"))
            
            test_env = gymnasium.make(self.test_env_name)
            test_env.reset(seed=self.seed)
            test_env = Monitor(test_env, os.path.join(run_log_dir, "test_monitor"))
        else:  # forward
            # Forward: Train on Gymnasium (MuJoCo), Test on Gym (PyBullet)
            train_env = gymnasium.make(self.train_env_name)
            train_env.reset(seed=self.seed)
            train_env = Monitor(train_env, os.path.join(run_log_dir, "train_monitor"))
            
            test_env = gym.make(self.test_env_name)
            test_env.seed(seed=self.seed)
            test_env.reset()
            test_env = GymMonitor(test_env, os.path.join(run_log_dir, "test_monitor"))
        
        print(self.device)
        # Select and create model
        if self.algorithm == 'PPO':
            model = PPO("MlpPolicy", train_env, verbose=0, tensorboard_log=run_log_dir, seed=self.seed, device=self.device, **params)
        elif self.algorithm == 'A2C':
            model = A2C("MlpPolicy", train_env, verbose=0, tensorboard_log=run_log_dir, seed=self.seed, device=self.device, **params)
        elif self.algorithm == 'DDPG':
            model = DDPG("MlpPolicy", train_env, verbose=0, tensorboard_log=run_log_dir, seed=self.seed, device=self.device, **params)
        elif self.algorithm == 'SAC':
            model = SAC("MlpPolicy", train_env, verbose=0, tensorboard_log=run_log_dir, seed=self.seed, device=self.device, **params)
        elif self.algorithm == 'TD3':
            model = TD3("MlpPolicy", train_env, verbose=0, tensorboard_log=run_log_dir, seed=self.seed, device=self.device, **params)
        else:
            raise ValueError(f"Unsupported RL algorithm: {self.algorithm}")
        
        # Set up early stopping with evaluation callback
        eval_callback = EvalCallback(
            train_env,
            best_model_save_path=os.path.join(run_log_dir, "best_model"),
            log_path=os.path.join(run_log_dir, "eval_results"),
            eval_freq=max(1000, self.train_steps//10),
            deterministic=True,
            render=False,
            verbose=0
        )
        
        # Custom early stopping callback
        early_stop_callback = EarlyStoppingCallback(
            check_freq=max(1000, self.train_steps//20),
            window_size=5,
            min_delta=0.005,  # Minimum improvement needed
            patience=3        # Stop after 3 checks with no improvement
        )
        
        # Combine callbacks
        callbacks = [eval_callback, early_stop_callback]
        
        # Train the model with callbacks
        model.learn(total_timesteps=self.train_steps, progress_bar=True, callback=callbacks)
        
        # Evaluate performance in the training environment
        train_mean_reward, train_std_reward = evaluate_policy(model, train_env, n_eval_episodes=self.eval_episodes, deterministic=True)
        
        # Evaluate performance in the testing environment
        test_mean_reward, test_std_reward = evaluate_policy(model, test_env, n_eval_episodes=self.eval_episodes, deterministic=True)
        
        # Calculate generalization gap (positive means better generalization to test environment)
        generalization_gap = train_mean_reward - test_mean_reward
        
        # Save the best model if available, otherwise save final model
        best_model_path = os.path.join(run_log_dir, "best_model", "best_model.zip")
        if os.path.exists(best_model_path):
            # Use the best model for final evaluation
            if self.algorithm == 'PPO':
                model = PPO.load(best_model_path)
            elif self.algorithm == 'A2C':
                model = A2C.load(best_model_path)
            elif self.algorithm == 'DDPG':
                model = DDPG.load(best_model_path)
            elif self.algorithm == 'SAC':
                model = SAC.load(best_model_path)
            elif self.algorithm == 'TD3':
                model = TD3.load(best_model_path)
        
        # Save final model
        model.save(os.path.join(run_log_dir, "final_model"))
        
        # Close environments
        train_env.close()
        test_env.close()
        
        return {
            'train_reward': train_mean_reward,
            'train_reward_std': train_std_reward,
            'test_reward': test_mean_reward,
            'test_reward_std': test_std_reward,
            'generalization_gap': generalization_gap
        }


# Cross-environment analysis example
def cross_environment_analysis_example(algorithms=None, env_pairs=None, n_samples=5, train_steps=100, eval_episodes=10, 
                                       target="generalization_gap", log_dir="./rl_cross_env_analysis_results", 
                                       seed=None, device="cuda:0", num_seeds=3, direction="forward"):
    """
    Run cross-environment RL hyperparameter analysis with SHAP
    
    Parameters:
    -----------
    algorithms : list of str, optional
        RL algorithms to analyze (e.g., ['PPO', 'SAC'])
    env_pairs : list of tuples, optional
        Environment pairs for cross-environment testing (train_env, test_env)
    n_samples : int
        Number of hyperparameter combinations to sample
    train_steps : int
        Number of training timesteps for each run
    eval_episodes : int
        Number of episodes for evaluation
    target : str
        Target variable for SHAP analysis (e.g., 'generalization_gap', 'test_reward')
    log_dir : str
        Directory for saving results
    seed : int, optional
        Random seed for reproducibility
    device : str
        Device to use for training ('cpu' or 'cuda:0')
    num_seeds : int
        Number of seed repetitions for robustness
    direction : str
        Direction of environment transition: 'forward' (Gymnasium→Gym) or 'reverse' (Gym→Gymnasium)
    """
    
    # Define RL algorithms and hyperparameter grids
    rl_param_grids = {
        'PPO': {
            'learning_rate': (0.0001, 0.01),
            'gamma': (0.8, 0.999),
            'clip_range': (0.1, 0.3),
            'n_steps': [128, 512, 1024, 2048]
        },
        'SAC': {
            'learning_rate': (0.0001, 0.01),
            'gamma': (0.8, 0.999),
            'tau': (0.001, 0.01),
            'ent_coef': (0.01, 0.2)
        },
        'A2C': {
            'learning_rate': (0.0001, 0.01),
            'gamma': (0.8, 0.999),
            'gae_lambda': (0.9, 0.99),
            'vf_coef': (0.25, 1.0)
        },
        'DDPG': {
            'learning_rate': (0.0001, 0.01),
            'gamma': (0.8, 0.999),
            'tau': (0.001, 0.01),
            'buffer_size': [10000, 100000, 1000000]
        }
    }

    # Define default environment pairs based on direction
    if direction == "reverse":
        # Reverse: Train on Gym (old API), Test on Gymnasium (new API)
        default_env_pairs = [
            ('InvertedPendulumMuJoCoEnv-v0', 'InvertedPendulum-v5'),
            ('HalfCheetahMuJoCoEnv-v0', 'HalfCheetah-v5'),
            ('HopperMuJoCoEnv-v0', 'Hopper-v5'),
            ('Walker2DMuJoCoEnv-v0', 'Walker2d-v5')
        ]
    else:  # forward
        # Forward: Train on Gymnasium (new API), Test on Gym (old API)
        default_env_pairs = [
            ('InvertedPendulum-v5', 'InvertedPendulumMuJoCoEnv-v0'),
            ('HalfCheetah-v5', 'HalfCheetahMuJoCoEnv-v0'),
            ('Hopper-v5', 'HopperMuJoCoEnv-v0'),
            ('Walker2d-v5', 'Walker2DMuJoCoEnv-v0')
        ]
    
    if env_pairs is None:
        env_pairs = default_env_pairs
        
    if algorithms is None:
        algorithms = list(rl_param_grids.keys())
    else:
        # Verify all requested algorithms exist in our param grids
        for algo in algorithms:
            if algo not in rl_param_grids:
                raise ValueError(f"Algorithm {algo} not supported. Choose from: {list(rl_param_grids.keys())}")

    # Run the analysis separately for each algorithm and environment pair
    for algorithm in algorithms:
        param_grid = rl_param_grids[algorithm]
        combined_results = []
        for train_env, test_env in env_pairs:
            print(f"\nRunning analysis for {algorithm} on {train_env} → {test_env} (direction: {direction})...")
            
            # Initialize the analyzer for the current algorithm and environment pair
            cross_env_analyzer = CrossEnvRLHyperparameter(
                train_env_name=train_env,
                test_env_name=test_env,
                algorithm=algorithm,
                param_grid=param_grid,
                n_samples=n_samples,
                seed=seed,
                train_steps=train_steps,
                eval_episodes=eval_episodes,
                log_dir=os.path.join(log_dir, algorithm, train_env),
                device=device,
                direction=direction
            )
            
            # Build dataset with multiple seeds per hyperparameter combination
            results = cross_env_analyzer.build_dataset(
                save_path=os.path.join(log_dir, algorithm, train_env, "results.csv"),
                num_seeds=num_seeds
            )
            combined_results.append(results)
            print("Results dataframe:")
            print(results.head())
        
        # Combine results from all environment pairs
        if combined_results:
            combined_results_df = pd.concat(combined_results, ignore_index=True)
            os.makedirs(os.path.join(log_dir, algorithm), exist_ok=True)
            combined_results_df.to_csv(os.path.join(log_dir, algorithm, "combined_results.csv"), index=False)
            print(f"\nCombined results saved to: {os.path.join(log_dir, algorithm, 'combined_results.csv')}")
            print(f"Total runs: {len(combined_results_df)} (n_samples={n_samples} × num_seeds={num_seeds} × num_env_pairs={len(env_pairs)})")
        
        print(f"Hyperparameter grid: {param_grid}")
        
        print(f"Cross-environment analysis complete for {algorithm}. Results saved in: {cross_env_analyzer.log_dir}")
