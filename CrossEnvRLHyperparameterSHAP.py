# Features: cross-env, earlystopping, random seed, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import gymnasium
import gym
import pybulletgym
import time
from tqdm import tqdm
import os
import argparse
import random  # Import random module for generating random seeds
from glob import glob
from sklearn.ensemble import RandomForestRegressor
from stable_baselines3 import PPO, A2C, DDPG, SAC, TD3
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from gym_monitor import Monitor as GymMonitor
from RLHyperparameterSHAP import RLHyperparameter, SHAPExplainer

# Custom callback for early stopping based on training plateau
class EarlyStoppingCallback(BaseCallback):
    def __init__(self, check_freq=1000, window_size=5, min_delta=0.001, patience=3, verbose=1):
        """
        Early stopping callback that monitors the mean reward and stops training when it plateaus.
        
        Parameters:
        -----------
        check_freq : int
            Frequency to check for improvement (in terms of timesteps)
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
    def __init__(self, train_env_name, test_env_name, algorithm, param_grid, n_samples=50, random_state=42, 
                 train_steps=100000, eval_episodes=10, log_dir=None, device="cpu"):
        """
        Initialize cross-environment RL hyperparameter SHAP analysis
        
        Parameters:
        -----------
        train_env_name : str
            OpenAI Gym environment for training
        test_env_name : str
            OpenAI Gym environment for testing
        algorithm : str
            Name of the RL algorithm (e.g., 'PPO', 'A2C', 'DDPG', 'SAC', 'TD3')
        param_grid : dict
            Hyperparameter ranges, including algorithm selection
        n_samples : int
            Number of hyperparameter combinations to sample
        random_state : int
            Random seed
        train_steps : int
            Total steps to train the RL algorithm
        eval_episodes : int
            Number of episodes to use for evaluation
        log_dir : str, optional
            Path to the log directory
        device : str
            Computing device ('cpu' or 'cuda')
        """
        super().__init__(
            env_name=None,  # Do not use the base class's env_name
            algorithm=algorithm,
            param_grid=param_grid,
            n_samples=n_samples,
            random_state=random_state,
            train_steps=train_steps,
            eval_episodes=eval_episodes,
            log_dir=log_dir if log_dir else "./rl_cross_env_hyperparams_logs",
            device=device
        )
        
        self.train_env_name = train_env_name
        self.test_env_name = test_env_name

    def evaluate_generalization(self, params, run_id):
        """Train the RL algorithm with given hyperparameters and evaluate generalization performance in different environments"""
        # Create a separate log directory for each run
        run_log_dir = os.path.join(self.log_dir, f"run_{run_id}")
        os.makedirs(run_log_dir, exist_ok=True)
        
        # Create training environment
        train_env = gymnasium.make(self.train_env_name)
        train_env.reset(seed=self.random_state)
        train_env = Monitor(train_env, os.path.join(run_log_dir, "train_monitor"))
        
        # Create testing environment
        test_env = gym.make(self.test_env_name)
        test_env.seed(seed=self.random_state)
        test_env.reset()
        test_env = GymMonitor(test_env, os.path.join(run_log_dir, "test_monitor"))
        
        # Select and create model
        if self.algorithm == 'PPO':
            model = PPO("MlpPolicy", train_env, verbose=0, tensorboard_log=run_log_dir, seed=self.random_state, device=self.device, **params)
        elif self.algorithm == 'A2C':
            model = A2C("MlpPolicy", train_env, verbose=0, tensorboard_log=run_log_dir, seed=self.random_state, device=self.device, **params)
        elif self.algorithm == 'DDPG':
            model = DDPG("MlpPolicy", train_env, verbose=0, tensorboard_log=run_log_dir, seed=self.random_state, device=self.device, **params)
        elif self.algorithm == 'SAC':
            model = SAC("MlpPolicy", train_env, verbose=0, tensorboard_log=run_log_dir, seed=self.random_state, device=self.device, **params)
        elif self.algorithm == 'TD3':
            model = TD3("MlpPolicy", train_env, verbose=0, tensorboard_log=run_log_dir, seed=self.random_state, device=self.device, **params)
        else:
            raise ValueError(f"Unsupported RL algorithm: {self.algorithm}")
        
        # Set up early stopping
        # First, create a standard evaluation callback
        eval_callback = EvalCallback(
            train_env,
            best_model_save_path=os.path.join(run_log_dir, "best_model"),
            log_path=os.path.join(run_log_dir, "eval_results"),
            eval_freq=max(1000, self.train_steps//10),
            deterministic=True,
            render=False,
            verbose=0
        )
        
        # Also set up our custom early stopping callback
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
        
        # Calculate generalization gap
        gen_gap = test_mean_reward - train_mean_reward # train_std_reward / test_std_reward
        
        # Save the model - use best model if saved successfully, otherwise use final model
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
        
        # Save final model anyway (might be the same as best model)
        model.save(os.path.join(run_log_dir, "final_model"))
        
        # Close environments
        train_env.close()
        test_env.close()
        
        return {
            'train_reward': train_mean_reward,
            'train_reward_std': train_std_reward,
            'test_reward': test_mean_reward,
            'test_reward_std': test_std_reward,
            'generalization_gap': gen_gap
        }


# Cross-environment analysis example
def cross_environment_analysis_example(algorithms=None, env_pairs=None, n_samples=5, train_steps=100, eval_episodes=10, 
                                       target="generalization_gap", log_dir="./rl_cross_env_analysis_results", seed=None, device="cpu"):
    # Use a random seed if none is provided
    print(f"Using random seed: {seed}")
    
    # Define RL algorithms and hyperparameter grids
    rl_param_grids = {
        'PPO': {
            'learning_rate': (0.0001, 0.01),
            'gamma': (0.8, 0.999),
            'batch_size': [64, 128, 256],
            'n_steps': [128, 512, 1024, 2048],
            'clip_range': (0.1, 0.3)
        },
        'A2C': {
            'learning_rate': (0.0001, 0.01),
            'gamma': (0.9, 0.999),
            'n_steps': [128, 512, 1024, 2048],
            'gae_lambda': (0.9, 0.99),
            'vf_coef': (0.5, 1.0)
        },
        'DDPG': {
            'learning_rate': (0.0001, 0.01),
            'gamma': (0.8, 0.999),
            'batch_size': [64, 128, 256],
            'tau': (0.005, 0.01),
            'buffer_size': [10000, 100000, 1000000]
        },
        'SAC': {
            'learning_rate': (0.0001, 0.01),
            'gamma': (0.8, 0.999),
            'batch_size': [64, 128, 256],
            'tau': (0.005, 0.01),
            'ent_coef': (0.1, 1.0)
        },
        'TD3': {
            'learning_rate': (0.0001, 0.01),
            'gamma': (0.8, 0.999),
            'batch_size': [64, 128, 256],
            'tau': (0.005, 0.01),
            'policy_delay': [1, 2]
        }
    }

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
            print(f"\nRunning analysis for {algorithm} on {train_env} -> {test_env}...")
            
            # Initialize the analyzer for the current algorithm and environment pair
            cross_env_analyzer = CrossEnvRLHyperparameter(
                train_env_name=train_env,  # Training environment
                test_env_name=test_env,  # Testing environment 
                algorithm=algorithm,
                param_grid=param_grid,
                n_samples=n_samples,
                random_state=seed,  # Use the random seed
                train_steps=train_steps,
                eval_episodes=eval_episodes,
                log_dir=os.path.join(log_dir, algorithm, train_env),
                device=device
            )
            
            # Build dataset
            results = cross_env_analyzer.build_dataset(save_path=os.path.join(log_dir, algorithm, train_env, "results.csv"))
            combined_results.append(results)
            print("Results dataframe:")
            print(results.head())
        
        # Combine results from all environment pairs
        combined_results_df = pd.concat(combined_results, ignore_index=True)
        os.makedirs(os.path.join(log_dir, algorithm), exist_ok=True)
        combined_results_df.to_csv(os.path.join(log_dir, algorithm, "combined_results.csv"), index=False)
        print(f"param_grid: {param_grid}")
        print(f"seed: {seed}")
        explainer = SHAPExplainer(param_grid=param_grid, 
                                results=combined_results_df, 
                                log_dir=os.path.join(log_dir, algorithm), 
                                random_state=seed)

        # Analyze results
        explainer.plot_summary(target)
        # Analyze important hyperparameters
        for param in param_grid.keys():
            explainer.plot_dependence(param, target)
        # Print optimal hyperparameter combinations
        explainer.print_optimal_hyperparams(target)
        
        print(f"Cross-environment analysis complete for {algorithm}. Results saved in:", cross_env_analyzer.log_dir)


def parse_env_pair(s):
    """Parse environment pair string in format 'env1,env2'"""
    try:
        train_env, test_env = s.split(',')
        return (train_env.strip(), test_env.strip())
    except ValueError:
        raise argparse.ArgumentTypeError("Environment pair must be formatted as 'env1,env2'")


# In actual use, you can run the following example
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Cross-Environment RL Hyperparameter SHAP Analysis')
    parser.add_argument('--algorithms', nargs='+', choices=['PPO', 'A2C', 'DDPG', 'SAC', 'TD3'],
                        help='RL algorithms to analyze')
    parser.add_argument('--env_pairs', type=parse_env_pair, nargs='+', 
                        help='Environment pairs to test (format: "env1,env2")')
    parser.add_argument('--n_samples', type=int, default=5,
                        help='Number of hyperparameter combinations to sample')
    parser.add_argument('--train_steps', type=int, default=100,
                        help='Number of training steps for each run')
    parser.add_argument('--eval_episodes', type=int, default=10,
                        help='Number of episodes for evaluation')
    parser.add_argument('--target', type=str, default="generalization_gap",
                        help='Target variable for SHAP analysis')
    parser.add_argument('--log_dir', type=str, default="./rl_cross_env_analysis_results",
                        help='Directory for saving results')
    parser.add_argument('--seed', type=int, default=None,  # Changed default to None for random seed
                        help='Random seed (default: random value)')
    parser.add_argument('--device', type=str, default="cpu",
                        help='Device to use for training (cpu or cuda)')
    
    args = parser.parse_args()
    
    # Set global random seed if provided
    if args.seed is not None:
        np.random.seed(args.seed)
    else:
        # Generate a random seed and use it
        random_seed = random.randint(0, 100000)
        np.random.seed(random_seed)
        args.seed = random_seed
    
    # Run the analysis with command line arguments
    cross_environment_analysis_example(
        algorithms=args.algorithms,
        env_pairs=args.env_pairs,
        n_samples=args.n_samples,
        train_steps=args.train_steps,
        eval_episodes=args.eval_episodes,
        target=args.target,
        log_dir=args.log_dir,
        seed=args.seed,
        device=args.device
    )
