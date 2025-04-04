import numpy as np
import pandas as pd
import gymnasium
import random # Import random module for generating random seeds
from tqdm import tqdm
import os
from stable_baselines3 import PPO, A2C, DDPG, SAC, TD3
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

class RLHyperparameter:
    def __init__(self, env_name, algorithm, param_grid, n_samples=50, seed=None, 
                 train_steps=100000, eval_episodes=10, log_dir=None, device="cuda:0"):
        """
        Initialize RL hyperparameter SHAP analysis
        
        Parameters:
        -----------
        env_name : str
            OpenAI Gym environment name
        param_grid : dict
            Hyperparameter range, including algorithm selection
        n_samples : int
            Number of hyperparameter combinations to sample
        seed : int
            Random or fixed seed
        train_steps : int
            Total steps to train the RL algorithm
        eval_episodes : int
            Number of episodes to use for evaluation
        log_dir : str, optional
            Log directory path, if None, no logging
        device : str
            Computing device ('cpu' or 'cuda')
        """
        self.env_name = env_name
        self.algorithm = algorithm
        self.param_grid = param_grid
        self.n_samples = n_samples
        self.seed = seed
        self.train_steps = train_steps
        self.eval_episodes = eval_episodes
        
        self.log_dir = log_dir if log_dir else "./rl_hyperparams_logs"
        os.makedirs(self.log_dir, exist_ok=True)

        self.device = device
        self.results = None
    
    def sample_hyperparameters(self):
        """Randomly sample hyperparameter combinations"""
        sampled_params = []
        param_names = list(self.param_grid.keys())
        
        for _ in range(self.n_samples):
            params = {}
            for param_name in param_names:
                param_values = self.param_grid[param_name]
                
                # Continuous parameters
                if isinstance(param_values, tuple) and len(param_values) == 2:
                    min_val, max_val = param_values
                    
                    # Log-uniform sampling (suitable for learning rate, etc.)
                    if param_name in ['learning_rate', 'gamma']:
                        params[param_name] = np.exp(np.random.uniform(np.log(min_val), np.log(max_val)))
                    else:
                        params[param_name] = np.random.uniform(min_val, max_val)
                
                # Discrete parameters
                else:
                    params[param_name] = np.random.choice(param_values)
            
            sampled_params.append(params)
            
        return sampled_params, param_names
    
    def evaluate_generalization(self, params, run_id):
        """Train the RL algorithm with given hyperparameters and evaluate generalization performance"""
        # Create a separate log directory for each run
        run_log_dir = os.path.join(self.log_dir, f"run_{run_id}")
        os.makedirs(run_log_dir, exist_ok=True)
        
        # Create training environment - use standard seed
        train_env = gymnasium.make(self.env_name)
        observation, info = train_env.reset(seed=self.seed)
        train_env = Monitor(train_env, os.path.join(run_log_dir, "train_monitor"))
        
        # Create testing environment - use different seed
        test_env = gymnasium.make(self.env_name)
        observation, info = test_env.reset(seed=self.seed + 100)
        test_env = Monitor(test_env, os.path.join(run_log_dir, "test_monitor"))
        
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
        
        # Train the model
        model.learn(total_timesteps=self.train_steps, progress_bar=True)
        
        # Evaluate training environment performance
        train_mean_reward, train_std_reward = evaluate_policy(model, train_env, n_eval_episodes=self.eval_episodes, deterministic=True)
        
        # Evaluate testing environment performance
        test_mean_reward, test_std_reward = evaluate_policy(model, test_env, n_eval_episodes=self.eval_episodes, deterministic=True)
        
        # Calculate generalization gap
        gen_gap = (test_mean_reward - train_mean_reward) / train_mean_reward if train_mean_reward != 0 else 0
        
        # Save the model
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
    
    def build_dataset(self, save_path=None):
        """Build hyperparameter-performance dataset"""
        sampled_params, param_names = self.sample_hyperparameters()
        
        results = []
        for i, params in enumerate(tqdm(sampled_params, desc="Evaluating hyperparameter combinations")):
            print(f"\nEvaluating combination {i+1}/{len(sampled_params)}:")
            for param_name, param_value in params.items():
                print(f"  {param_name}: {param_value}")
                
            # Generate a random seed and use it (TODO: ignored orignal seed)
            seed = random.randint(0, 100000)
            np.random.seed(seed)
            self.seed = seed
            print(f"Using random seed on RLROBO: {self.seed}")

            perf_metrics = self.evaluate_generalization(params, run_id=i)
            result = {**params, **perf_metrics}
            results.append(result)
            
            # Print current results
            print(f"Results: Train reward: {perf_metrics['train_reward']:.2f} ± {perf_metrics['train_reward_std']:.2f}, "
                  f"Test reward: {perf_metrics['test_reward']:.2f} ± {perf_metrics['test_reward_std']:.2f}, "
                  f"Gen gap: {perf_metrics['generalization_gap']:.2f}")
            
        self.results = pd.DataFrame(results)
        
        # Filter out invalid results
        self.results = self.results.dropna()
        
        if save_path:
            self.results.to_csv(save_path, index=False)
            
        return self.results
