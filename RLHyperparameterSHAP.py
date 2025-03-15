import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import gymnasium as gym
import time
from tqdm import tqdm
import os
from sklearn.ensemble import RandomForestRegressor
from stable_baselines3 import PPO, A2C, DDPG, SAC, TD3
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

class RLHyperparameter:
    def __init__(self, env_name, algorithm, param_grid, n_samples=50, random_state=42, 
                 train_steps=100000, eval_episodes=10, log_dir=None, device="cpu"):
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
        random_state : int
            Random seed
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
        self.random_state = random_state
        self.train_steps = train_steps
        self.eval_episodes = eval_episodes
        
        self.log_dir = log_dir if log_dir else "./rl_hyperparams_logs"
        os.makedirs(self.log_dir, exist_ok=True)

        self.device = device
        self.results = None
        
        np.random.seed(random_state)
    
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
        train_env = gym.make(self.env_name)
        observation, info = train_env.reset(seed=self.random_state)
        train_env = Monitor(train_env, os.path.join(run_log_dir, "train_monitor"))
        
        # Create testing environment - use different seed
        test_env = gym.make(self.env_name)
        observation, info = test_env.reset(seed=self.random_state + 100)
        test_env = Monitor(test_env, os.path.join(run_log_dir, "test_monitor"))
        
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
    

class SHAPExplainer:
    """
    A class for explaining the impact of hyperparameters on RL performance using SHAP values.
    This class encapsulates all explainability-related functionality.
    """
    def __init__(self, param_grid, results, log_dir, random_state=42):
        """
        Initialize the SHAP explainer
        
        Parameters:
        -----------
        param_grid : dict
            Hyperparameter grid used for sampling
        results : pandas.DataFrame
            Results dataframe containing hyperparameters and performance metrics
        log_dir : str
            Directory to save plots and results
        random_state : int
            Random seed for reproducibility
        """
        self.param_grid = param_grid
        self.results = results
        self.log_dir = log_dir
        self.random_state = random_state
        self.meta_model = None
        self.explainer = None
    
    def train_meta_model(self, target):
        """
        Train a meta-model to predict performance based on hyperparameters
        """
        X = self.results[list(self.param_grid.keys())]
        y = self.results[target]
        
        # Use random forest as the meta-model
        self.meta_model = RandomForestRegressor(n_estimators=100, random_state=self.random_state)
        self.meta_model.fit(X, y)
        
        # Evaluate meta-model performance
        r2_score = self.meta_model.score(X, y)
        print(f"Meta-model R² score for {target}: {r2_score:.4f}")
        
        return self.meta_model

    def explain(self, target):
        """
        Use SHAP to explain the impact of hyperparameters on the target metric
        """
        if self.meta_model is None:
            self.train_meta_model(target)
            
        X = self.results[list(self.param_grid.keys())]
        
        # Create SHAP explainer
        self.explainer = shap.TreeExplainer(self.meta_model)
        shap_values = self.explainer.shap_values(X)
        
        return shap_values, X
    
    def plot_summary(self, target):
        """
        Plot SHAP summary visualizations
        """
        shap_values, X = self.explain(target)
        
        # Bar plot of feature importance
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X, plot_type="bar")
        plt.title(f'SHAP Value Importance for {target}')
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, f'shap_importance_{target}.png'))
        
        # Beeswarm plot showing feature impacts
        plt.figure(figsize=(12, 8))
        # shap.summary_plot(shap_values, X)
        shap.plots.violin(shap_values, X, plot_type="layered_violin")
        plt.title(f'SHAP Summary Plot for {target}')
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, f'shap_summary_{target}.png'))
        
    def plot_dependence(self, param_name, target, interaction_index=None):
        """
        Plot dependence of target on a specific parameter
        """
        shap_values, X = self.explain(target)
        
        plt.figure(figsize=(10, 6))
        shap.dependence_plot(
            param_name, 
            shap_values, 
            X, 
            interaction_index=interaction_index
        )
        plt.title(f'SHAP Dependence Plot for {param_name} on {target}')
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, f'shap_dependence_{param_name}_{target}.png'))

    def plot_interaction(self, param1, param2, target):
        """
        Plot interaction between two parameters
        """
        shap_values, X = self.explain(target)
        
        # Get indices corresponding to parameters
        param_names = list(self.param_grid.keys())
        idx1 = param_names.index(param1)
        idx2 = param_names.index(param2)
        
        plt.figure(figsize=(10, 8))
        shap.dependence_plot(
            idx1, 
            shap_values, 
            X,
            interaction_index=idx2
        )
        plt.title(f'Interaction between {param1} and {param2} on {target}')
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, f'shap_interaction_{param1}_{param2}_{target}.png'))
        
    def print_optimal_hyperparams(self, target):
        """
        Print and save the best hyperparameter combination
        """
        # For generalization gap, we want to minimize it; for rewards, we want to maximize them
        if target == 'generalization_gap':
            best_idx = self.results[target].idxmin()
            print(f"Best hyperparameters for minimizing {target}:")
        else:
            best_idx = self.results[target].idxmax()
            print(f"Best hyperparameters for maximizing {target}:")
            
        best_params = self.results.loc[best_idx, list(self.param_grid.keys())]
        best_performance = self.results.loc[best_idx, target]
        
        for param, value in best_params.items():
            print(f"  {param}: {value}")
        print(f"{target}: {best_performance}")
        
        # Save the best parameters
        best_params_dict = {param: value for param, value in best_params.items()}
        best_params_dict['performance'] = best_performance
        
        pd.Series(best_params_dict).to_csv(
            os.path.join(self.log_dir, f'best_params_{target}.csv')
        )


# Example usage
if __name__ == "__main__":
    # Define separate hyperparameter grids for each algorithm
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

    # Run the analysis separately for each algorithm
    for algorithm, param_grid in rl_param_grids.items():
        print(f"\nRunning analysis for {algorithm}...")
        
        # Initialize the analyzer for the current algorithm
        analyzer = RLHyperparameter(
            env_name='InvertedPendulum-v5',
            algorithm=algorithm,
            param_grid=param_grid,
            n_samples=5,  # Adjust as needed
            train_steps=100,  # Adjust as needed
            eval_episodes=10,
            log_dir=f"./rl_gym_sb3_hyperparams_results/{algorithm}"
        )
        
        # Build the dataset
        results = analyzer.build_dataset(save_path=f"./rl_gym_sb3_hyperparams_results/{algorithm}/results.csv")
        print("Results dataframe:")
        print(results.head())
        
        # Analyze different targets
        for target in ['train_reward', 'test_reward', 'generalization_gap']:
            print(f"\nAnalyzing {target} for {algorithm}...")
            analyzer.plot_summary(target=target)
            
            # Analyze the most important hyperparameters
            for param in param_grid.keys():
                analyzer.plot_dependence(param, target=target)
            
            # Print the best hyperparameters
            analyzer.print_optimal_hyperparams(target=target)
        
        print(f"Analysis complete for {algorithm}. Results saved in:", analyzer.log_dir)
