import pandas as pd
import os
import random
import matplotlib.pyplot as plt
import shap
import time
import numpy as np
from sklearn.ensemble import RandomForestRegressor

class SHAPExplainer:
    """
    A class for explaining the impact of hyperparameters on RL performance using SHAP values.
    This class encapsulates all explainability-related functionality.
    """
    def __init__(self, param_grid, results, log_dir, seed=None):
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
        seed : int
            Random or fixed seed
        """
        self.param_grid = param_grid
        self.results = results
        self.log_dir = log_dir
        self.seed = seed
        self.meta_model = None
        self.explainer = None
    
        # Set global fixed seed if provided
        if self.seed is not None:
            np.random.seed(self.seed)
            print(f"Using fixed seed on RLROBO: {self.seed}")
        else:
            # Generate a random seed and use it
            seed = random.randint(0, 100000)
            np.random.seed(seed)
            self.seed = seed
            print(f"Using random seed on SHAP: {self.seed}")

    def train_meta_model(self, target):
        """
        Train a meta-model to predict performance based on hyperparameters
        """
        X = self.results[list(self.param_grid.keys())]
        y = self.results[target]
        
        # Use random forest as the meta-model
        self.meta_model = RandomForestRegressor(n_estimators=100)
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
        # Set x-axis range as a factor of the original range
        x_min, x_max = plt.xlim()  # Get current limits
        x_range = x_max - x_min
        factor = 0.6  # Adjust this factor as needed (1.2 = 20% wider than original range)
        center = (x_max + x_min) / 2
        plt.xlim(center - x_range * factor / 2, center + x_range * factor / 2)
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
        
    def plot_decision(self, target):
        """
        Plot decision tree for the meta-model
        """        
        plt.figure(figsize=(12, 8))
        shap.plots.decision(self.meta_model, self.results[list(self.param_grid.keys())])
        plt.title(f'Meta-model Decision Tree for {target}')
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, f'shap_decision_{target}.png'))

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
    # 'TD3': {
    #     'learning_rate': (0.0001, 0.01),
    #     'gamma': (0.8, 0.999),
    #     'batch_size': [64, 128, 256],
    #     'tau': (0.005, 0.01),
    #     'policy_delay': [1, 2]
    # }
}
log_dir = "/home/lin30127/workspace/SHAP-RLROBO/results"

for algorithm, param_grid in rl_param_grids.items():
    combined_results = pd.read_csv(f"/home/lin30127/workspace/SHAP-RLROBO/results/{algorithm}_combined_results.csv")
    combined_train_reward = combined_results["train_reward"]
    combined_train_reward_std = combined_results["train_reward_std"]
    combined_test_reward = combined_results["test_reward"]
    combined_test_reward_std = combined_results["test_reward_std"]

    # Calculate the gap between test and train performance
    combined_results["gap"] = (combined_test_reward - combined_train_reward) # * (combined_test_reward_std / combined_train_reward_std)
    # Create DataFrame for SHAP analysis
    combined_results_df = combined_results

    target = "gap"  # Use the gap as the target variable for analysis
    # Combine results from all environment pairs
    os.makedirs(os.path.join(log_dir, algorithm), exist_ok=True)
    print(f"algorithm: {algorithm}")
    print(f"param_grid: {param_grid}")
    explainer = SHAPExplainer(param_grid=param_grid, 
                            results=combined_results_df, 
                            log_dir=os.path.join(log_dir, algorithm))

    # # Analyze results
    # explainer.plot_summary(target)
    # # Analyze important hyperparameters
    # for param in param_grid.keys():
    #     explainer.plot_dependence(param, target)
    # # Print optimal hyperparameter combinations
    # explainer.print_optimal_hyperparams(target)
    explainer.plot_interaction("learning_rate", "gamma", target)

print(f"Cross-environment analysis complete for {algorithm}. Results saved in {log_dir}.")