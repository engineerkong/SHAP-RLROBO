import pandas as pd
import os
import random
import matplotlib.pyplot as plt
import shap
import time
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib.colors as mcolors
import argparse

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
    
    def plot_summary(self, algorithm, target):
        """
        Plot SHAP summary visualizations
        """
        shap_values, X = self.explain(target)
        
        # Beeswarm plot showing feature impacts
        plt.figure(figsize=(12, 8))
        shap.plots.violin(shap_values, X, plot_type="layered_violin")
        plt.title(f'SHAP Summary Plot for Gap using {algorithm}')
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, f'shap_summary_{algorithm}.pdf'))

    def plot_dependence(self, param_name, target, color, interaction_index=None):
        """
        Plot dependence of target on a specific parameter
        """
        shap_values, X = self.explain(target)
        
        plt.figure(figsize=(10, 6))
        shap.dependence_plot(
            param_name, 
            shap_values, 
            X, 
            color=color,
            interaction_index=interaction_index
        )
        plt.title(f'SHAP Dependence Plot for {param_name} on Gap')
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, f'shap_dependence_{param_name}_{target}.pdf'))

    def plot_importance(self, env, target):
        """
        Plot SHAP importance visualizations
        """                
        shap_values, X = self.explain(target)
        
        # Bar plot of feature importance
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X, plot_type="bar", color=['red', 'blue', 'yellow'])
        plt.title(f'SHAP Value Importance for Gap in {env}')
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, f'shap_importance_{env}.pdf'))
        
        return shap_values, X  # Return values for combined plots

    def plot_combined_importance(self, environments, target, log_dir):
        """
        Plot SHAP importance visualizations for multiple environments in one figure
        
        Parameters:
        -----------
        environments : list
            List of environment names to combine in the plot
        target : str
            Target metric for analysis
        log_dir : str
            Directory to save the combined plot
        """
        plt.figure(figsize=(14, 8))
        
        # Define feature colors and environment styles
        feature_colors = {
            'algorithm': '#1f77b4',  # blue
            'learning_rate': '#ff7f0e',  # orange
            'gamma': '#2ca02c',  # green
        }
        
        env_styles = {
            'InvertedPendulum': {'alpha': 0.9, 'hatch': '', 'edgecolor': 'black'},
            'HalfCheetah': {'alpha': 0.9, 'hatch': '///', 'edgecolor': 'black'},
            'Hopper': {'alpha': 0.9, 'hatch': '...', 'edgecolor': 'black'},
            'Walker2d': {'alpha': 0.9, 'hatch': 'xxx', 'edgecolor': 'black'}
        }
        
        # Calculate bar positions
        features = list(feature_colors.keys())
        n_features = len(features)
        n_envs = len(environments)
        bar_width = 0.8 / n_envs
        
        # Collect importance values for each environment and feature
        importance_values = {}
        
        for env_idx, env in enumerate(environments):
            # Load environment data
            env_data = pd.read_csv(os.path.join(log_dir, f"{env}_processed_results.csv"))
            env_explainer = SHAPExplainer(
                param_grid={'algorithm': [0, 1, 2, 3], 'learning_rate': (0.0001, 0.01), 'gamma': (0.8, 0.999)},
                results=env_data,
                log_dir=os.path.join(log_dir, env),
                seed=self.seed  # Use the same seed for consistency
            )
            
            # Get SHAP values
            shap_values, X = env_explainer.explain(target)
            
            # Calculate feature importance (mean absolute SHAP value for each feature)
            importance = np.abs(shap_values).mean(axis=0)
            feature_names = list(X.columns)
            
            # Store importance values
            importance_values[env] = {feature_names[i]: importance[i] for i in range(len(feature_names))}
        
        # Create grouped bar chart
        ax = plt.gca()
        
        for feature_idx, feature in enumerate(features):
            for env_idx, env in enumerate(environments):
                # Position for this bar
                pos = feature_idx + (env_idx - n_envs/2 + 0.5) * bar_width
                
                # Get importance value
                imp_value = importance_values[env].get(feature, 0)
                
                # Plot bar
                bar = ax.bar(
                    pos, imp_value, bar_width,
                    color=feature_colors[feature],
                    label=f'{env}' if feature_idx == 0 else "",  # Only add env to legend once
                    **env_styles[env]
                )
        
        # Add feature dividers
        for i in range(n_features-1):
            plt.axvline(i + 0.5, color='gray', linestyle='--', alpha=0.3)
        
        # Set x-axis labels and ticks
        plt.xticks(range(n_features), features)
        plt.xlabel('Features')
        plt.ylabel('Mean |SHAP Value|')
        plt.title(f'Combined SHAP Value Importance Across Environments')
        
        # Create legend for environments
        env_handles = [plt.Rectangle((0,0),1,1, **env_styles[env], color='gray') for env in environments]
        env_legend = plt.legend(env_handles, environments, loc='upper right', title='Environments', 
                               fontsize=14, title_fontsize=16)
        ax.add_artist(env_legend)
        
        # Create legend for features
        feature_handles = [plt.Rectangle((0,0),1,1, color=color) for color in feature_colors.values()]
        plt.legend(feature_handles, features, loc='upper left', title='Features', 
                  fontsize=14, title_fontsize=16)
        
        plt.tight_layout()
        plt.savefig(os.path.join(log_dir, f'combined_importance_{target}.pdf'))
        plt.savefig(os.path.join(log_dir, f'combined_importance_{target}.png'))
        print(f"Saved combined importance plot to {os.path.join(log_dir, f'combined_importance_{target}.pdf')}")

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

        # X["feature_3_bin"] = pd.qcut(X["feature_3"], q=4)  # Bin into quartiles
        # for bin_name in X["feature_3_bin"].unique():
        #     subset = X[X["feature_3_bin"] == bin_name]
        #     shap.dependence_plot("feature_1", shap_values[subset.index], subset, 
        #                         interaction_index="feature_2", 
        #                         title=f"Feature 3 Bin: {bin_name}")

    def print_optimal_hyperparams(self, target):
        """
        Print and save the best hyperparameter combination
        """
        # We maximize the target metric
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
        
    def plot_decision(self, target):
        """
        Plot decision tree for the meta-model
        """        
        plt.figure(figsize=(12, 8))
        shap.plots.decision(self.meta_model, self.results[list(self.param_grid.keys())])
        plt.title(f'Meta-model Decision Tree for {target}')
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, f'shap_decision_{target}.png'))
        
    def find_optimal_config_and_plot(self, target):
        """
        Find the optimal algorithm and hyperparameters across all results and 
        plot decision and force plots for this optimal configuration
        
        Parameters:
        -----------
        target : str
            Target metric for optimization (e.g., 'gap')
        """
        # Find the optimal configuration (maximizing the target metric)
        best_idx = self.results[target].idxmax()
        best_params = self.results.loc[best_idx, list(self.param_grid.keys())]
        best_performance = self.results.loc[best_idx, target]
        
        # Print the optimal configuration
        print(f"\nOptimal configuration for maximizing {target}:")
        for param, value in best_params.items():
            print(f"  {param}: {value}")
        print(f"{target} value: {best_performance}")
        
        # Save the optimal configuration
        best_params_dict = {param: value for param, value in best_params.items()}
        best_params_dict['performance'] = best_performance
        pd.Series(best_params_dict).to_csv(
            os.path.join(self.log_dir, f'optimal_config_{target}.csv')
        )
        
        # Ensure meta-model is trained
        if self.meta_model is None:
            self.train_meta_model(target)
            
        # Get SHAP values for explanation
        X = self.results[list(self.param_grid.keys())]
        if self.explainer is None:
            self.explainer = shap.TreeExplainer(self.meta_model)
        shap_values = self.explainer.shap_values(X)
        
        # # Plot decision plot for the optimal configuration
        # plt.figure(figsize=(12, 8))
        # shap.plots.decision(self.meta_model, X, feature_names=list(self.param_grid.keys()))
        # plt.title(f'Decision Plot for Optimal Configuration ({target})')
        # plt.tight_layout()
        # plt.savefig(os.path.join(self.log_dir, f'optimal_decision_{target}.png'))
        
        # Plot force plot for the optimal configuration
        plt.figure(figsize=(14, 6))
        # Get the index of the optimal configuration
        optimal_idx = best_idx
        # Create force plot for the optimal configuration
        shap.force_plot(
            self.explainer.expected_value, 
            shap_values[optimal_idx:optimal_idx+1, :], 
            X.iloc[optimal_idx:optimal_idx+1, :],
            matplotlib=True,
            show=False
        )
        plt.title(f'Force Plot for Optimal Configuration ({target})')
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, f'optimal_force_{target}.pdf'))
        
        # Return the optimal configuration for reference
        return best_params_dict

def process_results(rl_param_grids, log_dir):
    """
    Process results for each algorithm and perform SHAP analysis
    """
    all_algorithms_df = pd.DataFrame()
    all_param_keys = set()
    algorithm_encoding = {'PPO': 0, 'A2C': 1, 'DDPG': 2, 'SAC': 3}
    env_df = {"InvertedPendulum":pd.DataFrame(), "HalfCheetah":pd.DataFrame(), "Hopper":pd.DataFrame(), "Walker2d":pd.DataFrame()}
    for algorithm, param_grid in rl_param_grids.items():
        combined_results = pd.read_csv(f"/home/lin30127/workspace/SHAP-RLROBO/results/{algorithm}_combined_results.csv")
        combined_train_reward = combined_results["train_reward"]
        combined_test_reward = combined_results["test_reward"]

        # Calculate the gap between test and train performance
        combined_results["gap"] = (combined_test_reward - combined_train_reward)
        # Add algorithm as a feature
        combined_results["algorithm"] = algorithm_encoding[algorithm]
        # Save processed data to new CSV
        output_path = f"{log_dir}/{algorithm}_processed_results.csv"
        combined_results.to_csv(output_path, index=False)
        print(f"Saved processed results for {algorithm} to {output_path}")

        env_df['InvertedPendulum'] = pd.concat([env_df['InvertedPendulum'], combined_results[2:502]], ignore_index=True)
        env_df['HalfCheetah'] = pd.concat([env_df['HalfCheetah'], combined_results[502:1002]], ignore_index=True)
        env_df['Hopper'] = pd.concat([env_df['Hopper'], combined_results[1002:1502]], ignore_index=True)
        env_df['Walker2d'] = pd.concat([env_df['Walker2d'], combined_results[1502:2002]], ignore_index=True)
        all_algorithms_df = pd.concat([all_algorithms_df, combined_results], ignore_index=True)

    for env, df in env_df.items():
        # Save processed data to new CSV
        output_path = f"{log_dir}/{env}_processed_results.csv"
        df.to_csv(output_path, index=False)
        print(f"Saved processed results for {env} to {output_path}")
    all_algorithms_df.to_csv(os.path.join(log_dir, "all_algorithms_combined_results.csv"), index=False)
    print(f"Saved combined results for all algorithms to {os.path.join(log_dir, 'all_algorithms_combined_results.csv')}")

def main():
    parser = argparse.ArgumentParser(description="SHAP Analysis for RL experiments")
    parser.add_argument('--process', type=str, help='Process to be chosen', default='all')
    args = parser.parse_args()

    log_dir = "/home/lin30127/workspace/SHAP-RLROBO/results/experiments/"
    target = "gap"  # Use the gap as the target variable for analysis  

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
        }
    }
    if args.process == 'all' or args.process == 'process':
        print("=== Processing results for all algorithms ===")
        # Process results for each algorithm
        process_results(rl_param_grids, log_dir)
    if args.process == 'all' or args.process == 'exp1':
        print("=== SHAP Analysis for Experiment 1 ===")
        # # Experiment 1
        for algorithm, param_grid in rl_param_grids.items():
            # Read the combined results for each algorithm
            processed_results = pd.read_csv(f"{log_dir}{algorithm}_processed_results.csv")
            os.makedirs(os.path.join(log_dir, algorithm), exist_ok=True)
            print(f"algorithm: {algorithm}")
            print(f"param_grid: {param_grid}")
            explainer = SHAPExplainer(param_grid=param_grid, 
                                    results=processed_results, 
                                    log_dir=os.path.join(log_dir, algorithm))
            explainer.plot_summary(algorithm, target)
    if args.process == 'all' or args.process == 'exp2':
        print("=== SHAP Analysis for Experiment 2 ===")
        # Experiment 2
        combined_param_grid = {'algorithm': [0, 1, 2, 3], 'learning_rate': (0.0001, 0.01), 'gamma': (0.8, 0.999)}    
        all_algorithms_df = pd.read_csv(os.path.join(log_dir, "all_algorithms_combined_results.csv"))
        os.makedirs(os.path.join(log_dir, "combined_analysis"), exist_ok=True)
        combined_explainer = SHAPExplainer(
            param_grid=combined_param_grid,
            results=all_algorithms_df,
            log_dir=os.path.join(log_dir, "combined_analysis")
        )    
        combined_explainer.plot_dependence("algorithm", target, '#1f77b4') # experiment 2
        combined_explainer.plot_dependence("learning_rate", target, '#ff7f0e') # experiment 2
        combined_explainer.plot_dependence("gamma", target, '#2ca02c') # experiment 2
    if args.process == 'all' or args.process == 'exp3':
        print("=== SHAP Analysis for Experiment 3 ===")
        # Experiment 3
        combined_param_grid = {'algorithm': [0, 1, 2, 3], 'learning_rate': (0.0001, 0.01), 'gamma': (0.8, 0.999)}    
        envs = ["InvertedPendulum", "HalfCheetah", "Hopper", "Walker2d"]
        
        # Create a single explainer for the combined plot
        base_explainer = SHAPExplainer(
            param_grid=combined_param_grid,
            results=pd.DataFrame(),  # Empty dataframe, will load per-environment data in the method
            log_dir=log_dir
        )
        
        # Generate combined importance plot across environments
        base_explainer.plot_combined_importance(envs, target, log_dir)
    if args.process == 'all' or args.process == 'exp4':
        print("=== SHAP Analysis for Experiment 4 ===")
        # Experiment 4
        optimal_param_grid = {'learning_rate': (0.0001, 0.01), 'gamma': (0.8, 0.999), 'n_steps': [128, 512, 1024, 2048], 
                            'gae_lambda': (0.9, 0.99), 'vf_coef': (0.5, 1.0)}    
        A2C_df = pd.read_csv(os.path.join(log_dir, "A2C_processed_results.csv"))
        os.makedirs(os.path.join(log_dir, "optimal_analysis"), exist_ok=True)
        optimal_explainer = SHAPExplainer(
            param_grid=optimal_param_grid,
            results=A2C_df,
            log_dir=os.path.join(log_dir, "optimal_analysis")
        )    
        # Find optimal algorithm and hyperparameters across all results and plot
        print("\n=== Finding optimal configuration across all algorithms ===")
        optimal_config = optimal_explainer.find_optimal_config_and_plot(target)

if __name__ == "__main__":
    main()
