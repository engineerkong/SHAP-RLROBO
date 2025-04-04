import pandas as pd
import os
from RLHyperparameterSHAP import SHAPExplainer

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