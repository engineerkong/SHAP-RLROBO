"""
SHAP Analysis Pipeline for Reinforcement Learning Experiments

This script combines multiple SHAP explainer tools to analyze RL algorithm performance:
- SHAPImpactExplainer: Compare feature impacts across algorithms
- SHAPInteractionExplainer: Analyze feature interactions
- SHAPTransferExplainer: Analyze transfer learning across tasks
- SHAPConfigAnalyzer: Identify optimal configurations
"""

import pandas as pd
from .scripts.SHAPImpactExplainer import SHAPImpactExplainer
from .scripts.SHAPInteractionExplainer import SHAPInteractionExplainer
from .scripts.SHAPTransferExplainer import SHAPTransferExplainer
from .scripts.SHAPConfigAnalyzer import SHAPConfigAnalyzer


def define_parameter_grids():
    """Define parameter grids for all RL algorithms"""
    param_grids = {
        'PPO': {
            'learning_rate': (0.0001, 0.01), 
            'gamma': (0.8, 0.999), 
            'clip_range': (0.1, 0.3), 
            'n_steps': [128, 512, 1024, 2048]
        },
        'A2C': {
            'learning_rate': (0.0001, 0.01), 
            'gamma': (0.9, 0.999),
            'gae_lambda': (0.9, 0.99), 
            'vf_coef': (0.25, 1.0)
        },
        'DDPG': {
            'learning_rate': (0.0001, 0.01), 
            'gamma': (0.8, 0.999),
            'tau': (0.001, 0.01), 
            'buffer_size': [10000, 100000, 1000000]
        },
        'SAC': {
            'learning_rate': (0.0001, 0.01), 
            'gamma': (0.8, 0.999),
            'tau': (0.001, 0.01), 
            'ent_coef': (0.01, 0.2)
        }
    }
    return param_grids


def define_algorithms_dict():
    """Define file paths for algorithm results"""
    algorithms_dict = {
        'PPO': 'results_experiments/exp12/combined_PPO_averaged.csv',
        'A2C': 'results_experiments/exp12/combined_A2C_averaged.csv',
        'DDPG': 'results_experiments/exp12/combined_DDPG_averaged.csv',
        'SAC': 'results_experiments/exp12/combined_SAC_averaged.csv'
    }
    return algorithms_dict


def analyze_algorithm_impact(algorithms_dict, param_grids, target="generalization_gap"):
    """
    Compare feature impacts across all algorithms using beeswarm plots
    
    Args:
        algorithms_dict: Dictionary mapping algorithm names to result file paths
        param_grids: Dictionary of parameter grids for each algorithm
        target: Target variable to analyze (default: generalization_gap)
    """
    print("\n" + "="*80)
    print("ANALYSIS 1: Algorithm Impact Comparison")
    print("="*80)
    
    explainer = SHAPImpactExplainer(results=None, param_grid=None, log_dir="./analysis_logs")
    explainer.plot_beeswarm_comparison(algorithms_dict, param_grids, target=target)
    print(f"✓ Generated beeswarm comparison for {target}")


def analyze_feature_interactions(algorithms_dict, param_grids, target="generalization_gap"):
    """
    Analyze feature interactions across algorithms
    
    Args:
        algorithms_dict: Dictionary mapping algorithm names to result file paths
        param_grids: Dictionary of parameter grids for each algorithm
        target: Target variable to analyze (default: generalization_gap)
    """
    print("\n" + "="*80)
    print("ANALYSIS 2: Feature Interaction Analysis")
    print("="*80)
    
    # Compare interaction metrics across algorithms
    explainer = SHAPInteractionExplainer(None, None, "./analysis_logs")
    explainer.plot_interaction_metrics_comparison(
        algorithms_dict, 
        param_grids, 
        target=target,
        metric='shap_interaction'
    )
    print(f"✓ Generated interaction metrics comparison for {target}")
    
    # Individual algorithm analysis - DDPG example
    print("\nAnalyzing DDPG interactions in detail...")
    ddpg_results = pd.read_csv(algorithms_dict['DDPG'])
    explainer_ddpg = SHAPInteractionExplainer(
        results=ddpg_results,
        param_grid=param_grids['DDPG'],
        log_dir="./analysis_logs_ddpg"
    )
    explainer_ddpg.plot_feature_interaction_grid_binned(
        algorithm='DDPG',
        target=target,
        feat1='learning_rate',
        feat2='gamma',
    )
    print("✓ Generated DDPG feature interaction grid")


def analyze_transfer_learning(target='generalization_gap'):
    """
    Analyze transfer learning across multiple tasks
    
    Args:
        target: Target variable to analyze (default: generalization_gap)
    """
    print("\n" + "="*80)
    print("ANALYSIS 3: Transfer Learning Analysis")
    print("="*80)
    
    # Define unified parameter grid for all algorithms
    param_grid = {
        'algorithm': [0, 1, 2, 3],  # PPO=0, A2C=1, DDPG=2, SAC=3
        'learning_rate': (0.0001, 0.01),
        'gamma': (0.8, 0.999),
        'clip_range': (0.1, 0.3), 
        'n_steps': [128, 512, 1024, 2048],  # PPO
        'gae_lambda': (0.9, 0.99), 
        'vf_coef': (0.25, 1.0),  # A2C
        'tau': (0.001, 0.01), 
        'buffer_size': [10000, 100000, 1000000],  # DDPG
        'ent_coef': (0.01, 0.2)  # SAC
    }
    
    # Load multi-task results
    task_files = {
        'InvertedPendulum(MJ-PB)': "results_experiments/exp3/combined_InvertedPendulum-v5_averaged.csv",
        'HalfCheetah(MJ-PB)': "results_experiments/exp3/combined_HalfCheetah-v5_averaged.csv",
        'Hopper(MJ-PB)': "results_experiments/exp3/combined_Hopper-v5_averaged.csv",
        'Walker2d(MJ-PB)': "results_experiments/exp3/combined_Walker2d-v5_averaged.csv"
    }
    
    print(f"Loading {len(task_files)} task results...")
    results_dict = {task: pd.read_csv(path) for task, path in task_files.items()}
    
    # Initialize explainer
    explainer = SHAPTransferExplainer(
        param_grid, 
        results_dict, 
        log_dir='./analysis_logs_transfer', 
        seed=42
    )
    
    # Analyze all tasks separately
    print(f"\nAnalyzing {target} for learning_rate across all tasks...")
    explainer.analyze_all_tasks(target=target, feature='learning_rate')
    print("✓ Generated individual task analyses")
    
    # Combined plot with trends
    print("\nGenerating combined task dependence plot...")
    explainer.plot_combined_tasks_dependence(
        target=target, 
        feature='learning_rate',
        show_trends=True, 
        alpha=0.3
    )
    print("✓ Generated combined tasks dependence plot")
    
    # Two-feature panel plot
    print("\nGenerating two-feature panel plot...")
    explainer.plot_two_features_combined(
        target=target, 
        features=['learning_rate', 'gamma'],
        show_trends=True, 
        alpha=0.3
    )
    print("✓ Generated two-feature comparison plot")


def analyze_optimal_configurations(data_path='./results_experiments/exp4/combined_all_averaged.csv', 
                                   n_samples=1000, 
                                   n_top=3):
    """
    Identify optimal configurations using SHAP analysis
    
    Args:
        data_path: Path to combined results CSV
        n_samples: Number of samples to use for analysis
        n_top: Number of top configurations to identify
    """
    print("\n" + "="*80)
    print("ANALYSIS 4: Optimal Configuration Analysis")
    print("="*80)
    
    print(f"Analyzing configurations from {data_path}...")
    analyzer = SHAPConfigAnalyzer(data_path)
    model, explainer = analyzer.analyze(n_samples=n_samples, n_top=n_top)
    print(f"✓ Identified top {n_top} configurations")
    
    return model, explainer


def main():
    """
    Main pipeline to run all SHAP analyses
    """
    print("="*80)
    print("SHAP ANALYSIS PIPELINE FOR RL EXPERIMENTS")
    print("="*80)
    
    # Define parameter grids and algorithm paths
    param_grids = define_parameter_grids()
    algorithms_dict = define_algorithms_dict()
    
    # Set target variable
    target = "generalization_gap"
    
    try:
        # Run all analyses
        analyze_algorithm_impact(algorithms_dict, param_grids, target)
        analyze_feature_interactions(algorithms_dict, param_grids, target)
        analyze_transfer_learning(target)
        model, explainer = analyze_optimal_configurations()
        
        print("\n" + "="*80)
        print("ALL ANALYSES COMPLETED SUCCESSFULLY!")
        print("="*80)
        print("\nOutput directories:")
        print("  - ./analysis_logs - Algorithm impact comparisons")
        print("  - ./analysis_logs_ddpg - DDPG interaction analysis")
        print("  - ./analysis_logs_transfer - Transfer learning analysis")
        
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        raise


if __name__ == "__main__":
    main()