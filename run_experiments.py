import argparse
from .scripts.CrossEnvRLHyperparameter import cross_environment_analysis_example

def parse_env_pair(s):
    """Parse environment pair string in format 'env1,env2'"""
    try:
        train_env, test_env = s.split(',')
        return (train_env.strip(), test_env.strip())
    except ValueError:
        raise argparse.ArgumentTypeError("Environment pair must be formatted as 'env1,env2'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Cross-Environment RL Hyperparameter SHAP Analysis')
    parser.add_argument('--algorithms', nargs='+', choices=['PPO', 'A2C', 'DDPG', 'SAC'],
                        help='RL algorithms to analyze')
    parser.add_argument('--env_pairs', type=parse_env_pair, nargs='+', 
                        help='Environment pairs to test (format: "train_env,test_env")')
    parser.add_argument('--direction', type=str, default="forward", choices=['forward', 'reverse'],
                        help='Direction: "forward" (MuJoCo→PyBullet) or "reverse" (PyBullet→MuJoCo)')
    parser.add_argument('--n_samples', type=int, default=1,
                        help='Number of hyperparameter combinations to sample')
    parser.add_argument('--train_steps', type=int, default=100000,
                        help='Number of training timesteps for each run')
    parser.add_argument('--eval_episodes', type=int, default=10,
                        help='Number of episodes for evaluation')
    parser.add_argument('--target', type=str, default="generalization_gap",
                        help='Target variable for SHAP analysis (e.g., generalization_gap, test_reward)')
    parser.add_argument('--log_dir', type=str, default="./rl_cross_env_analysis_results",
                        help='Directory for saving results')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility (None for random)')
    parser.add_argument('--num_seeds', type=int, default=3,
                        help='Number of seed repetitions for robustness')
    parser.add_argument('--device', type=str, default="cuda:0",
                        help='Device to use for training (cpu or cuda:0)')
    
    args = parser.parse_args()
    
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
        device=args.device,
        num_seeds=args.num_seeds,
        direction=args.direction
    )