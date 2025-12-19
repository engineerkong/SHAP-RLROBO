import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import shap
import matplotlib.pyplot as plt


class SHAPConfigAnalyzer:
    """SHAP-based configuration analyzer for RL generalizability"""
    
    def __init__(self, csv_file, source_col='source'):
        self.csv_file = csv_file
        self.source_col = source_col
        self.df = None
        self.model = None
        self.explainer = None
        self.features = None
        self.X = None
        self.y = None
        
        self.param_ranges = {
            'learning_rate': (0.0001, 0.01),
            'gamma': (0.8, 0.999),
            'clip_range': (0.1, 0.3),
            'n_steps': [128, 512, 1024, 2048],
            'gae_lambda': (0.9, 0.99),
            'vf_coef': (0.25, 1.0),
            'tau': (0.001, 0.01),
            'buffer_size': [10000, 100000, 1000000],
            'ent_coef': (0.01, 0.2)
        }
        
        self.algo_info = {
            'PPO': {'code': 0, 'features': ['learning_rate', 'gamma', 'clip_range', 'n_steps']},
            'A2C': {'code': 1, 'features': ['learning_rate', 'gamma', 'gae_lambda', 'vf_coef']},
            'DDPG': {'code': 2, 'features': ['learning_rate', 'gamma', 'tau', 'buffer_size']},
            'SAC': {'code': 3, 'features': ['learning_rate', 'gamma', 'tau', 'ent_coef']}
        }
    
    def load_data(self):
        """Load and display dataset information"""
        self.df = pd.read_csv(self.csv_file)
        print(f"Loaded: {len(self.df)} rows")
        
        if self.source_col in self.df.columns:
            sources = sorted(self.df[self.source_col].unique())
            print(f"Sources ({len(sources)}): {sources}")
        
        if 'algorithm' in self.df.columns:
            algos = sorted(self.df['algorithm'].unique())
            print(f"Algorithms ({len(algos)}): {algos}")
        print()
        
        # Prepare features
        feature_candidates = ['algorithm', 'learning_rate', 'gamma', 'clip_range', 'n_steps', 
                            'gae_lambda', 'vf_coef', 'tau', 'buffer_size', 'ent_coef']
        self.features = [f for f in feature_candidates if f in self.df.columns]
        
        self.X = self.df[self.features].values
        self.y = self.df['generalization_gap'].values
    
    def train_model(self):
        """Train Random Forest meta-model"""
        print("Training meta-model...")
        self.model = RandomForestRegressor(
            n_estimators=200, 
            random_state=42, 
            n_jobs=-1, 
            max_depth=10
        )
        self.model.fit(self.X, self.y)
        print(f"Model R²: {self.model.score(self.X, self.y):.4f}\n")
    
    def compute_explainer(self):
        """Compute SHAP explainer"""
        print("Computing SHAP explainer...")
        self.explainer = shap.TreeExplainer(self.model)
    
    def generate_synthetic_configs(self, n_samples):
        """Generate synthetic configurations for all algorithms"""
        all_configs = []
        all_algos = []
        
        print(f"Generating synthetic configurations ({n_samples} per algorithm)...")
        for algo_name, info in self.algo_info.items():
            configs = self._generate_configs_for_algorithm(
                info['code'], 
                info['features'], 
                n_samples
            )
            all_configs.append(configs)
            all_algos.extend([algo_name] * n_samples)
            print(f"  {algo_name}: {n_samples} configs")
        
        all_configs = np.vstack(all_configs)
        print(f"\nTotal configs: {len(all_configs)}\n")
        
        return all_configs, all_algos
    
    def _generate_configs_for_algorithm(self, algo_code, relevant_features, n_samples):
        """Generate configs with algorithm encoding"""
        synthetic = np.full((n_samples, len(self.features)), np.nan)
        
        # Set algorithm code
        algo_idx = self.features.index('algorithm')
        synthetic[:, algo_idx] = algo_code
        
        # Fill relevant parameters
        for feat in relevant_features:
            feat_idx = self.features.index(feat)
            feat_range = self.param_ranges[feat]
            
            if isinstance(feat_range, tuple):
                synthetic[:, feat_idx] = np.random.uniform(
                    feat_range[0], feat_range[1], n_samples
                )
            else:
                synthetic[:, feat_idx] = np.random.choice(feat_range, n_samples)
        
        return synthetic
    
    def predict_and_explain(self, configs):
        """Predict generalization gaps and compute SHAP values"""
        print("Predicting generalization gaps...")
        predictions = self.model.predict(configs)
        
        print("Computing SHAP values...")
        shap_values = self.explainer.shap_values(configs)
        shap_sums = shap_values.sum(axis=1)
        
        return predictions, shap_values, shap_sums
    
    def find_best_worst_configs(self, shap_sums, n_top):
        """Find indices of best and worst configurations"""
        best_idx = np.argsort(shap_sums)[:n_top]
        worst_idx = np.argsort(shap_sums)[-n_top:][::-1]
        return best_idx, worst_idx
    
    def analyze_config(self, rank, idx, configs, predictions, shap_values, 
                      algorithm, config_type):
        """Analyze and print single configuration details"""
        config = configs[idx]
        relevant_features = self.algo_info[algorithm]['features']
        
        print(f"{config_type.upper()} RANK {rank}:")
        print(f"  Algorithm: {algorithm} (code={int(config[0])})")
        print(f"  Predicted Gap: {predictions[idx]:.4f}")
        print(f"  Total SHAP: {shap_values.sum(axis=1)[idx]:+.4f}\n")
        print(f"  Configuration:")
        
        for i, feat in enumerate(self.features):
            if not pd.isna(config[i]) and feat != 'algorithm':
                print(f"    {feat:15s}: {config[i]:.6f}")
        
        print(f"\n  Top SHAP Contributors:")
        valid_idx = [i for i, f in enumerate(self.features) 
                    if f != 'algorithm' and not pd.isna(config[i])]
        valid_shap = shap_values[idx][valid_idx]
        valid_names = [self.features[i] for i in valid_idx]
        
        top_contrib_idx = np.argsort(np.abs(valid_shap))[-3:][::-1]
        for ci in top_contrib_idx:
            print(f"    {valid_names[ci]:15s}: SHAP={valid_shap[ci]:+.4f}")
        print()
    
    def generate_force_plot(self, idx, configs, shap_values, predictions, 
                           algorithm, rank, config_type):
        """Generate SHAP force plot for a single configuration"""
        base_val = (self.explainer.expected_value[0] 
                   if isinstance(self.explainer.expected_value, np.ndarray) 
                   else self.explainer.expected_value)
        
        config = configs[idx]
        
        # Prepare display data
        feat_names, feat_vals, shap_disp = [], [], []
        for i, feat in enumerate(self.features):
            if not pd.isna(config[i]):
                if feat == 'algorithm':
                    feat_names.append(f"{feat}={algorithm}")
                    feat_vals.append(int(config[i]))
                else:
                    feat_names.append(feat)
                    feat_vals.append(config[i])
                shap_disp.append(shap_values[idx][i])
        
        # Create plot
        plt.figure(figsize=(20, 3))
        shap.force_plot(
            base_val, 
            np.array(shap_disp), 
            np.array(feat_vals),
            feature_names=feat_names, 
            matplotlib=True, 
            show=False
        )
        
        # Add title
        title = (f"{config_type.upper()} Rank {rank} - {algorithm} | "
                f"Gap: {predictions[idx]:.4f} | SHAP: {shap_values[idx].sum():+.4f}")
        plt.title(title, fontsize=14, pad=20)
        plt.tight_layout()
        
        # Save
        filename = f"force_plot_{config_type.lower()}_rank{rank}_{algorithm}.svg"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        
        return filename
    
    def generate_all_force_plots(self, best_idx, worst_idx, configs, shap_values, 
                                predictions, all_algos):
        """Generate force plots for all best and worst configurations"""
        print(f"\n{'='*100}\nGENERATING SHAP FORCE PLOTS\n{'='*100}\n")
        
        print("Generating BEST plots...")
        for rank, idx in enumerate(best_idx, 1):
            algo = all_algos[idx]
            filename = self.generate_force_plot(
                idx, configs, shap_values, predictions, algo, rank, 'best'
            )
            print(f"  Saved: {filename}")
        
        print("\nGenerating WORST plots...")
        for rank, idx in enumerate(worst_idx, 1):
            algo = all_algos[idx]
            filename = self.generate_force_plot(
                idx, configs, shap_values, predictions, algo, rank, 'worst'
            )
            print(f"  Saved: {filename}")
        
        print("\nAll force plots generated!\n")
    
    def analyze(self, n_samples=1000, n_top=3):
        """Main analysis pipeline"""
        # Load data
        self.load_data()
        
        # Train model
        self.train_model()
        
        # Compute explainer
        self.compute_explainer()
        
        # Generate synthetic configs
        all_configs, all_algos = self.generate_synthetic_configs(n_samples)
        
        # Predict and explain
        predictions, shap_values, shap_sums = self.predict_and_explain(all_configs)
        
        # Find best and worst
        best_idx, worst_idx = self.find_best_worst_configs(shap_sums, n_top)
        
        # Analyze best configs
        print(f"\n{'='*100}\nGLOBAL TOP {n_top} BEST CONFIGURATIONS\n{'='*100}\n")
        for rank, idx in enumerate(best_idx, 1):
            self.analyze_config(
                rank, idx, all_configs, predictions, shap_values, 
                all_algos[idx], 'Best'
            )
        
        # Analyze worst configs
        print(f"\n{'='*100}\nGLOBAL TOP {n_top} WORST CONFIGURATIONS\n{'='*100}\n")
        for rank, idx in enumerate(worst_idx, 1):
            self.analyze_config(
                rank, idx, all_configs, predictions, shap_values, 
                all_algos[idx], 'Worst'
            )
        
        # Generate force plots
        self.generate_all_force_plots(
            best_idx, worst_idx, all_configs, shap_values, predictions, all_algos
        )
        
        return self.model, self.explainer
