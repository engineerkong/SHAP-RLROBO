import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import shap
from sklearn.ensemble import RandomForestRegressor

class SHAPImpactExplainer:
    """Explain algorithm and hyperparameter impact on RL performance using SHAP values."""
    
    def __init__(self, param_grid, results, log_dir, seed=None):
        self.param_grid = param_grid
        self.results = results
        self.log_dir = log_dir
        self.seed = seed or np.random.randint(0, 100000)
        self.meta_model = None
        self.explainer = None
        np.random.seed(self.seed)
        print(f"Using seed: {self.seed}")

    def train_meta_model(self, target):
        """Train Random Forest meta-model to predict performance."""
        X = self.results[list(self.param_grid.keys())]
        y = self.results[target]
        self.meta_model = RandomForestRegressor(n_estimators=100)
        self.meta_model.fit(X, y)
        print(f"Meta-model R² for {target}: {self.meta_model.score(X, y):.4f}")
        return self.meta_model

    def explain(self, target, force_retrain=True):
        """Compute SHAP values for target metric."""
        if self.meta_model is None or force_retrain:
            self.train_meta_model(target)
        X = self.results[list(self.param_grid.keys())]
        self.explainer = shap.TreeExplainer(self.meta_model)
        return self.explainer(X), X
    
    def plot_beeswarm(self, algorithm, target):
        """Plot SHAP beeswarm showing feature impacts."""
        shap_values, X = self.explain(target)
        plt.figure(figsize=(12, 8))
        shap.plots.beeswarm(shap_values, show=False, color_bar=True, color=plt.get_cmap("plasma"))
        plt.xlim(-500, 1500)
        plt.title(f'SHAP Beeswarm: {target} using {algorithm}', fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, f'shap_beeswarm_{algorithm}.svg'))
        plt.close()

    def plot_beeswarm_comparison(self, algorithms_dict, param_grids, target):
        """Plot 2x2 comparison of SHAP beeswarms with shared colorbar."""
        algorithms = list(algorithms_dict.keys())
        n_rows, n_cols = 2, 2
        
        fig = plt.figure(figsize=(22, 10))
        gs = gridspec.GridSpec(n_rows, n_cols + 1, figure=fig, width_ratios=[1, 1, 0.03], 
                               wspace=0.15, hspace=0.15)
        axes = [fig.add_subplot(gs[i, j]) for i in range(n_rows) for j in range(n_cols)]
        
        # Compute all SHAP values
        all_shap_data = {}
        print("\nComputing SHAP values...")
        for algo in algorithms:
            print(f"  {algo}...")
            self.results = pd.read_csv(algorithms_dict[algo])
            self.param_grid = param_grids[algo]
            self.meta_model = None
            all_shap_data[algo] = self.explain(target)
        
        xlim = (-500, 1250)
        
        # Plot each algorithm
        for idx, algo in enumerate(algorithms):
            ax = axes[idx]
            shap_values, X = all_shap_data[algo]
            plt.sca(ax)
            shap.plots.beeswarm(shap_values, show=False, color_bar=False, 
                                color=plt.get_cmap("plasma"), plot_size=None)
            
            n_features = len(list(self.param_grid.keys()))
            ax.set_xlim(xlim)
            ax.set_ylim(-0.4, n_features - 0.6)
            ax.margins(0)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.text(0.02, 0.98, algo, transform=ax.transAxes, fontsize=16, 
                    fontweight='bold', verticalalignment='top')
            ax.set_xlabel('SHAP value', fontsize=12)
            ax.tick_params(axis='both', labelsize=13)
        
        # Add colorbar
        cbar_ax = fig.add_subplot(gs[:, -1])
        from matplotlib.colors import Normalize
        import matplotlib.cm as cm
        sm = cm.ScalarMappable(cmap='plasma', norm=Normalize(vmin=0, vmax=1))
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cbar_ax)
        cbar.set_label('Normalized Feature Value', rotation=270, labelpad=25, fontsize=14)
        cbar.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        cbar.ax.tick_params(labelsize=13)
        
        plt.savefig(os.path.join(self.log_dir, 'shap_beeswarm_comparison_all.svg'), 
                    dpi=150, bbox_inches='tight', pad_inches=0.3)
        plt.close()
        print(f"\n✅ Saved to: {self.log_dir}/shap_beeswarm_comparison_all.svg")


