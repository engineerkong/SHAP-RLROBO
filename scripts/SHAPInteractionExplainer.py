import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from itertools import combinations
import shap
import os

class SHAPInteractionExplainer:
    """Explain hyperparameter interaction patterns on RL performance using SHAP values."""

    def __init__(self, param_grid, results, log_dir, seed=None):
        self.param_grid = param_grid
        self.results = results
        self.log_dir = log_dir
        self.seed = seed if seed is not None else np.random.randint(0, 100000)
        self.meta_model = None
        self.explainer = None
        np.random.seed(self.seed)
        print(f"Using seed: {self.seed}")

    def train_meta_model(self, target):
        X = self.results[list(self.param_grid.keys())]
        y = self.results[target]
        self.meta_model = RandomForestRegressor(n_estimators=100)
        self.meta_model.fit(X, y)
        print(f"Meta-model R² score for {target}: {self.meta_model.score(X, y):.4f}")
        return self.meta_model

    def explain(self, target, force_retrain=True):
        if self.meta_model is None or force_retrain:
            self.train_meta_model(target)
        X = self.results[list(self.param_grid.keys())]
        self.explainer = shap.TreeExplainer(self.meta_model)
        return self.explainer(X), X

    def analyze_interaction_metrics(self, algorithm, target):
        features = list(self.param_grid.keys())
        results = []
        force_retrain = True
        
        for feat1, feat2 in combinations(features, 2):
            df = self.results.copy()
            interaction_term = df[feat1] * df[feat2]
            prod_corr = interaction_term.corr(df[target])
            
            X = pd.DataFrame({
                feat1: df[feat1],
                feat2: df[feat2],
                'interaction': interaction_term
            })
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            model = LinearRegression()
            model.fit(X_scaled, df[target])
            reg_coef = model.coef_[2]
            
            try:
                shap_values, X_shap = self.explain(target, force_retrain)
                shap_interaction_values = shap.TreeExplainer(self.meta_model).shap_interaction_values(X_shap)
                idx1, idx2 = features.index(feat1), features.index(feat2)
                shap_interact = shap_interaction_values[:, idx1, idx2].mean()
            except:
                shap_interact = np.nan
            
            X_no_interact = X[[feat1, feat2]]
            model_no_interact = LinearRegression()
            model_no_interact.fit(scaler.fit_transform(X_no_interact), df[target])
            r2_no_interact = model_no_interact.score(scaler.fit_transform(X_no_interact), df[target])
            r2_with_interact = model.score(X_scaled, df[target])
            
            results.append({
                'feature1': feat1,
                'feature2': feat2,
                'product_correlation': prod_corr,
                'regression_coefficient': reg_coef,
                'shap_interaction': shap_interact,
                'variance_explained': r2_with_interact - r2_no_interact
            })
            force_retrain = False
        
        results_df = pd.DataFrame(results).sort_values('product_correlation', key=abs, ascending=False)
        results_df.to_csv(os.path.join(self.log_dir, f'interaction_metrics_{algorithm}.csv'), index=False)
        return results_df

    def plot_interaction_metrics_heatmap(self, algorithm, target, metric='shap_interaction'):
        metrics_df = self.analyze_interaction_metrics(algorithm, target)
        features = list(self.param_grid.keys())
        n_features = len(features)
        matrix = np.zeros((n_features, n_features))
        
        for _, row in metrics_df.iterrows():
            i, j = features.index(row['feature1']), features.index(row['feature2'])
            matrix[i, j] = matrix[j, i] = row[metric]
        
        mask = np.triu(np.ones_like(matrix, dtype=bool))
        plt.figure(figsize=(10, 8))
        sns.heatmap(matrix, mask=mask, annot=True, fmt='.4f', cmap='RdBu_r', center=0,
                    xticklabels=features, yticklabels=features, square=True,
                    cbar_kws={'label': metric.replace('_', ' ').title()})
        plt.title(f'{metric.replace("_", " ").title()}\n{algorithm} on {target}', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, f'interaction_metric_{metric}_{algorithm}.png'), dpi=150)
        plt.close()

    def plot_interaction_metrics_comparison(self, algorithms_dict, param_grids, target, metric='shap_interaction'):
        algorithms = list(algorithms_dict.keys())
        fig, axes = plt.subplots(1, 4, figsize=(20, 5), gridspec_kw={'wspace': 0.1})
        axes = axes.flatten()
        
        all_matrices, all_features, all_values = {}, {}, []
        
        for algo in algorithms:
            self.results = pd.read_csv(algorithms_dict[algo])
            self.param_grid = param_grids[algo]
            metrics_df = self.analyze_interaction_metrics(algo, target)
            
            features = list(self.param_grid.keys())
            matrix = np.zeros((len(features), len(features)))
            for _, row in metrics_df.iterrows():
                i, j = features.index(row['feature1']), features.index(row['feature2'])
                matrix[i, j] = matrix[j, i] = row[metric]
            
            all_matrices[algo] = matrix
            all_features[algo] = features
            mask = np.triu(np.ones_like(matrix, dtype=bool))
            all_values.extend(matrix[~mask].flatten())
        
        vmax = max(abs(np.min(all_values)), abs(np.max(all_values)))
        vmin = -vmax
        
        for idx, algo in enumerate(algorithms):
            matrix, features = all_matrices[algo], all_features[algo]
            mask = np.triu(np.ones_like(matrix, dtype=bool))
            annot_size = 13 if len(features) <= 4 else 6
            fmt = '.4f' if len(features) <= 4 else '.2f'
            
            sns.heatmap(matrix, mask=mask, annot=True, fmt=fmt, annot_kws={'size': annot_size},
                       cmap='RdBu_r', center=0, vmin=vmin, vmax=vmax, xticklabels=features,
                       yticklabels=features, cbar=False, square=True, ax=axes[idx],
                       linewidths=0.5, linecolor='white')
            axes[idx].set_title(algo, fontsize=14, fontweight='bold', pad=10)
            axes[idx].tick_params(axis='both', labelsize=12)
        
        fig.subplots_adjust(right=0.95)
        cbar_ax = fig.add_axes([0.96, 0.25, 0.015, 0.5])
        sm = plt.cm.ScalarMappable(cmap='RdBu_r', norm=plt.Normalize(vmin=vmin, vmax=vmax))
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cbar_ax)
        cbar.set_label(metric.replace('_', ' ').title(), rotation=270, labelpad=20, fontsize=13)
        
        plt.savefig(os.path.join(self.log_dir, f'interaction_metric_{metric}_comparison_all.svg'),
                   dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved: {self.log_dir}/interaction_metric_{metric}_comparison_all.svg")

    def plot_dependence(self, algorithm, target, feature):
        """Plot SHAP dependence for specific feature."""
        shap_values, X = self.explain(target)
        plt.figure(figsize=(10, 6))
        shap.dependence_plot(feature, shap_values.values, X, show=False)
        plt.title(f'SHAP Dependence: {feature} on {target} ({algorithm})', fontsize=13)
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, f'shap_dependence_{feature}_{algorithm}.svg'))
        plt.close()

    def plot_feature_interaction_grid_binned(self, algorithm, target, feat1, feat2, n_bins=5):
        """Plot interaction heatmap with binned continuous features."""
        df = self.results.copy()
        df[f'{feat1}_binned'], f1_bins = pd.cut(df[feat1], bins=n_bins, retbins=True, duplicates='drop')
        df[f'{feat2}_binned'], f2_bins = pd.cut(df[feat2], bins=n_bins, retbins=True, duplicates='drop')
        
        f1_labels = [f'{f1_bins[i]:.3f}-{f1_bins[i+1]:.3f}' for i in range(len(f1_bins)-1)]
        f2_labels = [f'{f2_bins[i]:.3f}-{f2_bins[i+1]:.3f}' for i in range(len(f2_bins)-1)]
        
        grouped = df.groupby([f'{feat2}_binned', f'{feat1}_binned'])[target].mean()
        grid = np.full((n_bins, n_bins), np.nan)
        
        for (f2_bin, f1_bin), val in grouped.items():
            if pd.notna(f2_bin) and pd.notna(f1_bin):
                i = list(df[f'{feat2}_binned'].cat.categories).index(f2_bin)
                j = list(df[f'{feat1}_binned'].cat.categories).index(f1_bin)
                grid[i, j] = val
        
        grid = np.flipud(grid)
        f2_labels = f2_labels[::-1]
        
        plt.figure(figsize=(12, 10))
        ax = sns.heatmap(grid, mask=np.isnan(grid), annot=True, fmt='.1f', 
                         annot_kws={'size': 16}, cmap='RdYlGn_r', center=np.nanmean(grid),
                         xticklabels=f1_labels, yticklabels=f2_labels, 
                         cbar_kws={'label': target}, linewidths=1, linecolor='white')
        ax.collections[0].colorbar.set_label(target, fontsize=14, weight='bold')
        ax.collections[0].colorbar.ax.tick_params(labelsize=12)
        ax.tick_params(axis='both', labelsize=14)
        plt.xlabel(f'{feat1} (binned)', fontsize=14, fontweight='bold')
        plt.ylabel(f'{feat2} (binned)', fontsize=14, fontweight='bold')
        plt.title(f'Interaction: {feat1} × {feat2}\n{algorithm} on {target}', fontsize=14)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, f'feature_grid_{feat1}_{feat2}_{algorithm}.svg'), 
                    dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()

    def plot_all_interaction_grids(self, algorithm, target, n_top=6):
        """Plot interaction grids for top N feature pairs."""
        features = list(self.param_grid.keys())
        shap_values, X = self.explain(target)
        shap_interaction_values = shap.TreeExplainer(self.meta_model).shap_interaction_values(X)
        
        interactions = [(features[i], features[j], np.abs(shap_interaction_values[:, i, j]).mean())
                        for i in range(len(features)) for j in range(i+1, len(features))]
        interactions.sort(key=lambda x: x[2], reverse=True)
        
        print(f"\nTop interactions for {algorithm}:")
        for feat1, feat2, strength in interactions[:n_top]:
            print(f"  {feat1} × {feat2}: {strength:.2f}")
            self.plot_feature_interaction_grid_binned(algorithm, target, feat1, feat2)