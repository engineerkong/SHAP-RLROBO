import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from scipy.interpolate import UnivariateSpline
import random
import os
import warnings


class SHAPTransferExplainer:
    """
    SHAP-based explainer for cross-task and cross-environment analysis on RL generalizability.
    Supports unified parameter grid with algorithm encoding (0=PPO, 1=A2C, 2=DDPG, 3=SAC).
    Supports multi-task analysis.
    """
    def __init__(self, param_grid, results, log_dir, seed=None, algorithm_names=None, task_name=None):
        self.param_grid = param_grid
        self.log_dir = log_dir
        self.seed = seed
        self.task_name = task_name
        self.meta_model = None
        self.explainer = None
        self.shap_values = None
        self.X = None
        
        # Handle multi-task mode
        if isinstance(results, dict):
            self.results_dict = results
            self.results = None
            self.multi_task = True
        else:
            self.results = results
            self.results_dict = None
            self.multi_task = False
        
        # Algorithm mapping
        self.algorithm_names = algorithm_names or {0: 'PPO', 1: 'A2C', 2: 'DDPG', 3: 'SAC'}
        
        # Set seed
        if self.seed is None:
            self.seed = random.randint(0, 100000)
        np.random.seed(self.seed)
        print(f"Using seed: {self.seed}")

    def set_task(self, task_name):
        """Set current task for analysis (multi-task mode only)."""
        if not self.multi_task:
            raise ValueError("Not in multi-task mode")
        if task_name not in self.results_dict:
            raise ValueError(f"Task '{task_name}' not found")
        self.task_name = task_name
        self.results = self.results_dict[task_name]
        print(f"Set task: {task_name}")

    def prepare_features(self, algorithm_code=None):
        """Prepare feature matrix, handling missing values."""
        if self.results is None:
            raise ValueError("No results loaded")
        
        data = self.results[self.results['algorithm'] == algorithm_code].copy() if algorithm_code is not None else self.results.copy()
        available_features = [f for f in self.param_grid.keys() if f in data.columns]
        X = data[available_features].copy()
        
        # Fill NaN values
        for col in X.columns:
            if X[col].isna().any():
                X[col] = X[col].fillna(0 if col in ['n_steps', 'buffer_size', 'algorithm'] else X[col].median())
        return X

    def train_meta_model(self, target, algorithm_code=None):
        """Train meta-model to predict performance from hyperparameters."""
        if self.results is None:
            raise ValueError("No results loaded")
        
        X = self.prepare_features(algorithm_code)
        y = self.results[self.results['algorithm'] == algorithm_code][target] if algorithm_code is not None else self.results[target]
        y = y.loc[X.index]
        
        self.meta_model = RandomForestRegressor(n_estimators=100, random_state=self.seed, max_depth=10, min_samples_split=5)
        self.meta_model.fit(X, y)
        
        r2 = self.meta_model.score(X, y)
        algo_label = f" ({self.algorithm_names.get(algorithm_code, f'Alg{algorithm_code}')})" if algorithm_code is not None else " (all)"
        task_label = f" ({self.task_name})" if self.task_name else ""
        print(f"Meta-model R² for {target}{algo_label}{task_label}: {r2:.4f}")
        return self.meta_model

    def explain(self, target, algorithm_code=None, force_retrain=True):
        """Use SHAP to explain hyperparameter impact."""
        if self.meta_model is None or force_retrain:
            self.train_meta_model(target, algorithm_code)
        
        X = self.prepare_features(algorithm_code)
        self.explainer = shap.TreeExplainer(self.meta_model)
        shap_values = self.explainer(X)
        self.shap_values = shap_values
        self.X = X
        return shap_values, X
    
    def plot_dependence(self, feature, target, interaction_feature='auto', algorithm_code=None, 
                       save_plot=True, show_plot=False, figsize=(10, 6), alpha=0.6, dot_size=50):
        """Create SHAP dependence plot."""
        if self.shap_values is None or self.X is None:
            raise ValueError("Run explain() first")
        if feature not in self.X.columns:
            raise ValueError(f"Feature '{feature}' not found")
        
        fig, ax = plt.subplots(figsize=figsize)
        shap.dependence_plot(feature, self.shap_values.values, self.X, interaction_index=interaction_feature,
                           ax=ax, show=False, alpha=alpha, dot_size=dot_size)
        
        # Customize plot
        task_label = f"{self.task_name}, " if self.task_name else ""
        algo_label = f"{self.algorithm_names.get(algorithm_code, f'Alg{algorithm_code}')}" if algorithm_code is not None else "All Algorithms"
        ax.set_xlabel(feature.replace('_', ' ').title(), fontsize=12, fontweight='bold')
        ax.set_ylabel(f'SHAP value for {feature}', fontsize=12, fontweight='bold')
        ax.set_title(f'SHAP Dependence: {feature} ({task_label}{algo_label}, Target: {target})', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        plt.tight_layout()
        
        if save_plot:
            os.makedirs(self.log_dir, exist_ok=True)
            parts = [target]
            if self.task_name:
                parts.append(self.task_name.replace(' ', '_'))
            parts.append(self.algorithm_names.get(algorithm_code, 'all') if algorithm_code is not None else 'all')
            parts.extend([feature, 'dependence.svg'])
            filepath = os.path.join(self.log_dir, '_'.join(parts))
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Saved: {filepath}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
        return fig
    
    def analyze_all_tasks(self, target, feature='learning_rate', algorithm_code=None, 
                         interaction_feature='auto', save_plot=True, show_plot=False, figsize=(10, 6)):
        """Analyze and plot for all tasks (multi-task mode only)."""
        if not self.multi_task:
            raise ValueError("Multi-task mode only")
        
        results = {}
        for task_name in self.results_dict.keys():
            print(f"\n{'='*60}\nAnalyzing: {task_name}\n{'='*60}")
            self.set_task(task_name)
            self.explain(target, algorithm_code=algorithm_code, force_retrain=True)
            fig = self.plot_dependence(feature, target, interaction_feature, algorithm_code, save_plot, show_plot, figsize)
            results[task_name] = fig
        
        print(f"\n{'='*60}\nCompleted {len(results)} tasks\n{'='*60}")
        return results
    
    def plot_combined_tasks_dependence(self, target, feature='learning_rate', algorithm_code=None,
                                      save_plot=True, show_plot=False, figsize=(12, 8), 
                                      alpha=0.6, dot_size=50, show_trends=False,
                                      trend_method='polynomial', trend_degree=3,
                                      trend_alpha=0.8, trend_linewidth=3):
        """Create combined SHAP dependence plot for all tasks (multi-task mode only)."""
        if not self.multi_task:
            raise ValueError("Multi-task mode only")
        
        # Define colors and markers
        base_colors = {'InvertedPendulum': '#1f77b4', 'HalfCheetah': '#ff7f0e', 
                      'Hopper': '#2ca02c', 'Walker2d': '#d62728'}
        marker_styles = {'MJ-PB': 'o', 'PB-MJ': 's'}
        line_styles = {'MJ-PB': '-', 'PB-MJ': '--'}
        
        # Collect data
        task_data = {}
        print(f"\n{'='*60}\nCombining data for: {feature}\n{'='*60}")
        
        for task_name in self.results_dict.keys():
            print(f"Processing: {task_name}")
            self.set_task(task_name)
            shap_values, X = self.explain(target, algorithm_code=algorithm_code, force_retrain=True)
            
            if feature not in X.columns:
                print(f"Warning: '{feature}' not in {task_name}, skipping")
                continue
            
            env_name = task_name.split('(')[0]
            transfer_type = task_name.split('(')[1].rstrip(')')
            
            task_data[task_name] = {
                'feature': X[feature].values,
                'shap': shap_values.values[:, X.columns.get_loc(feature)],
                'env_name': env_name,
                'transfer_type': transfer_type,
                'color': base_colors.get(env_name, '#000000'),
                'marker': marker_styles.get(transfer_type, 'o'),
                'linestyle': line_styles.get(transfer_type, '-')
            }
        
        # Plot
        fig, ax = plt.subplots(figsize=figsize)
        
        for task_name, data in task_data.items():
            transfer_label = "solid" if data['transfer_type'] == 'MJ-PB' else "dotted"
            ax.scatter(data['feature'], data['shap'], c=data['color'], marker=data['marker'],
                      alpha=alpha, s=dot_size, edgecolors='black', linewidth=0.5,
                      label=f"{task_name} ({transfer_label})", zorder=2)
        
        # Add trends
        if show_trends:
            print(f"\nFitting trends: {trend_method}")
            for task_name, data in task_data.items():
                try:
                    x_trend, y_trend = self._fit_trend(np.sort(data['feature']), 
                                                       data['shap'][np.argsort(data['feature'])],
                                                       method=trend_method, degree=trend_degree)
                    ax.plot(x_trend, y_trend, color=data['color'], linestyle=data['linestyle'],
                           alpha=trend_alpha, linewidth=trend_linewidth, zorder=3)
                except Exception as e:
                    print(f"Warning: Trend fit failed for {task_name}: {e}")
        
        # Customize
        algo_label = f" ({self.algorithm_names.get(algorithm_code, f'Alg{algorithm_code}')})" if algorithm_code is not None else ""
        trend_label = f" with {trend_method.upper()} trends" if show_trends else ""
        ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=9, framealpha=0.9)
        ax.set_xlabel(feature.replace('_', ' ').title(), fontsize=12, fontweight='bold')
        ax.set_ylabel(f'SHAP value for {feature}', fontsize=12, fontweight='bold')
        ax.set_title(f'SHAP Dependence: {feature} across Tasks{algo_label}{trend_label}\nTarget: {target}', 
                    fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        plt.tight_layout()
        plt.subplots_adjust(right=0.85)
        
        if save_plot:
            os.makedirs(self.log_dir, exist_ok=True)
            parts = [target, 'combined_tasks']
            parts.append(self.algorithm_names.get(algorithm_code, 'all') if algorithm_code is not None else 'all')
            parts.append(feature)
            if show_trends:
                parts.append(f'{trend_method}_trends')
            parts.append('dependence.svg')
            filepath = os.path.join(self.log_dir, '_'.join(parts))
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"\nSaved: {filepath}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
        
        print(f"\n{'='*60}\nCombined plot: {len(task_data)} tasks, {sum(len(d['feature']) for d in task_data.values())} points\n{'='*60}")
        return fig
    
    def plot_two_features_combined(self, target, features=['learning_rate', 'gamma'], algorithm_code=None,
                                   save_plot=True, show_plot=False, figsize=(20, 7), alpha=0.3, dot_size=50,
                                   show_trends=True, trend_method='polynomial', trend_degree=3,
                                   trend_alpha=0.9, trend_linewidth=3):
        """Create 1x2 subplot for two features (multi-task mode only)."""
        if not self.multi_task or len(features) != 2:
            raise ValueError("Multi-task mode only, exactly 2 features required")
        
        # Define colors and markers
        base_colors = {'InvertedPendulum': '#1f77b4', 'HalfCheetah': '#ff7f0e',
                      'Hopper': '#2ca02c', 'Walker2d': '#d62728'}
        marker_styles = {'MJ-PB': 'o', 'PB-MJ': 's'}
        line_styles = {'MJ-PB': '-', 'PB-MJ': '--'}
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        for feat_idx, feature in enumerate(features):
            ax = axes[feat_idx]
            task_data = {}
            print(f"\n{'='*60}\nProcessing: {feature}\n{'='*60}")
            
            # Collect data
            for task_name in self.results_dict.keys():
                print(f"Processing: {task_name}")
                self.set_task(task_name)
                shap_values, X = self.explain(target, algorithm_code=algorithm_code, force_retrain=True)
                
                if feature not in X.columns:
                    print(f"Warning: '{feature}' not in {task_name}, skipping")
                    continue
                
                env_name = task_name.split('(')[0]
                transfer_type = task_name.split('(')[1].rstrip(')')
                
                task_data[task_name] = {
                    'feature': X[feature].values,
                    'shap': shap_values.values[:, X.columns.get_loc(feature)],
                    'env_name': env_name,
                    'transfer_type': transfer_type,
                    'color': base_colors.get(env_name, '#000000'),
                    'marker': marker_styles.get(transfer_type, 'o'),
                    'linestyle': line_styles.get(transfer_type, '-')
                }
            
            # Plot
            for task_name, data in task_data.items():
                ax.scatter(data['feature'], data['shap'], c=data['color'], marker=data['marker'],
                          alpha=alpha, s=dot_size, edgecolors='black', linewidth=0.5,
                          label=task_name if feat_idx == 0 else "", zorder=2)
            
            # Add trends
            if show_trends:
                print(f"Fitting trends for {feature}")
                for task_name, data in task_data.items():
                    try:
                        x_trend, y_trend = self._fit_trend(np.sort(data['feature']),
                                                           data['shap'][np.argsort(data['feature'])],
                                                           method=trend_method, degree=trend_degree)
                        ax.plot(x_trend, y_trend, color=data['color'], linestyle=data['linestyle'],
                               alpha=trend_alpha, linewidth=trend_linewidth, zorder=3)
                    except Exception as e:
                        print(f"Warning: Trend fit failed for {task_name}: {e}")
            
            # Customize subplot
            ax.set_xlabel(feature.replace('_', ' ').title(), fontsize=12, fontweight='bold')
            ax.set_ylabel(f'SHAP value for {feature}', fontsize=12, fontweight='bold')
            ax.tick_params(axis='both', labelsize=12)
            ax.set_title(f'{feature.replace("_", " ").title()}', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, linestyle='--')
        
        # Add legend
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch
        
        legend_elements = []
        for env_name, color in base_colors.items():
            legend_elements.append(Patch(facecolor=color, edgecolor='black', label=env_name))
        legend_elements.append(Patch(facecolor='none', edgecolor='none', label=''))
        legend_elements.append(Line2D([0], [0], color='gray', linewidth=2, linestyle='-',
                                     marker='o', markersize=10, label='MJ-PB (Solid line)'))
        legend_elements.append(Line2D([0], [0], color='gray', linewidth=2, linestyle='--',
                                     marker='s', markersize=10, label='PB-MJ (Dotted line)'))
        
        fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.05),
                  ncol=8, fontsize=12, framealpha=0.9)
        plt.tight_layout(rect=[0, 0.05, 1, 0.96])
        
        if save_plot:
            os.makedirs(self.log_dir, exist_ok=True)
            parts = [target, 'combined_tasks']
            parts.append(self.algorithm_names.get(algorithm_code, 'all') if algorithm_code is not None else 'all')
            parts.append('_'.join(features))
            if show_trends:
                parts.append(f'{trend_method}_trends')
            parts.append('2panel.svg')
            filepath = os.path.join(self.log_dir, '_'.join(parts))
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"\nSaved: {filepath}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
        
        print(f"\n{'='*60}\n2-panel plot: {features}\n{'='*60}")
        return fig
    
    def _fit_trend(self, x, y, method='polynomial', degree=3, n_points=100):
        """Fit trend line/curve to data."""
        mask = ~(np.isnan(x) | np.isnan(y))
        x, y = x[mask], y[mask]
        
        if len(x) < 2:
            raise ValueError("Insufficient data points")
        
        x_trend = np.linspace(x.min(), x.max(), n_points)
        
        if method == 'polynomial':
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                y_trend = np.polyval(np.polyfit(x, y, degree), x_trend)
        elif method == 'svr':
            scaler_x, scaler_y = StandardScaler(), StandardScaler()
            x_scaled = scaler_x.fit_transform(x.reshape(-1, 1))
            y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()
            svr = SVR(kernel='rbf', C=100, gamma='auto', epsilon=0.1)
            svr.fit(x_scaled, y_scaled)
            y_trend = scaler_y.inverse_transform(svr.predict(scaler_x.transform(x_trend.reshape(-1, 1))).reshape(-1, 1)).ravel()
        elif method == 'spline':
            if len(x) > degree:
                y_trend = UnivariateSpline(x, y, k=min(degree, len(x)-1), s=len(x))(x_trend)
            else:
                y_trend = np.polyval(np.polyfit(x, y, min(degree, len(x)-1)), x_trend)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return x_trend, y_trend
