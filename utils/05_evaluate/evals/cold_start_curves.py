#!/usr/bin/env python3

"""
Cold Start Curves Evaluation Module

This module analyzes how model performance varies with the amount of user
history available for featurization (the "cold start" problem).

Users are binned by the number of likes used to create their user embeddings,
and performance metrics are computed for each bin to understand:
- How much history is needed for reliable predictions?
- Do users with more history receive better predictions?
- Where does performance plateau?

Outputs:
- cold_start_summary.json: Summary statistics and bin-level metrics
- precision_vs_likes.png: Precision curve as function of embedding likes
- recall_vs_likes.png: Recall curve as function of embedding likes
- auc_vs_likes.png: AUC-ROC curve as function of embedding likes
- combined_cold_start.png: All metrics on one plot
- binned_metrics.csv: Full per-bin metrics table
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import (
    EvalContext,
    EvalModule,
    compute_per_user_metrics,
)


class ColdStartCurvesModule(EvalModule):
    """
    Evaluation module for analyzing cold start behavior.
    
    Bins users by the number of embedding likes and computes performance
    metrics for each bin to understand how model quality varies with
    available user history.
    """
    
    name = "cold_start_curves"
    description = "Analyzes model performance as a function of user history length (embedding likes count)"
    
    # Default bin edges for number of embedding likes
    # Users are grouped into bins: [1-2], [3-5], [6-10], [11-20], [21-50], [51+]
    DEFAULT_BIN_EDGES = [0, 2, 5, 10, 20, 50, 100, 500, float('inf')]
    
    # Metrics to plot
    METRICS = ['precision', 'recall', 'auc_roc', 'accuracy', 'f1']
    
    # Plot styling
    FIGURE_SIZE = (10, 6)
    DPI = 150
    
    def run(self, ctx: EvalContext) -> Dict[str, Any]:
        """
        Run cold start analysis.
        
        Args:
            ctx: EvalContext with predictions, user metadata, and output directory.
        
        Returns:
            Dict with binned metrics and artifact paths.
        """
        out_dir = self.get_output_dir(ctx)
        
        # Get bin edges from config or use defaults
        bin_edges = ctx.config.get('cold_start_bin_edges', self.DEFAULT_BIN_EDGES)
        
        # Step 1: Compute per-user metrics
        print(f"    Computing per-user metrics for {ctx.num_holdout_users} users...")
        per_user_df = compute_per_user_metrics(ctx.predictions_df)
        
        # Step 2: Merge with user metadata (num_embedding_likes)
        if 'num_embedding_likes' not in ctx.user_metadata_df.columns:
            print("    Warning: num_embedding_likes not in user_metadata_df, skipping cold start analysis")
            return {
                'status': 'skipped',
                'reason': 'num_embedding_likes not available in user metadata',
            }
        
        merged_df = per_user_df.merge(
            ctx.user_metadata_df[['did', 'num_embedding_likes', 'num_total_likes']],
            on='did',
            how='left',
        )
        
        # Drop users without metadata
        merged_df = merged_df.dropna(subset=['num_embedding_likes'])
        if len(merged_df) == 0:
            print("    Warning: No users with embedding likes metadata")
            return {
                'status': 'skipped',
                'reason': 'No users with embedding likes metadata',
            }
        
        print(f"    Analyzing {len(merged_df)} users with embedding likes metadata...")
        
        # Step 3: Bin users by num_embedding_likes
        merged_df['likes_bin'] = pd.cut(
            merged_df['num_embedding_likes'],
            bins=bin_edges,
            labels=self._make_bin_labels(bin_edges),
            include_lowest=True,
        )
        
        # Step 4: Compute metrics per bin
        print("    Computing metrics per bin...")
        binned_metrics = self._compute_binned_metrics(merged_df, bin_edges)
        
        # Save binned metrics
        binned_path = out_dir / "binned_metrics.csv"
        binned_metrics.to_csv(binned_path, index=False)
        
        # Step 5: Generate plots
        print("    Generating cold start curve plots...")
        plot_paths = {}
        
        # Individual metric plots
        for metric in self.METRICS:
            if metric in binned_metrics.columns:
                plot_path = out_dir / f"{metric}_vs_likes.png"
                self._plot_metric_vs_likes(
                    binned_metrics=binned_metrics,
                    metric=metric,
                    save_path=plot_path,
                )
                plot_paths[f"{metric}_plot_path"] = str(plot_path)
        
        # Combined plot
        combined_path = out_dir / "combined_cold_start.png"
        self._plot_combined(binned_metrics, combined_path)
        plot_paths['combined_plot_path'] = str(combined_path)
        
        # User distribution by bin
        dist_path = out_dir / "user_distribution_by_bin.png"
        self._plot_user_distribution(merged_df, dist_path)
        plot_paths['distribution_plot_path'] = str(dist_path)
        
        # Scatter plot of metric vs likes (raw, not binned)
        scatter_path = out_dir / "scatter_performance_vs_likes.png"
        self._plot_scatter(merged_df, scatter_path)
        plot_paths['scatter_plot_path'] = str(scatter_path)
        
        # Step 6: Compute summary statistics
        summary = self._compute_summary(merged_df, binned_metrics)
        summary.update(plot_paths)
        summary['binned_metrics_path'] = str(binned_path)
        summary['bin_edges'] = [float(e) if e != float('inf') else 'inf' for e in bin_edges]
        
        # Save summary
        summary_path = out_dir / "cold_start_summary.json"
        self.save_json(summary, summary_path)
        
        return summary
    
    def _make_bin_labels(self, bin_edges: List[float]) -> List[str]:
        """Create human-readable bin labels."""
        labels = []
        for i in range(len(bin_edges) - 1):
            low = int(bin_edges[i]) + 1 if i > 0 else int(bin_edges[i])
            high = bin_edges[i + 1]
            if high == float('inf'):
                labels.append(f"{low}+")
            else:
                labels.append(f"{low}-{int(high)}")
        return labels
    
    def _compute_binned_metrics(
        self,
        merged_df: pd.DataFrame,
        bin_edges: List[float],
    ) -> pd.DataFrame:
        """Compute aggregate metrics for each bin."""
        rows = []
        
        for bin_label in merged_df['likes_bin'].cat.categories:
            bin_data = merged_df[merged_df['likes_bin'] == bin_label]
            
            if len(bin_data) == 0:
                continue
            
            row = {
                'bin': str(bin_label),
                'n_users': len(bin_data),
                'mean_embedding_likes': float(bin_data['num_embedding_likes'].mean()),
                'median_embedding_likes': float(bin_data['num_embedding_likes'].median()),
            }
            
            # Compute mean and std for each metric
            for metric in self.METRICS:
                if metric in bin_data.columns:
                    values = bin_data[metric].dropna()
                    if len(values) > 0:
                        row[metric] = float(values.mean())
                        row[f'{metric}_std'] = float(values.std())
                        row[f'{metric}_median'] = float(values.median())
                        row[f'{metric}_n'] = len(values)
                    else:
                        row[metric] = float('nan')
                        row[f'{metric}_std'] = float('nan')
                        row[f'{metric}_median'] = float('nan')
                        row[f'{metric}_n'] = 0
            
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def _plot_metric_vs_likes(
        self,
        binned_metrics: pd.DataFrame,
        metric: str,
        save_path: Path,
    ) -> None:
        """Plot a single metric vs. embedding likes bins."""
        fig, ax = plt.subplots(figsize=self.FIGURE_SIZE)
        
        # Filter to bins with data
        plot_df = binned_metrics[binned_metrics[f'{metric}_n'] > 0].copy()
        if len(plot_df) == 0:
            plt.close(fig)
            return
        
        x = range(len(plot_df))
        y = plot_df[metric].values
        yerr = plot_df[f'{metric}_std'].values
        
        # Plot with error bars
        ax.errorbar(x, y, yerr=yerr, fmt='o-', linewidth=2, markersize=8,
                   capsize=5, capthick=2, color='steelblue')
        
        # Add user counts as secondary info
        for i, (xi, yi, n) in enumerate(zip(x, y, plot_df['n_users'].values)):
            ax.annotate(f'n={n}', (xi, yi), textcoords="offset points",
                       xytext=(0, 10), ha='center', fontsize=9, alpha=0.7)
        
        ax.set_xticks(x)
        ax.set_xticklabels(plot_df['bin'].values, rotation=45, ha='right')
        ax.set_xlabel('Number of Embedding Likes (User History)', fontsize=12)
        ax.set_ylabel(f'{metric.replace("_", " ").title()}', fontsize=12)
        ax.set_title(f'Cold Start Analysis: {metric.replace("_", " ").title()} vs. User History Length',
                    fontsize=13)
        ax.grid(True, alpha=0.3)
        
        # Set y-axis limits based on metric type
        if metric in ['precision', 'recall', 'accuracy', 'f1', 'auc_roc']:
            ax.set_ylim(0, 1.05)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=self.DPI, bbox_inches='tight')
        plt.close(fig)
    
    def _plot_combined(
        self,
        binned_metrics: pd.DataFrame,
        save_path: Path,
    ) -> None:
        """Plot all metrics on a single combined plot."""
        fig, ax = plt.subplots(figsize=(12, 7))
        
        colors = {
            'precision': '#1f77b4',
            'recall': '#ff7f0e',
            'auc_roc': '#2ca02c',
            'accuracy': '#d62728',
            'f1': '#9467bd',
        }
        
        markers = {
            'precision': 'o',
            'recall': 's',
            'auc_roc': '^',
            'accuracy': 'D',
            'f1': 'v',
        }
        
        x = range(len(binned_metrics))
        
        for metric in self.METRICS:
            if metric not in binned_metrics.columns:
                continue
            if binned_metrics[f'{metric}_n'].sum() == 0:
                continue
            
            y = binned_metrics[metric].values
            yerr = binned_metrics[f'{metric}_std'].values
            
            # Replace NaN with None for plotting
            mask = ~np.isnan(y)
            if mask.sum() == 0:
                continue
            
            color = colors.get(metric, '#333333')
            marker = markers.get(metric, 'o')
            
            ax.errorbar(
                np.array(x)[mask], y[mask], yerr=yerr[mask],
                fmt=f'{marker}-', linewidth=2, markersize=8,
                capsize=4, capthick=1.5, color=color,
                label=metric.replace("_", " ").title(),
            )
        
        ax.set_xticks(x)
        ax.set_xticklabels(binned_metrics['bin'].values, rotation=45, ha='right')
        ax.set_xlabel('Number of Embedding Likes (User History)', fontsize=12)
        ax.set_ylabel('Metric Value', fontsize=12)
        ax.set_title('Cold Start Analysis: All Metrics vs. User History Length', fontsize=14)
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=self.DPI, bbox_inches='tight')
        plt.close(fig)
    
    def _plot_user_distribution(
        self,
        merged_df: pd.DataFrame,
        save_path: Path,
    ) -> None:
        """Plot distribution of users across bins."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Bar chart of users per bin
        bin_counts = merged_df['likes_bin'].value_counts().sort_index()
        ax1.bar(range(len(bin_counts)), bin_counts.values, color='steelblue', edgecolor='black')
        ax1.set_xticks(range(len(bin_counts)))
        ax1.set_xticklabels(bin_counts.index, rotation=45, ha='right')
        ax1.set_xlabel('Embedding Likes Bin', fontsize=11)
        ax1.set_ylabel('Number of Users', fontsize=11)
        ax1.set_title('User Distribution Across History Length Bins', fontsize=12)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Add count labels on bars
        for i, v in enumerate(bin_counts.values):
            ax1.text(i, v + 0.5, str(v), ha='center', fontsize=9)
        
        # Histogram of raw embedding likes
        ax2.hist(merged_df['num_embedding_likes'], bins=50, color='steelblue',
                edgecolor='black', alpha=0.7)
        ax2.set_xlabel('Number of Embedding Likes', fontsize=11)
        ax2.set_ylabel('Number of Users', fontsize=11)
        ax2.set_title('Distribution of User History Lengths', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        # Add median line
        median_likes = merged_df['num_embedding_likes'].median()
        ax2.axvline(median_likes, color='red', linestyle='--', linewidth=2,
                   label=f'Median: {median_likes:.0f}')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=self.DPI, bbox_inches='tight')
        plt.close(fig)
    
    def _plot_scatter(
        self,
        merged_df: pd.DataFrame,
        save_path: Path,
    ) -> None:
        """Plot scatter of performance vs embedding likes (raw, not binned)."""
        # Select a metric to show (prefer AUC, fallback to accuracy)
        metric = 'auc_roc' if 'auc_roc' in merged_df.columns else 'accuracy'
        
        fig, ax = plt.subplots(figsize=self.FIGURE_SIZE)
        
        valid_df = merged_df.dropna(subset=[metric, 'num_embedding_likes'])
        if len(valid_df) == 0:
            plt.close(fig)
            return
        
        # Scatter plot with some transparency
        ax.scatter(
            valid_df['num_embedding_likes'],
            valid_df[metric],
            alpha=0.3,
            s=20,
            color='steelblue',
        )
        
        # Add trend line (rolling mean)
        sorted_df = valid_df.sort_values('num_embedding_likes')
        window = max(len(sorted_df) // 20, 5)  # 5% window or at least 5 users
        if len(sorted_df) > window:
            rolling_mean = sorted_df[metric].rolling(window=window, center=True).mean()
            ax.plot(sorted_df['num_embedding_likes'], rolling_mean,
                   color='red', linewidth=2, label=f'Rolling Mean (window={window})')
            ax.legend()
        
        ax.set_xlabel('Number of Embedding Likes', fontsize=12)
        ax.set_ylabel(f'{metric.replace("_", " ").title()}', fontsize=12)
        ax.set_title(f'{metric.replace("_", " ").title()} vs. User History Length (Raw)', fontsize=13)
        ax.grid(True, alpha=0.3)
        
        # Log scale for x-axis if range is large
        if valid_df['num_embedding_likes'].max() > 100:
            ax.set_xscale('log')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=self.DPI, bbox_inches='tight')
        plt.close(fig)
    
    def _compute_summary(
        self,
        merged_df: pd.DataFrame,
        binned_metrics: pd.DataFrame,
    ) -> Dict[str, Any]:
        """Compute summary statistics for the cold start analysis."""
        summary = {
            'total_users_analyzed': len(merged_df),
            'num_bins': len(binned_metrics),
            'embedding_likes_stats': {
                'mean': float(merged_df['num_embedding_likes'].mean()),
                'median': float(merged_df['num_embedding_likes'].median()),
                'std': float(merged_df['num_embedding_likes'].std()),
                'min': int(merged_df['num_embedding_likes'].min()),
                'max': int(merged_df['num_embedding_likes'].max()),
            },
        }
        
        # For each metric, find the "cold start threshold" - bin where performance stabilizes
        for metric in self.METRICS:
            if metric not in binned_metrics.columns:
                continue
            
            values = binned_metrics[metric].dropna().values
            if len(values) < 2:
                continue
            
            # Find the bin where performance is >= 90% of max
            max_val = values.max()
            threshold_idx = np.argmax(values >= 0.9 * max_val)
            
            summary[f'{metric}_cold_start_threshold_bin'] = str(binned_metrics['bin'].iloc[threshold_idx])
            summary[f'{metric}_max_bin'] = str(binned_metrics['bin'].iloc[values.argmax()])
            summary[f'{metric}_improvement_first_to_last'] = float(values[-1] - values[0]) if len(values) > 1 else 0.0
        
        return summary


# Export the module class
__all__ = ['ColdStartCurvesModule']
