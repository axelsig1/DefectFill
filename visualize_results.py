"""
DefectFill Experiment Results Visualization Module

Generates the following visualization charts:
1. Heatmaps - Shows config vs class performance matrix
2. Scatter Plot - KID vs IC-LPIPS quality-diversity tradeoff analysis
3. Grouped Bar Charts - Comparison of each config across different classes

Usage:
    python visualize_results.py --csv_path evaluation_results.csv --output_dir ./figures
"""

import os
import argparse
import pandas as pd
import numpy as np

# Set non-interactive backend for headless environments (e.g., servers/clusters)
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import seaborn as sns

# Global Plotting Configuration
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.2)

# Visualization Color Scheme
CONFIG_COLORS = {
    'base': '#3498db',  # Blue
    'tex': '#e74c3c',   # Red
    'obj': '#2ecc71'    # Green
}

# Shapes to distinguish between Object and Texture datasets
CATEGORY_MARKERS = {
    'object': 'o',    # Circle
    'texture': 's'    # Square
}

# MVTec AD Dataset Groupings
OBJECT_CLASSES = ['bottle', 'cable', 'hazelnut', 'metal_nut', 'toothbrush']
TEXTURE_CLASSES = ['carpet', 'grid', 'leather', 'tile', 'wood']


def load_and_validate_data(csv_path):
    """Load and validate the evaluation CSV data."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    # Required metric columns
    required_columns = ['class', 'config', 'category_type', 'KID_mean', 'KID_std', 
                        'IC_LPIPS_mean', 'IC_LPIPS_std']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        raise ValueError(f"CSV missing required columns: {missing_columns}")
    
    print(f"Successfully loaded data: {len(df)} records")
    print(f"Classes found: {df['class'].unique().tolist()}")
    print(f"Configs found: {df['config'].unique().tolist()}")
    
    return df


def create_heatmaps(df, output_dir):
    """
    Generate performance matrices.
    1. KID Heatmap: Quality evaluation (lower is better).
    2. IC-LPIPS Heatmap: Diversity evaluation (higher is better).
    """
    print("\nGenerating performance heatmaps...")
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # === KID Heatmap ===
    pivot_kid = df.pivot_table(index='class', columns='config', values='KID_mean', aggfunc='mean')
    
    # Sort by Category (Object classes first, then Texture)
    class_order = [c for c in OBJECT_CLASSES if c in pivot_kid.index] + \
                  [c for c in TEXTURE_CLASSES if c in pivot_kid.index]
    pivot_kid = pivot_kid.reindex(class_order)
    
    config_order = ['base', 'tex', 'obj']
    pivot_kid = pivot_kid.reindex(columns=[c for c in config_order if c in pivot_kid.columns])
    
    ax1 = axes[0]
    sns.heatmap(pivot_kid, annot=True, fmt='.4f', cmap='RdYlGn_r',
                linewidths=0.5, ax=ax1,
                cbar_kws={'label': 'KID (lower is better)', 'shrink': 0.8},
                annot_kws={'size': 11, 'weight': 'bold'})
    ax1.set_title('KID Quality Evaluation Matrix', fontsize=14, fontweight='bold', pad=15)
    
    # Visual separators for dataset types
    n_object = len([c for c in OBJECT_CLASSES if c in class_order])
    if 0 < n_object < len(class_order):
        ax1.axhline(y=n_object, color='black', linewidth=2)
    
    # === IC-LPIPS Heatmap ===
    pivot_lpips = df.pivot_table(index='class', columns='config', values='IC_LPIPS_mean', aggfunc='mean')
    pivot_lpips = pivot_lpips.reindex(class_order)
    pivot_lpips = pivot_lpips.reindex(columns=[c for c in config_order if c in pivot_lpips.columns])
    
    ax2 = axes[1]
    sns.heatmap(pivot_lpips, annot=True, fmt='.4f', cmap='RdYlGn',
                linewidths=0.5, ax=ax2,
                cbar_kws={'label': 'IC-LPIPS (higher is better)', 'shrink': 0.8},
                annot_kws={'size': 11, 'weight': 'bold'})
    ax2.set_title('IC-LPIPS Diversity Evaluation Matrix', fontsize=14, fontweight='bold', pad=15)
    
    if 0 < n_object < len(class_order):
        ax2.axhline(y=n_object, color='black', linewidth=2)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'heatmaps.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  Heatmaps saved to: {output_path}")
    plt.close()


def create_scatter_plot(df, output_dir):
    """
    Generates a Quality vs. Diversity Trade-off scatter plot.
    Top-left corner represents the "Ideal Region" (Low KID, High Diversity).
    """
    print("\nGenerating Quality vs Diversity scatter plot...")
    
    fig, ax = plt.subplots(figsize=(12, 9))
    
    for _, row in df.iterrows():
        color = CONFIG_COLORS.get(row['config'], '#95a5a6')
        marker = CATEGORY_MARKERS.get(row['category_type'], 'o')
        
        ax.scatter(row['KID_mean'], row['IC_LPIPS_mean'],
                   c=color, marker=marker, s=200, alpha=0.8,
                   edgecolors='black', linewidth=1.5, zorder=3)
        
        # Annotate class names
        label_text = row['class'][:4] if len(row['class']) > 4 else row['class']
        ax.annotate(label_text, (row['KID_mean'], row['IC_LPIPS_mean']),
                    fontsize=8, ha='center', va='bottom', xytext=(0, 8), 
                    textcoords='offset points', fontweight='bold')
    
    # Reference Median Lines
    ax.axhline(y=df['IC_LPIPS_mean'].median(), color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=df['KID_mean'].median(), color='gray', linestyle='--', alpha=0.5)
    
    # Highlight the Ideal Region
    kid_min, lpips_max = df['KID_mean'].min(), df['IC_LPIPS_mean'].max()
    ax.annotate('Ideal Region\n(High Quality + High Diversity)', 
                xy=(kid_min, lpips_max), fontsize=11, color='#27ae60', fontweight='bold',
                ha='left', va='top', bbox=dict(boxstyle='round,pad=0.3', facecolor='#d5f4e6', edgecolor='#27ae60'))
    
    ax.set_xlabel('KID (Lower = Higher Fidelity)', fontsize=13, fontweight='bold')
    ax.set_ylabel('IC-LPIPS (Higher = More Diverse)', fontsize=13, fontweight='bold')
    ax.set_title('Quality vs Diversity Trade-off Analysis', fontsize=15, fontweight='bold', pad=15)
    
    # Dynamic Legend Generation
    legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor=c, markersize=12, label=f'Config {k.upper()}', markeredgecolor='black') 
                       for k, c in CONFIG_COLORS.items() if k in df['config'].values]
    
    ax.legend(handles=legend_elements, loc='upper right', framealpha=0.95)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'scatter_tradeoff.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  Scatter plot saved to: {output_path}")
    plt.close()


def create_grouped_bar_charts(df, output_dir):
    """Generates comparison bar charts across all classes and configs."""
    print("\nGenerating comparison bar charts...")
    
    fig, axes = plt.subplots(2, 1, figsize=(16, 12))
    all_classes = df['class'].unique().tolist()
    class_order = [c for c in OBJECT_CLASSES if c in all_classes] + \
                  [c for c in TEXTURE_CLASSES if c in all_classes]
    
    config_order = ['base', 'tex', 'obj']
    configs = [c for c in config_order if c in df['config'].values]
    
    bar_width = 0.25
    x = np.arange(len(class_order))
    
    for idx, (metric, ax_title, ylabel) in enumerate([
        ('KID', 'KID Quality Evaluation - Config Comparison', 'KID (lower is better)'),
        ('IC_LPIPS', 'IC-LPIPS Diversity Evaluation - Config Comparison', 'IC-LPIPS (higher is better)')
    ]):
        ax = axes[idx]
        for i, config in enumerate(configs):
            config_data = df[df['config'] == config].set_index('class')
            values = [config_data.loc[c, f'{metric}_mean'] if c in config_data.index else 0 for c in class_order]
            errors = [config_data.loc[c, f'{metric}_std'] if c in config_data.index else 0 for c in class_order]
            
            ax.bar(x + i * bar_width, values, bar_width, label=f'Config {config.upper()}',
                   color=CONFIG_COLORS.get(config, '#95a5a6'), yerr=errors, capsize=3, 
                   alpha=0.85, edgecolor='black', linewidth=0.5)
        
        ax.set_title(ax_title, fontsize=14, fontweight='bold')
        ax.set_ylabel(ylabel, fontweight='bold')
        ax.set_xticks(x + bar_width * (len(configs) - 1) / 2)
        ax.set_xticklabels(class_order, rotation=45, ha='right')
        ax.legend()

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'grouped_bar_charts.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Bar charts saved to: {output_path}")
    plt.close()


def create_summary_table(df, output_dir):
    """Calculates and saves grouped summary statistics."""
    print("\nCalculating summary statistics...")
    
    summary = df.groupby(['category_type', 'config']).agg({
        'KID_mean': ['mean', 'std', 'min', 'max'],
        'IC_LPIPS_mean': ['mean', 'std', 'min', 'max']
    }).round(4)
    
    summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
    summary_path = os.path.join(output_dir, 'summary_statistics.csv')
    summary.to_csv(summary_path)
    
    print("\n" + "="*80 + "\nSummary Statistics:\n" + "="*80)
    print(summary.to_string())
    return summary


def main():
    parser = argparse.ArgumentParser(description="Visualize DefectFill Experiment Results")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to evaluation results CSV")
    parser.add_argument("--output_dir", type=str, default="./figures", help="Output directory for plots")
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process all visualizations
    df = load_and_validate_data(args.csv_path)
    create_heatmaps(df, args.output_dir)
    create_scatter_plot(df, args.output_dir)
    create_grouped_bar_charts(df, args.output_dir)
    create_summary_table(df, args.output_dir)
    
    print(f"\nAll visualization charts generated in: {args.output_dir}")

if __name__ == "__main__":
    main()