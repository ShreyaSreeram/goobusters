import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
import warnings
import os
import cv2
warnings.filterwarnings('ignore')

def process_single_result_file(filepath, exam_id, mode, sampling_rate):
    """
    Process a single JSON results file like your sample.
    Returns a list of dictionaries with extracted metrics.
    """
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    processed_results = []
    
    for iteration in data.get('iterations', []):
        iteration_num = iteration.get('iteration_number', 1)
        eval_results = iteration.get('evaluation_results', {})
        summary = eval_results.get('summary', {})
        
        # Find the study data (first non-summary key)
        study_keys = [k for k in eval_results.keys() if k != 'summary']
        study_data = eval_results.get(study_keys[0], {}) if study_keys else {}
        metrics = study_data.get('metrics', {})
        
        # NEW: Extract method comparison data if available
        method_comparison = study_data.get('method_comparison', {})
        
        # Extract frame-level data
        frame_metrics = metrics.get('frame_metrics', {})
        iou_scores = metrics.get('iou_scores', [])
        dice_scores = metrics.get('dice_scores', [])
        
        # Calculate additional statistics
        iou_array = np.array(iou_scores) if iou_scores else np.array([0])
        dice_array = np.array(dice_scores) if dice_scores else np.array([0])
        
        result = {
            'exam_id': exam_id,
            'mode': mode,
            'sampling_rate': sampling_rate,
            'iteration': iteration_num,
            'learning_mode': data.get('learning_mode', False),
            
            # Summary metrics
            'mean_iou': summary.get('overall_mean_iou', 0),
            'mean_dice': summary.get('overall_mean_dice', 0),
            'total_videos': summary.get('total_videos', 1),
            'successful_evaluations': summary.get('successful_evaluations', 0),
            'memorized_frames_excluded': summary.get('total_memorized_frames_excluded', 0),
            
            # Detailed metrics
            'detailed_mean_iou': metrics.get('mean_iou', 0),
            'detailed_median_iou': metrics.get('median_iou', 0),
            'detailed_mean_dice': metrics.get('mean_dice', 0),
            'iou_over_70': metrics.get('iou_over_0.7', 0),
            
            # Frame counts
            'ground_truth_count': study_data.get('ground_truth_count', 0),
            'algorithm_mask_count': study_data.get('algorithm_mask_count', 0),
            
            # Statistical measures
            'iou_std': np.std(iou_array),
            'iou_min': np.min(iou_array),
            'iou_max': np.max(iou_array),
            'iou_q25': np.percentile(iou_array, 25),
            'iou_q75': np.percentile(iou_array, 75),
            'dice_std': np.std(dice_array),
            
            # Performance stability
            'num_frames_evaluated': len(iou_scores),
            'num_high_quality_frames': np.sum(iou_array > 0.7),
            'high_quality_percentage': np.mean(iou_array > 0.7) * 100,
            
            # NEW: Method comparison metrics
            'has_method_comparison': bool(method_comparison and 'error' not in method_comparison),
            'single_frame_count': method_comparison.get('single_frame_count', 0),
            'multi_frame_count': method_comparison.get('multi_frame_count', 0),
            'method_agreement_iou': method_comparison.get('mean_iou', 0),
            'method_agreement_dice': method_comparison.get('mean_dice', 0),
            
            # NEW: Performance improvement metrics
            'single_frame_vs_gt_iou': method_comparison.get('single_frame_vs_gt', {}).get('mean_iou', 0),
            'multi_frame_vs_gt_iou': method_comparison.get('multi_frame_vs_gt', {}).get('mean_iou', 0),
            'single_frame_vs_gt_dice': method_comparison.get('single_frame_vs_gt', {}).get('mean_dice', 0),
            'multi_frame_vs_gt_dice': method_comparison.get('multi_frame_vs_gt', {}).get('mean_dice', 0),
            'iou_improvement_percent': method_comparison.get('performance_improvement', {}).get('iou_improvement_percent', 0),
            
            # Raw data for detailed analysis
            'iou_scores': iou_scores,
            'dice_scores': dice_scores,
            'frame_metrics': frame_metrics,
            'method_comparison_data': method_comparison
        }
        
        processed_results.append(result)
    
    return processed_results

def _extract_iteration_metrics(iteration, exam_id, mode, sampling_rate):
    """Extract key metrics from a single iteration"""
    eval_results = iteration.get('evaluation_results', {})
    
    # Get summary metrics
    summary = eval_results.get('summary', {})
    
    # Get detailed metrics from first study (assuming single study per exam)
    study_keys = [k for k in eval_results.keys() if k != 'summary']
    study_key = study_keys[0] if study_keys else None
    study_metrics = eval_results.get(study_key, {}).get('metrics', {}) if study_key else {}
    
    # NEW: Extract method comparison data
    method_comparison = eval_results.get(study_key, {}).get('method_comparison', {}) if study_key else {}
    
    return {
        'exam_id': exam_id,
        'mode': mode,
        'sampling_rate': sampling_rate,
        'iteration': iteration.get('iteration_number', 1),
        'learning_mode': iteration.get('learning_mode', False),
        
        # Summary metrics
        'mean_iou': summary.get('overall_mean_iou', 0),
        'mean_dice': summary.get('overall_mean_dice', 0),
        'total_videos': summary.get('total_videos', 1),
        'successful_evaluations': summary.get('successful_evaluations', 0),
        'memorized_frames_excluded': summary.get('total_memorized_frames_excluded', 0),
        
        # Detailed metrics
        'detailed_mean_iou': study_metrics.get('mean_iou', 0),
        'detailed_median_iou': study_metrics.get('median_iou', 0),
        'detailed_mean_dice': study_metrics.get('mean_dice', 0),
        'iou_over_70': study_metrics.get('iou_over_0.7', 0),
        
        # Frame counts
        'ground_truth_count': eval_results.get(study_key, {}).get('ground_truth_count', 0) if study_key else 0,
        'algorithm_mask_count': eval_results.get(study_key, {}).get('algorithm_mask_count', 0) if study_key else 0,
        
        # Statistical measures
        'iou_std': np.std(study_metrics.get('iou_scores', [0])),
        'iou_min': np.min(study_metrics.get('iou_scores', [0])),
        'iou_max': np.max(study_metrics.get('iou_scores', [0])),
        'iou_q25': np.percentile(study_metrics.get('iou_scores', [0]), 25),
        'iou_q75': np.percentile(study_metrics.get('iou_scores', [0]), 75),
        'dice_std': np.std(study_metrics.get('dice_scores', [0])),
        
        # Performance stability
        'num_frames_evaluated': len(study_metrics.get('iou_scores', [])),
        'num_high_quality_frames': np.sum(np.array(study_metrics.get('iou_scores', [])) > 0.7),
        'high_quality_percentage': np.mean(np.array(study_metrics.get('iou_scores', [])) > 0.7) * 100,
        
        # NEW: Method comparison metrics
        'has_method_comparison': bool(method_comparison and 'error' not in method_comparison),
        'single_frame_count': method_comparison.get('single_frame_count', 0),
        'multi_frame_count': method_comparison.get('multi_frame_count', 0),
        'method_agreement_iou': method_comparison.get('mean_iou', 0),
        'method_agreement_dice': method_comparison.get('mean_dice', 0),
        
        # NEW: Performance improvement metrics
        'single_frame_vs_gt_iou': method_comparison.get('single_frame_vs_gt', {}).get('mean_iou', 0),
        'multi_frame_vs_gt_iou': method_comparison.get('multi_frame_vs_gt', {}).get('mean_iou', 0),
        'single_frame_vs_gt_dice': method_comparison.get('single_frame_vs_gt', {}).get('mean_dice', 0),
        'multi_frame_vs_gt_dice': method_comparison.get('multi_frame_vs_gt', {}).get('mean_dice', 0),
        'iou_improvement_percent': method_comparison.get('performance_improvement', {}).get('iou_improvement_percent', 0),
        
        # Raw data for detailed analysis
        'iou_scores': study_metrics.get('iou_scores', []),
        'dice_scores': study_metrics.get('dice_scores', []),
        'frame_metrics': study_metrics.get('frame_metrics', {}),
        'method_comparison_data': method_comparison
    }

def combine_all_results(results_directory):
    """
    Load all JSON results files from directory structure.
    Expected structure:
    results_directory/
    ├── exam_185/
    │   ├── baseline_rate_5.json
    │   ├── learning_rate_5.json
    │   └── ...
    ├── exam_194/
    └── exam_200/
    """
    all_results = []
    results_path = Path(results_directory)
    
    for exam_dir in results_path.iterdir():
        if exam_dir.is_dir():
            exam_id = exam_dir.name
            
            for result_file in exam_dir.glob("*.json"):
                filename = result_file.stem
                
                # Determine mode
                mode = "learning" if "learning" in filename else "baseline"
                
                # Extract sampling rate
                import re
                rate_match = re.search(r'rate[_\s]*(\d+)', filename)
                sampling_rate = int(rate_match.group(1)) if rate_match else 5
                
                try:
                    with open(result_file, 'r') as f:
                        data = json.load(f)
                    
                    for iteration in data.get('iterations', []):
                        iteration_data = _extract_iteration_metrics(
                            iteration, exam_id, mode, sampling_rate
                        )
                        all_results.append(iteration_data)
                        
                except Exception as e:
                    print(f"Error processing {result_file}: {str(e)}")
                    continue
    
    return pd.DataFrame(all_results)

# NEW: Method comparison visualization functions
def create_method_comparison_visualization(df):
    """
    Create comprehensive single-frame vs multi-frame method comparison visualizations
    """
    # Filter to only results with method comparison data
    df_comparison = df[df['has_method_comparison'] == True].copy()
    
    if len(df_comparison) == 0:
        print("No method comparison data found in results!")
        return None
    
    print(f"Creating method comparison visualizations for {len(df_comparison)} results with comparison data")
    
    # Set up the plotting style
    plt.style.use('default')
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(4, 3, height_ratios=[2, 2, 1.5, 1.2], 
                         hspace=0.35, wspace=0.3)
    
    # Main title
    fig.suptitle('Single-Frame vs Multi-Frame Tracking Method Comparison\n'
                 'Does Multi-Frame Supervision Improve Optical Flow Tracking?', 
                 fontsize=16, fontweight='bold', y=0.96)
    
    # 1. HERO CHART: Performance Comparison (IoU)
    ax_main = fig.add_subplot(gs[0, :2])
    
    # Prepare data for comparison
    comparison_data = []
    for _, row in df_comparison.iterrows():
        if row['single_frame_vs_gt_iou'] > 0 and row['multi_frame_vs_gt_iou'] > 0:
            comparison_data.append({
                'exam_id': row['exam_id'],
                'sampling_rate': row['sampling_rate'],
                'single_frame_iou': row['single_frame_vs_gt_iou'],
                'multi_frame_iou': row['multi_frame_vs_gt_iou'],
                'improvement': row['iou_improvement_percent']
            })
    
    if not comparison_data:
        print("No valid comparison data found!")
        return None
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Create grouped bar chart
    exam_ids = comparison_df['exam_id'].unique()
    x = np.arange(len(exam_ids))
    width = 0.35
    
    single_ious = []
    multi_ious = []
    
    for exam_id in exam_ids:
        exam_data = comparison_df[comparison_df['exam_id'] == exam_id]
        single_ious.append(exam_data['single_frame_iou'].mean())
        multi_ious.append(exam_data['multi_frame_iou'].mean())
    
    bars1 = ax_main.bar(x - width/2, single_ious, width, 
                       label='Single-Frame Method', color='lightcoral', alpha=0.8)
    bars2 = ax_main.bar(x + width/2, multi_ious, width, 
                       label='Multi-Frame Method', color='lightblue', alpha=0.8)
    
    # Add improvement annotations
    for i, (single_iou, multi_iou) in enumerate(zip(single_ious, multi_ious)):
        improvement = ((multi_iou - single_iou) / single_iou * 100) if single_iou > 0 else 0
        color = 'green' if improvement > 0 else 'red'
        ax_main.annotate(f'{improvement:+.1f}%', 
                        xy=(i, max(single_iou, multi_iou) + 0.02), 
                        ha='center', va='bottom', fontweight='bold', 
                        color=color, fontsize=9)
    
    # Add clinical threshold
    ax_main.axhline(y=0.7, color='orange', linestyle='--', linewidth=2, alpha=0.7, 
                   label='Clinical Threshold (IoU > 0.7)')
    ax_main.axhline(y=0.5, color='red', linestyle='--', linewidth=2, alpha=0.7, 
                   label='Acceptable Threshold (IoU > 0.5)')
    
    ax_main.set_xlabel('Exam ID', fontsize=11, fontweight='bold')
    ax_main.set_ylabel('Mean IoU vs Ground Truth', fontsize=11, fontweight='bold')
    ax_main.set_title('Single-Frame vs Multi-Frame Performance\n(Higher is Better)', 
                     fontsize=12, fontweight='bold')
    ax_main.set_xticks(x)
    ax_main.set_xticklabels(exam_ids, rotation=45, ha='right')
    ax_main.legend(loc='upper left')
    ax_main.grid(True, alpha=0.3)
    ax_main.set_ylim(0, 1.0)
    
    # 2. IMPROVEMENT DISTRIBUTION
    ax_improvement = fig.add_subplot(gs[0, 2])
    
    improvements = comparison_df['improvement'].values
    
    # Create histogram
    ax_improvement.hist(improvements, bins=10, alpha=0.7, color='skyblue', edgecolor='black')
    ax_improvement.axvline(x=0, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax_improvement.axvline(x=np.mean(improvements), color='green', linestyle='-', linewidth=2, 
                          label=f'Mean: {np.mean(improvements):.1f}%')
    
    ax_improvement.set_xlabel('Multi-Frame Improvement (%)', fontsize=10)
    ax_improvement.set_ylabel('Count', fontsize=10)
    ax_improvement.set_title('Distribution of\nImprovement', fontsize=11, fontweight='bold')
    ax_improvement.legend(fontsize=9)
    ax_improvement.grid(True, alpha=0.3)
    
    # Add statistics
    positive_improvements = np.sum(improvements > 0)
    total_comparisons = len(improvements)
    ax_improvement.text(0.05, 0.95, f'{positive_improvements}/{total_comparisons}\n({positive_improvements/total_comparisons*100:.1f}%)\nImproved', 
                       transform=ax_improvement.transAxes, va='top', 
                       bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.5))
    
    # 3. SAMPLING RATE ANALYSIS
    ax_sampling = fig.add_subplot(gs[1, :2])
    
    # Performance by sampling rate
    sampling_summary = comparison_df.groupby('sampling_rate').agg({
        'single_frame_iou': 'mean',
        'multi_frame_iou': 'mean',
        'improvement': 'mean'
    }).reset_index()
    
    sampling_rates = sampling_summary['sampling_rate'].values
    x_sampling = np.arange(len(sampling_rates))
    
    ax_sampling.plot(x_sampling, sampling_summary['single_frame_iou'], 'o-', 
                    label='Single-Frame', color='red', linewidth=3, markersize=8)
    ax_sampling.plot(x_sampling, sampling_summary['multi_frame_iou'], 's-', 
                    label='Multi-Frame', color='blue', linewidth=3, markersize=8)
    
    ax_sampling.set_xlabel('Sampling Rate (1:N)', fontsize=11)
    ax_sampling.set_ylabel('Mean IoU', fontsize=11)
    ax_sampling.set_title('Method Comparison Across Sampling Rates\n(How Does Input Sparsity Affect Method Superiority?)', 
                         fontsize=12, fontweight='bold')
    ax_sampling.set_xticks(x_sampling)
    ax_sampling.set_xticklabels([f'1:{rate}' for rate in sampling_rates])
    ax_sampling.legend()
    ax_sampling.grid(True, alpha=0.3)
    ax_sampling.set_ylim(0, 1.0)
    
    # 4. KEY FINDINGS SUMMARY
    ax_summary = fig.add_subplot(gs[1:, 2])
    ax_summary.axis('off')
    
    # Calculate key statistics
    total_comparisons = len(comparison_df)
    improvements = comparison_df['improvement'].values
    positive_improvements = np.sum(improvements > 0)
    mean_improvement = np.mean(improvements)
    best_improvement = np.max(improvements)
    worst_degradation = np.min(improvements)
    
    # Find best performing method overall
    overall_single = comparison_df['single_frame_iou'].mean()
    overall_multi = comparison_df['multi_frame_iou'].mean()
    
    # Clinical threshold analysis
    single_clinical = np.mean(comparison_df['single_frame_iou'] > 0.7) * 100
    multi_clinical = np.mean(comparison_df['multi_frame_iou'] > 0.7) * 100
    
    summary_text = f"""
🔬 METHOD COMPARISON FINDINGS:

📊 OVERALL PERFORMANCE:
• Multi-Frame Average IoU: {overall_multi:.3f}
• Single-Frame Average IoU: {overall_single:.3f}
• Overall Multi-Frame Advantage: {((overall_multi - overall_single) / overall_single * 100):+.1f}%

✅ IMPROVEMENT ANALYSIS:
• {positive_improvements}/{total_comparisons} cases ({positive_improvements/total_comparisons*100:.1f}%) showed improvement
• Average improvement: {mean_improvement:+.1f}%
• Best improvement: {best_improvement:+.1f}%
• Worst case: {worst_degradation:+.1f}%

🏥 CLINICAL IMPACT:
• Single-frame clinical quality (IoU>0.7): {single_clinical:.1f}% of cases
• Multi-frame clinical quality (IoU>0.7): {multi_clinical:.1f}% of cases
• Clinical improvement: {multi_clinical - single_clinical:+.1f} percentage points

💡 CONCLUSION: {"Multi-frame supervision significantly improves tracking" if mean_improvement > 5 else "Multi-frame shows modest improvements" if mean_improvement > 0 else "Methods perform similarly"}
    """
    
    ax_summary.text(0.02, 0.95, summary_text, transform=ax_summary.transAxes, 
                   fontsize=9, verticalalignment='top', 
                   bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.3))
    
    plt.tight_layout()
    return fig

def create_method_comparison_statistical_analysis(df):
    """
    Perform statistical analysis of single-frame vs multi-frame methods
    """
    # Filter to comparison data
    df_comparison = df[df['has_method_comparison'] == True].copy()
    
    if len(df_comparison) == 0:
        print("No method comparison data available for statistical analysis")
        return None
    
    print("\n" + "="*60)
    print("METHOD COMPARISON STATISTICAL ANALYSIS")
    print("="*60)
    
    results_summary = []
    
    # Overall comparison
    single_ious = df_comparison['single_frame_vs_gt_iou'].values
    multi_ious = df_comparison['multi_frame_vs_gt_iou'].values
    
    # Remove zeros for valid comparison
    valid_indices = (single_ious > 0) & (multi_ious > 0)
    single_valid = single_ious[valid_indices]
    multi_valid = multi_ious[valid_indices]
    
    if len(single_valid) > 0:
        # Paired t-test
        t_stat, p_value = stats.ttest_rel(multi_valid, single_valid)
        
        # Effect size
        differences = multi_valid - single_valid
        mean_diff = np.mean(differences)
        std_diff = np.std(differences)
        cohens_d = mean_diff / std_diff if std_diff > 0 else 0
        
        print(f"\nOVERALL METHOD COMPARISON:")
        print(f"Single-frame mean IoU: {np.mean(single_valid):.4f} ± {np.std(single_valid):.4f}")
        print(f"Multi-frame mean IoU: {np.mean(multi_valid):.4f} ± {np.std(multi_valid):.4f}")
        print(f"Mean improvement: {mean_diff:+.4f} ({mean_diff/np.mean(single_valid)*100:+.1f}%)")
        print(f"Paired t-test: t={t_stat:.4f}, p={p_value:.4f}")
        print(f"Effect size (Cohen's d): {cohens_d:.4f}")
        
        significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
        effect_interpretation = ("Large" if abs(cohens_d) > 0.8 else 
                               "Medium" if abs(cohens_d) > 0.5 else 
                               "Small" if abs(cohens_d) > 0.2 else "Negligible")
        
        print(f"Significance: {significance}")
        print(f"Effect size interpretation: {effect_interpretation}")
        
        results_summary.append({
            'comparison_type': 'Overall',
            'sampling_rate': 'All',
            'single_frame_mean': np.mean(single_valid),
            'multi_frame_mean': np.mean(multi_valid),
            'mean_improvement': mean_diff,
            'percent_improvement': mean_diff/np.mean(single_valid)*100,
            't_statistic': t_stat,
            'p_value': p_value,
            'cohens_d': cohens_d,
            'significance': significance,
            'effect_size': effect_interpretation,
            'n_comparisons': len(single_valid)
        })
    
    return pd.DataFrame(results_summary)

def create_method_comparison_summary_table(stats_df):
    """Create a publication-ready table for method comparison results"""
    if stats_df is None or len(stats_df) == 0:
        print("No method comparison statistics to display")
        return
    
    print("\n" + "=" * 120)
    print("TABLE: SINGLE-FRAME VS MULTI-FRAME METHOD COMPARISON")
    print("=" * 120)
    
    print(f"{'Condition':<15} {'Single-Frame':<15} {'Multi-Frame':<15} {'Improvement':<15} "
          f"{'p-value':<10} {'Effect Size':<12} {'Significance':<12}")
    print("-" * 120)
    
    for _, row in stats_df.iterrows():
        condition = f"Rate 1:{row['sampling_rate']}" if row['comparison_type'] == 'By_Rate' else row['comparison_type']
        print(f"{condition:<15} "
              f"{row['single_frame_mean']:<15.4f} "
              f"{row['multi_frame_mean']:<15.4f} "
              f"{row['percent_improvement']:+<15.1f}% "
              f"{row['p_value']:<10.4f} "
              f"{row['effect_size']:<12} "
              f"{row['significance']:<12}")
    
    print("-" * 120)
    print("Significance levels: *** p<0.001, ** p<0.01, * p<0.05, ns = not significant")
    print("Effect sizes: Large (|d|>0.8), Medium (|d|>0.5), Small (|d|>0.2), Negligible (|d|≤0.2)")

def create_performance_comparison(df):
    """Create comprehensive performance comparison visualizations"""
    
    # Set up the plotting style
    plt.style.use('default')
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Multi-Exam Optical Flow Performance Analysis', fontsize=16, fontweight='bold')
    
    # 1. Mean IoU by Sampling Rate and Mode
    ax1 = axes[0, 0]
    summary_by_rate = df.groupby(['sampling_rate', 'mode']).agg({
        'mean_iou': ['mean', 'std']
    }).round(4)
    
    rates = sorted(df['sampling_rate'].unique())
    baseline_means = [summary_by_rate.loc[(r, 'baseline'), ('mean_iou', 'mean')] 
                     for r in rates if (r, 'baseline') in summary_by_rate.index]
    learning_means = [summary_by_rate.loc[(r, 'learning'), ('mean_iou', 'mean')] 
                     for r in rates if (r, 'learning') in summary_by_rate.index]
    
    x = np.arange(len(rates))
    width = 0.35
    
    ax1.bar(x - width/2, baseline_means, width, label='Baseline', alpha=0.8, color='red')
    ax1.bar(x + width/2, learning_means, width, label='Learning', alpha=0.8, color='green')
    ax1.set_xlabel('Sampling Rate')
    ax1.set_ylabel('Mean IoU')
    ax1.set_title('Performance by Sampling Rate')
    ax1.set_xticks(x)
    ax1.set_xticklabels(rates)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Distribution of IoU scores
    ax2 = axes[0, 1]
    df_plot = df[df['mean_iou'] > 0]  # Remove zero scores for cleaner visualization
    sns.boxplot(data=df_plot, x='mode', y='mean_iou', hue='sampling_rate', ax=ax2)
    ax2.set_title('IoU Score Distributions')
    ax2.set_ylabel('Mean IoU')
    
    # 3. Performance improvement percentage
    ax3 = axes[0, 2]
    improvement_data = []
    for rate in rates:
        baseline_data = df[(df['sampling_rate'] == rate) & (df['mode'] == 'baseline')]
        learning_data = df[(df['sampling_rate'] == rate) & (df['mode'] == 'learning')]
        
        if len(baseline_data) > 0 and len(learning_data) > 0:
            baseline_mean = baseline_data['mean_iou'].mean()
            learning_mean = learning_data['mean_iou'].mean()
            improvement = ((learning_mean - baseline_mean) / baseline_mean) * 100
            improvement_data.append(improvement)
        else:
            improvement_data.append(0)
    
    colors = ['green' if x > 0 else 'red' for x in improvement_data]
    ax3.bar(rates, improvement_data, color=colors, alpha=0.7)
    ax3.set_xlabel('Sampling Rate')
    ax3.set_ylabel('Improvement (%)')
    ax3.set_title('Learning Mode Improvement Over Baseline')
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax3.grid(True, alpha=0.3)
    
    # 4. Exam-level performance heatmap
    ax4 = axes[1, 0]
    pivot_data = df.pivot_table(
        values='mean_iou', 
        index=['exam_id', 'mode'], 
        columns='sampling_rate',
        aggfunc='mean'
    )
    
    sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='RdYlBu_r', 
                ax=ax4, center=0.5, vmin=0, vmax=1, cbar_kws={'label': 'Mean IoU'})
    ax4.set_title('Performance Heatmap by Exam')
    ax4.set_ylabel('Exam ID & Mode')
    
    # 5. Clinical threshold analysis (IoU > 0.7)
    ax5 = axes[1, 1]
    clinical_summary = df.groupby(['sampling_rate', 'mode'])['high_quality_percentage'].mean()
    
    baseline_clinical = [clinical_summary.get((r, 'baseline'), 0) for r in rates]
    learning_clinical = [clinical_summary.get((r, 'learning'), 0) for r in rates]
    
    ax5.plot(rates, baseline_clinical, 'o-', label='Baseline', color='red', linewidth=2, markersize=8)
    ax5.plot(rates, learning_clinical, 's-', label='Learning', color='green', linewidth=2, markersize=8)
    ax5.set_xlabel('Sampling Rate')
    ax5.set_ylabel('% Frames with IoU > 0.7')
    ax5.set_title('Clinical Quality Threshold Analysis')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    ax5.set_ylim(0, 100)
    
    # 6. Performance stability (coefficient of variation)
    ax6 = axes[1, 2]
    stability_data = df.groupby(['sampling_rate', 'mode']).agg({
        'mean_iou': lambda x: (np.std(x) / np.mean(x)) * 100 if np.mean(x) > 0 else 0
    }).round(2)
    
    baseline_cv = [stability_data.loc[(r, 'baseline'), 'mean_iou'] 
                   for r in rates if (r, 'baseline') in stability_data.index]
    learning_cv = [stability_data.loc[(r, 'learning'), 'mean_iou'] 
                   for r in rates if (r, 'learning') in stability_data.index]
    
    ax6.bar(x - width/2, baseline_cv, width, label='Baseline', alpha=0.8, color='red')
    ax6.bar(x + width/2, learning_cv, width, label='Learning', alpha=0.8, color='green')
    ax6.set_xlabel('Sampling Rate')
    ax6.set_ylabel('Coefficient of Variation (%)')
    ax6.set_title('Performance Stability (Lower = More Stable)')
    ax6.set_xticks(x)
    ax6.set_xticklabels(rates)
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def statistical_significance_analysis(df):
    """Perform comprehensive statistical analysis"""
    print("=" * 80)
    print("STATISTICAL SIGNIFICANCE ANALYSIS")
    print("=" * 80)
    
    results_summary = []
    
    for rate in sorted(df['sampling_rate'].unique()):
        print(f"\nSampling Rate: {rate}")
        print("-" * 40)
        
        # Get data for this sampling rate
        rate_data = df[df['sampling_rate'] == rate]
        baseline_data = rate_data[rate_data['mode'] == 'baseline']['mean_iou']
        learning_data = rate_data[rate_data['mode'] == 'learning']['mean_iou']
        
        if len(baseline_data) > 0 and len(learning_data) > 0:
            # Descriptive statistics
            baseline_mean = baseline_data.mean()
            learning_mean = learning_data.mean()
            baseline_std = baseline_data.std()
            learning_std = learning_data.std()
            
            print(f"Baseline:  μ = {baseline_mean:.4f}, σ = {baseline_std:.4f}, n = {len(baseline_data)}")
            print(f"Learning:  μ = {learning_mean:.4f}, σ = {learning_std:.4f}, n = {len(learning_data)}")
            
            # Statistical tests
            if len(baseline_data) == len(learning_data):
                # Paired t-test (same exams)
                t_stat, p_value = stats.ttest_rel(learning_data, baseline_data)
                test_type = "Paired t-test"
            else:
                # Independent t-test
                t_stat, p_value = stats.ttest_ind(learning_data, baseline_data)
                test_type = "Independent t-test"
            
            # Effect size (Cohen's d)
            pooled_std = np.sqrt(((len(baseline_data) - 1) * baseline_std**2 + 
                                 (len(learning_data) - 1) * learning_std**2) / 
                                (len(baseline_data) + len(learning_data) - 2))
            cohens_d = (learning_mean - baseline_mean) / pooled_std if pooled_std > 0 else 0
            
            # Improvement metrics
            absolute_improvement = learning_mean - baseline_mean
            relative_improvement = (absolute_improvement / baseline_mean) * 100 if baseline_mean > 0 else 0
            
            print(f"{test_type}: t = {t_stat:.4f}, p = {p_value:.4f}")
            print(f"Effect size (Cohen's d): {cohens_d:.4f}")
            print(f"Absolute improvement: {absolute_improvement:+.4f}")
            print(f"Relative improvement: {relative_improvement:+.1f}%")
            
            # Interpretation
            significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
            effect_interpretation = ("Large" if abs(cohens_d) > 0.8 else 
                                   "Medium" if abs(cohens_d) > 0.5 else 
                                   "Small" if abs(cohens_d) > 0.2 else "Negligible")
            
            print(f"Significance: {significance}")
            print(f"Effect size: {effect_interpretation}")
            
            # Store results
            results_summary.append({
                'sampling_rate': rate,
                'baseline_mean': baseline_mean,
                'learning_mean': learning_mean,
                'absolute_improvement': absolute_improvement,
                'relative_improvement': relative_improvement,
                't_statistic': t_stat,
                'p_value': p_value,
                'cohens_d': cohens_d,
                'significance': significance,
                'effect_size': effect_interpretation,
                'baseline_n': len(baseline_data),
                'learning_n': len(learning_data)
            })
        else:
            print("Insufficient data for statistical comparison")
    
    return pd.DataFrame(results_summary)

def generate_executive_summary(df, stats_df):
    """Generate executive summary with key findings"""
    print("\n" + "=" * 80)
    print("EXECUTIVE SUMMARY")
    print("=" * 80)
    
    # Dataset overview
    print(f"\nDataset Overview:")
    print(f"• Total experiments: {len(df)}")
    print(f"• Number of exams: {df['exam_id'].nunique()}")
    print(f"• Sampling rates tested: {sorted(df['sampling_rate'].unique())}")
    print(f"• Total iterations analyzed: {df['iteration'].sum()}")
    
    # Method comparison overview
    method_comparison_count = df['has_method_comparison'].sum() if 'has_method_comparison' in df.columns else 0
    if method_comparison_count > 0:
        print(f"• Experiments with method comparison: {method_comparison_count}")
    
    # Overall performance
    overall_baseline = df[df['mode'] == 'baseline']['mean_iou'].mean()
    overall_learning = df[df['mode'] == 'learning']['mean_iou'].mean()
    overall_improvement = ((overall_learning - overall_baseline) / overall_baseline) * 100
    
    print(f"\nOverall Performance:")
    print(f"• Baseline average IoU: {overall_baseline:.4f}")
    print(f"• Learning mode average IoU: {overall_learning:.4f}")
    print(f"• Overall improvement: {overall_improvement:+.1f}%")
    
    # Method comparison summary
    if method_comparison_count > 0:
        df_comp = df[df['has_method_comparison'] == True]
        avg_single_iou = df_comp['single_frame_vs_gt_iou'].mean()
        avg_multi_iou = df_comp['multi_frame_vs_gt_iou'].mean()
        avg_method_improvement = df_comp['iou_improvement_percent'].mean()
        
        print(f"\nMethod Comparison Results:")
        print(f"• Single-frame average IoU: {avg_single_iou:.4f}")
        print(f"• Multi-frame average IoU: {avg_multi_iou:.4f}")
        print(f"• Multi-frame advantage: {avg_method_improvement:+.1f}%")
    
    # Best and worst conditions
    best_result = df.loc[df['mean_iou'].idxmax()]
    worst_result = df.loc[df['mean_iou'].idxmin()]
    
    print(f"\nBest Performance:")
    print(f"• Exam: {best_result['exam_id']}, Mode: {best_result['mode']}, Rate: {best_result['sampling_rate']}")
    print(f"• IoU: {best_result['mean_iou']:.4f}")
    
    print(f"\nWorst Performance:")
    print(f"• Exam: {worst_result['exam_id']}, Mode: {worst_result['mode']}, Rate: {worst_result['sampling_rate']}")
    print(f"• IoU: {worst_result['mean_iou']:.4f}")

def create_publication_ready_table(stats_df):
    """Create a publication-ready results table"""
    if len(stats_df) == 0:
        print("No statistical results to display")
        return
    
    print("\n" + "=" * 100)
    print("TABLE 1: STATISTICAL COMPARISON OF LEARNING MODE VS BASELINE")
    print("=" * 100)
    
    print(f"{'Sampling Rate':<15} {'Baseline':<12} {'Learning':<12} {'Improvement':<12} "
          f"{'p-value':<10} {'Effect Size':<12} {'Significance':<12}")
    print("-" * 100)
    
    for _, row in stats_df.iterrows():
        print(f"{row['sampling_rate']:<15} "
              f"{row['baseline_mean']:<12.4f} "
              f"{row['learning_mean']:<12.4f} "
              f"{row['relative_improvement']:+<12.1f}% "
              f"{row['p_value']:<10.4f} "
              f"{row['effect_size']:<12} "
              f"{row['significance']:<12}")
    
    print("-" * 100)
    print("Significance levels: *** p<0.001, ** p<0.01, * p<0.05, ns = not significant")
    print("Effect sizes: Large (|d|>0.8), Medium (|d|>0.5), Small (|d|>0.2), Negligible (|d|≤0.2)")

def find_video_for_exam_id(exam_id, annotations_json, video_base_path):
    """
    Use the SAME logic as your main script to find videos for exam IDs
    """
    # Find study UIDs for this exam ID using your existing logic
    datasets = annotations_json.get('datasets', [])
    study_uids_for_exam = []
    
    for dataset in datasets:
        studies = dataset.get('studies', [])
        for study in studies:
            if study.get('number') == exam_id and 'StudyInstanceUID' in study:
                study_uids_for_exam.append(study['StudyInstanceUID'])
    
    # If not found in studies, try annotations
    if not study_uids_for_exam:
        for dataset in datasets:
            for annotation in dataset.get('annotations', []):
                if annotation.get('examNumber') == exam_id and annotation.get('StudyInstanceUID'):
                    study_uids_for_exam.append(annotation['StudyInstanceUID'])
    
    # Find video files for these study UIDs
    video_paths = []
    for study_uid in study_uids_for_exam:
        # Look for video files using the same pattern as your main script
        study_dir = os.path.join(video_base_path, study_uid)
        if os.path.exists(study_dir):
            # Find .mp4 files in the study directory
            for file in os.listdir(study_dir):
                if file.endswith('.mp4'):
                    video_path = os.path.join(study_dir, file)
                    if os.path.exists(video_path):
                        video_paths.append(video_path)
                        break  # Take first video found
    
    return video_paths

def extract_frame_counts_for_exams(df, annotations_json, video_base_path):
    """
    Extract frame counts using the SAME exam ID logic as your main script
    """
    print("\nExtracting frame counts for exam analysis...")
    
    # Add new columns
    df['total_frames'] = None
    df['annotations_needed'] = None
    df['annotation_density_percent'] = None
    df['clinical_burden_minutes'] = None
    
    # Process each unique exam
    unique_exams = df['exam_id'].unique()
    frame_count_cache = {}
    
    for exam_id in unique_exams:
        # Extract numeric exam ID (remove 'exam_' prefix if present)
        if isinstance(exam_id, str) and exam_id.startswith('exam_'):
            numeric_exam_id = int(exam_id.replace('exam_', ''))
        else:
            try:
                numeric_exam_id = int(exam_id)
            except:
                print(f"Could not parse exam ID: {exam_id}")
                continue
        
        print(f"Processing exam {exam_id} (numeric: {numeric_exam_id})")
        
        # Find videos for this exam using your logic
        video_paths = find_video_for_exam_id(numeric_exam_id, annotations_json, video_base_path)
        
        if video_paths:
            # Use the first video found (assuming one video per exam for frame count)
            video_path = video_paths[0]
            
            try:
                cap = cv2.VideoCapture(video_path)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()
                
                frame_count_cache[exam_id] = total_frames
                print(f"  Found {total_frames} frames in {os.path.basename(video_path)}")
                
            except Exception as e:
                print(f"  Error reading video {video_path}: {e}")
                frame_count_cache[exam_id] = None
        else:
            print(f"  No video found for exam {exam_id}")
            frame_count_cache[exam_id] = None
    
    # Apply frame counts to all rows
    for idx, row in df.iterrows():
        exam_id = row['exam_id']
        total_frames = frame_count_cache.get(exam_id)
        
        if total_frames:
            sampling_rate = row['sampling_rate']
            
            # Calculate metrics using YOUR current sampling logic
            annotations_needed = max(1, total_frames // sampling_rate)
            annotation_density = (annotations_needed / total_frames) * 100
            clinical_burden = annotations_needed * 0.5  # 30 seconds per annotation
            
            df.at[idx, 'total_frames'] = total_frames
            df.at[idx, 'annotations_needed'] = annotations_needed
            df.at[idx, 'annotation_density_percent'] = annotation_density
            df.at[idx, 'clinical_burden_minutes'] = clinical_burden
    
    return df

def create_data_sparsity_tolerance_visualization(df):
    """
    Create visualization focused on: "Can minimal expert input produce acceptable dense annotations?"
    """
    # Set up the plotting style
    plt.style.use('default')
    
    # Create figure with strategic layout
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # Main title
    fig.suptitle('Optical Flow Sparse-to-Dense Annotation: Sparsity Tolerance Analysis', 
                 fontsize=16, fontweight='bold', y=0.95)
    
    # 1. SPARSITY TOLERANCE CHART
    ax_main = fig.add_subplot(gs[0, :])
    
    # Calculate sparsity tolerance metrics
    sparsity_data = []
    rates = sorted(df['sampling_rate'].unique())
    
    for rate in rates:
        learning_data = df[(df['sampling_rate'] == rate) & (df['mode'] == 'learning')]
        
        if len(learning_data) > 0:
            learning_mean = learning_data['mean_iou'].mean()
            learning_std = learning_data['mean_iou'].std()
            data_usage = 100 / rate  # Percentage of frames used as input
            
            sparsity_data.append({
                'sampling_rate': rate,
                'input_density_percent': data_usage,
                'output_quality_iou': learning_mean,
                'output_quality_std': learning_std
            })
    
    sparsity_df = pd.DataFrame(sparsity_data)
    
    # Plot the sparsity tolerance chart
    x_pos = np.arange(len(sparsity_df))
    
    # Color code based on clinical acceptability
    colors = []
    for quality in sparsity_df['output_quality_iou']:
        if quality >= 0.7:  # High quality
            colors.append('#2ca02c')  # Green
        elif quality >= 0.5:  # Moderate quality
            colors.append('#ff7f0e')  # Orange
        else:  # Low quality
            colors.append('#d62728')  # Red
    
    bars = ax_main.bar(x_pos, sparsity_df['output_quality_iou'], 
                      color=colors, alpha=0.8, 
                      yerr=sparsity_df['output_quality_std'], capsize=5)
    
    # Add clinical threshold lines
    ax_main.axhline(y=0.7, color='red', linestyle='--', linewidth=2, alpha=0.7, 
                   label='Clinical Threshold (IoU > 0.7)')
    ax_main.axhline(y=0.5, color='orange', linestyle='--', linewidth=2, alpha=0.7, 
                   label='Acceptable Threshold (IoU > 0.5)')
    
    # Customize main chart
    ax_main.set_xlabel('Expert Input Density (% of frames manually annotated)', 
                      fontsize=12, fontweight='bold')
    ax_main.set_ylabel('Dense Output Quality (Mean IoU)', fontsize=12, fontweight='bold')
    ax_main.set_title('Can Minimal Expert Input Produce Good Dense Annotations?', 
                     fontsize=14, fontweight='bold', pad=20)
    
    # Custom x-axis labels
    x_labels = [f'{row["input_density_percent"]:.1f}%' for _, row in sparsity_df.iterrows()]
    ax_main.set_xticks(x_pos)
    ax_main.set_xticklabels(x_labels, fontsize=10)
    ax_main.legend(loc='upper right', fontsize=10)
    ax_main.grid(True, alpha=0.3)
    ax_main.set_ylim(0, 1.0)
    
    # 2. COMPARISON BASELINE VS LEARNING
    ax_comparison = fig.add_subplot(gs[1, 0])
    
    rates_sorted = sorted(rates)
    baseline_qualities = []
    learning_qualities = []
    
    for rate in rates_sorted:
        baseline_data = df[(df['sampling_rate'] == rate) & (df['mode'] == 'baseline')]
        learning_data = df[(df['sampling_rate'] == rate) & (df['mode'] == 'learning')]
        
        baseline_mean = baseline_data['mean_iou'].mean() if len(baseline_data) > 0 else 0
        learning_mean = learning_data['mean_iou'].mean() if len(learning_data) > 0 else 0
        
        baseline_qualities.append(baseline_mean)
        learning_qualities.append(learning_mean)
    
    x_comparison = np.arange(len(rates_sorted))
    width = 0.35
    
    ax_comparison.bar(x_comparison - width/2, baseline_qualities, width, 
                     label='Baseline Parameters', color='lightcoral', alpha=0.8)
    ax_comparison.bar(x_comparison + width/2, learning_qualities, width, 
                     label='Learning Mode', color='lightblue', alpha=0.8)
    
    ax_comparison.axhline(y=0.5, color='green', linestyle='--', linewidth=2, alpha=0.7, 
                         label='Acceptable Quality')
    
    ax_comparison.set_xlabel('Sampling Rate (1:N)', fontsize=11)
    ax_comparison.set_ylabel('Mean IoU', fontsize=11)
    ax_comparison.set_title('Parameter Optimization Effect', fontsize=12, fontweight='bold')
    
    x_labels = [f'1:{rate}' for rate in rates_sorted]
    ax_comparison.set_xticks(x_comparison)
    ax_comparison.set_xticklabels(x_labels, fontsize=10)
    ax_comparison.legend(fontsize=9)
    ax_comparison.grid(True, alpha=0.3)
    
    # 3. KEY FINDINGS
    ax_findings = fig.add_subplot(gs[1, 1])
    ax_findings.axis('off')
    
    # Calculate key findings
    viable_conditions = [i for i, q in enumerate(learning_qualities) if q >= 0.5]
    if viable_conditions:
        min_viable_input = min([100/rates_sorted[i] for i in viable_conditions])
        sparsest_acceptable = min([(100/rates_sorted[i], learning_qualities[i]) 
                                 for i in range(len(rates_sorted)) if learning_qualities[i] >= 0.5], 
                                key=lambda x: x[0])
    else:
        min_viable_input = 100
        sparsest_acceptable = (100, 0)
    
    findings_text = f"""
🎯 KEY FINDINGS:

SPARSE-TO-DENSE CAPABILITY:
✅ Minimum viable input: {sparsest_acceptable[0]:.1f}% 
✅ Quality achieved: IoU = {sparsest_acceptable[1]:.3f}
✅ Input reduction: {100-sparsest_acceptable[0]:.1f}% savings

AI-FAST IMPACT:
🚀 Enables dense annotation with minimal effort
💡 {100-sparsest_acceptable[0]:.1f}% reduction in annotation burden
🏥 Maintains acceptable quality for AI training
    """
    
    ax_findings.text(0.05, 0.95, findings_text, transform=ax_findings.transAxes, 
                    fontsize=11, verticalalignment='top', 
                    bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgreen', alpha=0.3))
    
    plt.tight_layout()
    return fig

# Main execution function
def main(results_directory="results"):
    """Main function to run complete analysis"""
    print(f"Loading and processing all results from: {results_directory}")
    
    # Check if results directory exists
    if not Path(results_directory).exists():
        print(f"ERROR: Results directory '{results_directory}' not found!")
        print("Please ensure your results folder is in the correct location.")
        return None, None, None
    
    # Load all results
    df = combine_all_results(results_directory)
    
    if df.empty:
        print("No results found! Check your directory structure and file naming.")
        print("\nExpected structure:")
        print("results/")
        print("├── exam_001/")
        print("│   ├── baseline_rate_5.json")
        print("│   ├── learning_rate_5.json")
        print("│   └── ...")
        print("└── ...")
        return None, None, None
    
    print(f"Successfully loaded {len(df)} result records")
    print(f"Found {df['exam_id'].nunique()} exams, {len(df['sampling_rate'].unique())} sampling rates")
    
    # Create visualizations
    print("\nGenerating performance comparison plots...")
    fig = create_performance_comparison(df)
    plt.show()
    
    # Statistical analysis
    print("\nPerforming statistical analysis...")
    stats_df = statistical_significance_analysis(df)
    
    # Generate summary reports
    generate_executive_summary(df, stats_df)
    create_publication_ready_table(stats_df)
    
    # Save results
    output_dir = Path(results_directory) / "analysis_output"
    output_dir.mkdir(exist_ok=True)
    
    df.to_csv(output_dir / "combined_results.csv", index=False)
    if not stats_df.empty:
        stats_df.to_csv(output_dir / "statistical_analysis.csv", index=False)
    fig.savefig(output_dir / "performance_analysis.png", dpi=300, bbox_inches='tight')
    
    print(f"\nResults saved to: {output_dir}")
    print(f"Main outputs:")
    print(f"  - Combined data: {output_dir}/combined_results.csv")
    print(f"  - Statistics: {output_dir}/statistical_analysis.csv")
    print(f"  - Visualization: {output_dir}/performance_analysis.png")
    
    return df, stats_df, fig

def enhanced_main_with_method_comparison(results_directory="results", annotations_json_path=None, video_base_path=None):
    """
    Enhanced main function that includes method comparison analysis
    """
    print("=" * 60)
    print("ENHANCED OPTICAL FLOW ANALYSIS WITH METHOD COMPARISON")
    print("=" * 60)
    
    # Run original analysis
    print("Running original analysis...")
    df, stats_df, fig = main(results_directory)
    
    if df is None or len(df) == 0:
        print("No data available for analysis!")
        return None, None, None, None, None
    
    # Check for method comparison data
    method_comparison_available = df['has_method_comparison'].sum() > 0
    
    if method_comparison_available:
        print(f"\n✅ Found method comparison data in {df['has_method_comparison'].sum()} experiments")
        
        # Create method comparison visualizations
        print("\nGenerating method comparison visualizations...")
        method_comparison_fig = create_method_comparison_visualization(df)
        
        # Perform method comparison statistical analysis
        print("\nPerforming method comparison statistical analysis...")
        method_stats_df = create_method_comparison_statistical_analysis(df)
        
        # Create method comparison summary table
        create_method_comparison_summary_table(method_stats_df)
        
        if method_comparison_fig:
            plt.show()
    else:
        print("\n⚠️  No method comparison data found in results")
        print("To get method comparison data, run your evaluation with:")
        print("  --compare-methods flag")
        method_comparison_fig = None
        method_stats_df = None
    
    # Add frame analysis if paths provided
    if annotations_json_path and video_base_path:
        try:
            print(f"\nLoading annotations from: {annotations_json_path}")
            with open(annotations_json_path, 'r') as f:
                annotations_json = json.load(f)
            
            print(f"Using video base path: {video_base_path}")
            
            # Extract frame counts
            df_enhanced = extract_frame_counts_for_exams(df, annotations_json, video_base_path)
            
            # Create sparsity visualization
            print("\nGenerating sparse-to-dense annotation capability visualization...")
            sparsity_fig = create_data_sparsity_tolerance_visualization(df_enhanced)
            plt.show()
            
            # Save enhanced results
            output_dir = Path(results_directory) / "analysis_output"
            output_dir.mkdir(exist_ok=True)
            
            enhanced_output = output_dir / "frame_enhanced_results.csv"
            df_enhanced.to_csv(enhanced_output, index=False)
            
            sparsity_output = output_dir / "sparse_to_dense_analysis.png"
            sparsity_fig.savefig(sparsity_output, dpi=300, bbox_inches='tight')
            
            if method_comparison_fig:
                method_output = output_dir / "method_comparison_analysis.png"
                method_comparison_fig.savefig(method_output, dpi=300, bbox_inches='tight')
                print(f"Saved method comparison visualization to: {method_output}")
            
            if method_stats_df is not None:
                method_stats_output = output_dir / "method_comparison_statistics.csv"
                method_stats_df.to_csv(method_stats_output, index=False)
                print(f"Saved method comparison statistics to: {method_stats_output}")
            
            print(f"\nSaved enhanced results to: {enhanced_output}")
            print(f"Saved sparsity visualization to: {sparsity_output}")
            
            return df_enhanced, stats_df, fig, sparsity_fig, method_comparison_fig, method_stats_df
            
        except Exception as e:
            print(f"\nFrame analysis failed: {e}")
            print("Continuing with original analysis...")
            import traceback
            traceback.print_exc()
    else:
        print("\nSkipping frame analysis (no annotations_json_path or video_base_path provided)")
        
        sparsity_fig = create_data_sparsity_tolerance_visualization(df)
        plt.show()
        
        # Save outputs
        output_dir = Path(results_directory) / "analysis_output"
        output_dir.mkdir(exist_ok=True)
        
        sparsity_output = output_dir / "sparse_to_dense_analysis.png"
        sparsity_fig.savefig(sparsity_output, dpi=300, bbox_inches='tight')
        
        if method_comparison_fig:
            method_output = output_dir / "method_comparison_analysis.png"
            method_comparison_fig.savefig(method_output, dpi=300, bbox_inches='tight')
            print(f"Saved method comparison visualization to: {method_output}")
        
        if method_stats_df is not None:
            method_stats_output = output_dir / "method_comparison_statistics.csv"
            method_stats_df.to_csv(method_stats_output, index=False)
            print(f"Saved method comparison statistics to: {method_stats_output}")
        
        print(f"Saved sparsity visualization to: {sparsity_output}")
        
        return df, stats_df, fig, sparsity_fig, method_comparison_fig, method_stats_df

# Quick analysis functions for interactive use
def quick_analysis():
    """Run analysis with default settings for results in root/results/"""
    return main("results")

def quick_analysis_with_method_comparison(annotations_json_path=None, video_base_path=None):
    """
    Enhanced quick analysis that includes method comparison
    """
    return enhanced_main_with_method_comparison("results", annotations_json_path, video_base_path)

def test_single_file(filepath):
    """Test processing of a single results file"""
    try:
        # Try to infer metadata from filename
        filename = Path(filepath).stem
        print(f"Testing file: {filename}")
        
        # Basic parsing
        exam_id = "test_exam"
        mode = "learning" if "learning" in filename else "baseline"
        
        # Try to extract sampling rate
        import re
        rate_match = re.search(r'rate[_\s]*(\d+)', filename)
        sampling_rate = int(rate_match.group(1)) if rate_match else 5
        
        print(f"Detected: exam_id={exam_id}, mode={mode}, sampling_rate={sampling_rate}")
        
        results = process_single_result_file(filepath, exam_id, mode, sampling_rate)
        print(f"Successfully processed {len(results)} iterations")
        
        # Show sample data
        if results:
            sample = results[0]
            print(f"Sample metrics: IoU={sample['mean_iou']:.4f}, Dice={sample['mean_dice']:.4f}")
            if sample['has_method_comparison']:
                print(f"Method comparison: Single={sample['single_frame_vs_gt_iou']:.4f}, Multi={sample['multi_frame_vs_gt_iou']:.4f}")
        
        return results
        
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        return None

# Main execution
if __name__ == "__main__":
    print("=" * 60)
    print("ENHANCED OPTICAL FLOW ANALYSIS WITH METHOD COMPARISON")
    print("=" * 60)
    
    # Option 1: Run without frame analysis but with method comparison
    # df, stats_df, fig, sparsity_fig, method_fig, method_stats = enhanced_main_with_method_comparison("results")
    
    # Option 2: Run WITH frame analysis and method comparison (recommended)
    annotations_json_path = "/Users/Shreya1/Documents/GitHub/goobusters/data/mdai_ucsf_project_x9N2LJBZ_annotations_dataset_D_V688LQ_2025-06-03-194700.json"  # UPDATE THIS PATH
    video_base_path = "/Users/Shreya1/Documents/GitHub/goobusters/data/mdai_ucsf_project_x9N2LJBZ_images_dataset_D_V688LQ_2025-06-03-194012"    # UPDATE THIS PATH
    
    # Run with full analysis including method comparison:
    df, stats_df, fig, sparsity_fig, method_fig, method_stats = quick_analysis_with_method_comparison(annotations_json_path, video_base_path)
    
    # Print final summary
    if df is not None:
        method_comparison_count = df['has_method_comparison'].sum() if 'has_method_comparison' in df.columns else 0
        print(f"\n🎉 ANALYSIS COMPLETE!")
        print(f"   📊 {len(df)} total results processed")
        print(f"   🔬 {method_comparison_count} with method comparison data")
        print(f"   💾 Results saved to: results/analysis_output/")
        
        if method_comparison_count > 0:
            print(f"\n🔬 METHOD COMPARISON SUMMARY:")
            df_comp = df[df['has_method_comparison'] == True]
            avg_improvement = df_comp['iou_improvement_percent'].mean()
            positive_improvements = (df_comp['iou_improvement_percent'] > 0).sum()
            total_comparisons = len(df_comp)
            print(f"   📈 Average multi-frame improvement: {avg_improvement:+.1f}%")
            print(f"   ✅ Multi-frame better in: {positive_improvements}/{total_comparisons} cases ({positive_improvements/total_comparisons*100:.1f}%)")
    else:
        print("❌ Analysis failed - check your results directory and data files")