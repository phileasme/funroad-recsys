import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

def run_evaluation_pipeline(gumroad_dir, endpoint_dir, expert_dir, output_dir, 
                             k_values=[1, 3, 5, 10], 
                             selected_metrics=None):
    """
    Run the complete evaluation pipeline and return summary metrics.
    
    Args:
        gumroad_dir: Directory containing processed Gumroad results
        endpoint_dir: Directory containing endpoint results
        expert_dir: Directory containing expert rankings
        output_dir: Directory to save evaluation results
        k_values: List of k values to calculate metrics for
        selected_metrics: List of metrics to display/calculate. 
                          If None, calculate all metrics.
                          Supported values:
                          - 'ndcg': Normalized Discounted Cumulative Gain
                          - 'precision': Precision@k
                          - 'weighted_precision': Weighted Precision@k
                          - 'recall': Recall@k
                          - 'success_rate': Success Rate@k
                          - 'rank_info': Rank-related metrics (MRR, First Relevant Position, Avg Relevant Rank)
                          - 'coverage': Relevant Coverage@k
                          - 'map': Mean Average Precision
    
    Returns:
        Dictionary containing summary metrics
    """
    # If no metrics selected, calculate all metrics
    if selected_metrics is None:
        selected_metrics = [
            'ndcg', 'precision', 'weighted_precision', 'recall', 
            'success_rate', 'rank_info', 'coverage', 'map'
        ]

    os.makedirs(output_dir, exist_ok=True)

    # Initialize metrics dictionary based on selected metrics
    metrics_to_calculate = {
        'ndcg': 'ndcg' in selected_metrics,
        'precision': 'precision' in selected_metrics,
        'weighted_precision': 'weighted_precision' in selected_metrics,
        'recall': 'recall' in selected_metrics,
        'success_rate': 'success_rate' in selected_metrics,
        'mrr': 'rank_info' in selected_metrics,
        'first_relevant_position': 'rank_info' in selected_metrics,
        'avg_relevant_rank': 'rank_info' in selected_metrics,
        'map': 'map' in selected_metrics,
        'relevant_coverage': 'coverage' in selected_metrics
    }

    # Final metrics to store with safe initialization
    def safe_list_for_k(condition):
        return {k: [] for k in k_values} if condition else {}
    
    def safe_list(condition):
        return [] if condition else None

    # Initialize all metrics with safe method
    gumroad_ndcg = safe_list_for_k(metrics_to_calculate['ndcg'])
    endpoint_ndcg = safe_list_for_k(metrics_to_calculate['ndcg'])
    gumroad_precision = safe_list_for_k(metrics_to_calculate['precision'])
    endpoint_precision = safe_list_for_k(metrics_to_calculate['precision'])
    gumroad_weighted_precision = safe_list_for_k(metrics_to_calculate['weighted_precision'])
    endpoint_weighted_precision = safe_list_for_k(metrics_to_calculate['weighted_precision'])
    gumroad_recall = safe_list_for_k(metrics_to_calculate['recall'])
    endpoint_recall = safe_list_for_k(metrics_to_calculate['recall'])
    
    # Rank-related metrics
    gumroad_mrr = safe_list(metrics_to_calculate['mrr'])
    endpoint_mrr = safe_list(metrics_to_calculate['mrr'])
    gumroad_first_relevant_position = safe_list(metrics_to_calculate['first_relevant_position'])
    endpoint_first_relevant_position = safe_list(metrics_to_calculate['first_relevant_position'])
    gumroad_avg_relevant_rank = safe_list(metrics_to_calculate['avg_relevant_rank'])
    endpoint_avg_relevant_rank = safe_list(metrics_to_calculate['avg_relevant_rank'])
    
    # Additional metrics
    gumroad_success_rate = safe_list_for_k(metrics_to_calculate['success_rate'])
    endpoint_success_rate = safe_list_for_k(metrics_to_calculate['success_rate'])
    gumroad_relevant_coverage = safe_list_for_k(metrics_to_calculate['relevant_coverage'])
    endpoint_relevant_coverage = safe_list_for_k(metrics_to_calculate['relevant_coverage'])
    
    # MAP metrics
    gumroad_ap = safe_list(metrics_to_calculate['map'])
    endpoint_ap = safe_list(metrics_to_calculate['map'])

    # Rest of the function remains largely the same as the original script

    os.makedirs(output_dir, exist_ok=True)

    # Load expert rankings
    print("Loading expert rankings...")
    expert_rankings = {}
    for file in os.listdir(expert_dir):
        if file.endswith('_expert.json'):
            try:
                with open(os.path.join(expert_dir, file)) as f:
                    data = json.load(f)
                    query = data['query']
                    ranked_product_ids = data['ranked_product_ids']
                    expert_rankings[query] = ranked_product_ids
                    print(f"  Loaded expert ranking for query '{query}' with {len(ranked_product_ids)} products")
            except Exception as e:
                print(f"  Error loading expert ranking file {file}: {e}")

    # Load Gumroad results
    print("\nLoading Gumroad results...")
    gumroad_results = {}
    for file in os.listdir(gumroad_dir):
        if file.endswith('_gumroad.json'):
            try:
                with open(os.path.join(gumroad_dir, file)) as f:
                    data = json.load(f)
                    query = data['query']
                    products = data['products']
                    if query in expert_rankings:
                        gumroad_results[query] = products
                        print(f"  Loaded Gumroad results for query '{query}' with {len(products)} products")
            except Exception as e:
                print(f"  Error loading Gumroad file {file}: {e}")

    # Load endpoint results
    print("\nLoading endpoint results...")
    endpoint_results = {}
    for file in os.listdir(endpoint_dir):
        if file.endswith('_endpoint.json'):
            try:
                with open(os.path.join(endpoint_dir, file)) as f:
                    data = json.load(f)
                    query = data['query']
                    results = data['results']['results']
                    if query in expert_rankings:
                        endpoint_results[query] = results
                        print(f"  Loaded endpoint results for query '{query}' with {len(results)} products")
            except Exception as e:
                print(f"  Error loading endpoint file {file}: {e}")

    print(f"\nFound {len(expert_rankings)} expert rankings")
    print(f"Found {len(gumroad_results)} Gumroad results")
    print(f"Found {len(endpoint_results)} endpoint results")

    # Process each query
    processed_queries = []
    print("\nCalculating metrics for each query:")
    queries_list = list(expert_rankings.keys())
    print("Processing queries: " + ", ".join(["'" + q + "'" for q in sorted(queries_list)]))

    for query, expert_ids in expert_rankings.items():
        # Skip if missing data
        if query not in gumroad_results or query not in endpoint_results:
            print(f"  Skipping query '{query}' due to missing data")
            continue
        
        processed_queries.append(query)
        
        # Get product IDs from results
        gumroad_ids = [p.get('id', '') for p in gumroad_results[query]]
        endpoint_ids = [p.get('id', '') for p in endpoint_results[query]]
        
        # Create relevance scores (higher rank = higher relevance)
        # Use a more nuanced graded relevance - exponentially decreasing
        max_relevance = len(expert_ids)
        relevance_scores = {}
        for i, pid in enumerate(expert_ids):
            # Calculate relevance score using an exponential decay function
            # This gives much higher weight to top expert-ranked items
            position_weight = np.exp(-0.1 * i)  # Exponential decay with position
            relevance_scores[pid] = position_weight
            
        # Normalize relevance scores to [0, 1] range
        if relevance_scores:
            max_score = max(relevance_scores.values())
            min_score = min(relevance_scores.values())
            range_score = max_score - min_score
            if range_score > 0:
                for pid in relevance_scores:
                    relevance_scores[pid] = (relevance_scores[pid] - min_score) / range_score
        
        # All expert IDs are considered relevant with varying degrees of relevance
        relevant_items = set(expert_ids)
        
        # Conditional metric calculations
        # MRR Calculation
        if metrics_to_calculate['mrr']:
            # Find first relevant result in Gumroad
            gumroad_mrr_score = 0
            gumroad_first_pos = -1  # -1 indicates no relevant result found
            for i, pid in enumerate(gumroad_ids):
                if pid in expert_ids:
                    gumroad_mrr_score = 1.0 / (i + 1)
                    gumroad_first_pos = i
                    break
            gumroad_mrr.append(gumroad_mrr_score)
            gumroad_first_relevant_position.append(gumroad_first_pos if gumroad_first_pos >= 0 else float('inf'))
            
            # Find first relevant result in endpoint
            endpoint_mrr_score = 0
            endpoint_first_pos = -1  # -1 indicates no relevant result found
            for i, pid in enumerate(endpoint_ids):
                if pid in expert_ids:
                    endpoint_mrr_score = 1.0 / (i + 1)
                    endpoint_first_pos = i
                    break
            endpoint_mrr.append(endpoint_mrr_score)
            endpoint_first_relevant_position.append(endpoint_first_pos if endpoint_first_pos >= 0 else float('inf'))
        
        # Average Relevant Rank Calculation
        if metrics_to_calculate['avg_relevant_rank']:
            # For Gumroad
            gumroad_relevant_positions = [i for i, pid in enumerate(gumroad_ids) if pid in expert_ids]
            if gumroad_relevant_positions:
                gumroad_avg_relevant_rank.append(np.mean(gumroad_relevant_positions))
            else:
                gumroad_avg_relevant_rank.append(float('inf'))
                
            # For Endpoint
            endpoint_relevant_positions = [i for i, pid in enumerate(endpoint_ids) if pid in expert_ids]
            if endpoint_relevant_positions:
                endpoint_avg_relevant_rank.append(np.mean(endpoint_relevant_positions))
            else:
                endpoint_avg_relevant_rank.append(float('inf'))
        
        # MAP Calculation
        if metrics_to_calculate['map']:
            # Gumroad AP
            gumroad_ap_score = calculate_average_precision(gumroad_ids, expert_ids)
            gumroad_ap.append(gumroad_ap_score)
            
            # Endpoint AP
            endpoint_ap_score = calculate_average_precision(endpoint_ids, expert_ids)
            endpoint_ap.append(endpoint_ap_score)
        
        # For each k value - metrics that depend on k
        for k in k_values:
            # Skip if k is larger than available expert rankings
            if k > len(expert_ids):
                continue
            
            # Create a utility method to handle k-dependent metrics
            def process_k_metrics(ids, op_name, k):
                # Create relevance vectors - using graded relevance
                rel = np.array([relevance_scores.get(pid, 0) for pid in ids[:k]])
                
                # Metric calculation methods
                def dcg(rel_vector, k):
                    rel = rel_vector[:k]
                    if len(rel) == 0:
                        return 0.0
                    # Use the 2^relevance - 1 formula for DCG, which gives more weight to highly relevant items
                    return np.sum((2**rel - 1) / np.log2(np.arange(2, len(rel) + 2)))
                
                # Metrics
                metrics_results = {}
                
                # nDCG Calculation
                if metrics_to_calculate['ndcg']:
                    # Ideal DCG - using the sorted relevance scores
                    ideal_rel = sorted([relevance_scores.get(pid, 0) for pid in expert_ids[:k]], reverse=True)
                    ideal_dcg = dcg(np.array(ideal_rel), k)
                    
                    # Calculate nDCG
                    ndcg_score = dcg(rel, k) / ideal_dcg if ideal_dcg > 0 else 0
                    metrics_results['ndcg'] = ndcg_score
                
                # Precision Calculation
                if metrics_to_calculate['precision']:
                    relevant = len(set(ids[:k]) & relevant_items)
                    retrieved = min(k, len(ids))
                    precision = relevant / retrieved if retrieved > 0 else 0
                    metrics_results['precision'] = precision
                
                # Weighted Precision Calculation
                if metrics_to_calculate['weighted_precision']:
                    weighted_prec = sum(relevance_scores.get(pid, 0) for pid in ids[:k]) / k if k > 0 else 0
                    metrics_results['weighted_precision'] = weighted_prec
                
                # Recall Calculation
                if metrics_to_calculate['recall']:
                    relevant = len(set(ids[:k]) & relevant_items)
                    total_relevant = len(relevant_items)
                    recall = relevant / total_relevant if total_relevant > 0 else 0
                    metrics_results['recall'] = recall
                
                # Success Rate Calculation
                if metrics_to_calculate['success_rate']:
                    success = 1 if any(pid in expert_ids for pid in ids[:k]) else 0
                    metrics_results['success_rate'] = success
                
                # Relevant Coverage Calculation
                if metrics_to_calculate['relevant_coverage']:
                    coverage = len(set(ids[:k]) & relevant_items) / len(relevant_items) if relevant_items else 0
                    metrics_results['relevant_coverage'] = coverage
                
                return metrics_results
            
            # Calculate metrics for Gumroad and Endpoint
            gumroad_metrics = process_k_metrics(gumroad_ids, 'Gumroad', k)
            endpoint_metrics = process_k_metrics(endpoint_ids, 'Endpoint', k)
            
            # Store metrics for each k
            if metrics_to_calculate['ndcg']:
                gumroad_ndcg[k].append(gumroad_metrics['ndcg'])
                endpoint_ndcg[k].append(endpoint_metrics['ndcg'])
            
            if metrics_to_calculate['precision']:
                gumroad_precision[k].append(gumroad_metrics['precision'])
                endpoint_precision[k].append(endpoint_metrics['precision'])
            
            if metrics_to_calculate['weighted_precision']:
                gumroad_weighted_precision[k].append(gumroad_metrics['weighted_precision'])
                endpoint_weighted_precision[k].append(endpoint_metrics['weighted_precision'])
            
            if metrics_to_calculate['recall']:
                gumroad_recall[k].append(gumroad_metrics['recall'])
                endpoint_recall[k].append(endpoint_metrics['recall'])
            
            if metrics_to_calculate['success_rate']:
                gumroad_success_rate[k].append(gumroad_metrics['success_rate'])
                endpoint_success_rate[k].append(endpoint_metrics['success_rate'])
            
            if metrics_to_calculate['relevant_coverage']:
                gumroad_relevant_coverage[k].append(gumroad_metrics['relevant_coverage'])
                endpoint_relevant_coverage[k].append(endpoint_metrics['relevant_coverage'])

    # Calculate average metrics
    print("\nCalculating average metrics...")
    
    # Create a utility function to calculate averages safely
    def safe_mean(metric_list):
        return np.mean(metric_list) if metric_list else None
    
    def safe_k_means(metric_dict):
        return {k: safe_mean(metric_dict[k]) for k in metric_dict} if metric_dict else {}

    # Compute average metrics only for calculated metrics
    avg_gumroad_ndcg = safe_k_means(gumroad_ndcg) if metrics_to_calculate['ndcg'] else None
    avg_endpoint_ndcg = safe_k_means(endpoint_ndcg) if metrics_to_calculate['ndcg'] else None
    
    avg_gumroad_precision = safe_k_means(gumroad_precision) if metrics_to_calculate['precision'] else None
    avg_endpoint_precision = safe_k_means(endpoint_precision) if metrics_to_calculate['precision'] else None
    
    avg_gumroad_weighted_precision = safe_k_means(gumroad_weighted_precision) if metrics_to_calculate['weighted_precision'] else None
    avg_endpoint_weighted_precision = safe_k_means(endpoint_weighted_precision) if metrics_to_calculate['weighted_precision'] else None
    
    avg_gumroad_recall = safe_k_means(gumroad_recall) if metrics_to_calculate['recall'] else None
    avg_endpoint_recall = safe_k_means(endpoint_recall) if metrics_to_calculate['recall'] else None
    
    avg_gumroad_success_rate = safe_k_means(gumroad_success_rate) if metrics_to_calculate['success_rate'] else None
    avg_endpoint_success_rate = safe_k_means(endpoint_success_rate) if metrics_to_calculate['success_rate'] else None
    
    avg_gumroad_relevant_coverage = safe_k_means(gumroad_relevant_coverage) if metrics_to_calculate['relevant_coverage'] else None
    avg_endpoint_relevant_coverage = safe_k_means(endpoint_relevant_coverage) if metrics_to_calculate['relevant_coverage'] else None
    
    avg_gumroad_mrr = safe_mean(gumroad_mrr) if metrics_to_calculate['mrr'] else None
    avg_endpoint_mrr = safe_mean(endpoint_mrr) if metrics_to_calculate['mrr'] else None
    
    avg_gumroad_map = safe_mean(gumroad_ap) if metrics_to_calculate['map'] else None
    avg_endpoint_map = safe_mean(endpoint_ap) if metrics_to_calculate['map'] else None
    
    # Calculate avg first relevant position and avg relevant rank
    def filter_and_mean(metric_list):
        if not metric_list:
            return None
        filtered = [x for x in metric_list if x != float('inf')]
        return np.mean(filtered) if filtered else float('inf')
    
    avg_gumroad_first_relevant_position = filter_and_mean(gumroad_first_relevant_position) if metrics_to_calculate['first_relevant_position'] else None
    avg_endpoint_first_relevant_position = filter_and_mean(endpoint_first_relevant_position) if metrics_to_calculate['first_relevant_position'] else None
    
    avg_gumroad_avg_relevant_rank = filter_and_mean(gumroad_avg_relevant_rank) if metrics_to_calculate['avg_relevant_rank'] else None
    avg_endpoint_avg_relevant_rank = filter_and_mean(endpoint_avg_relevant_rank) if metrics_to_calculate['avg_relevant_rank'] else None

    # Print results with conditional printing
    print("\n" + "="*50)
    print("Evaluation Results")
    print("="*50)
    print(f"Processed {len(processed_queries)} out of {len(expert_rankings)} queries")

    # Utility function to print metrics safely
    def print_metric_group(metric_name, avg_gumroad, avg_endpoint):
        if avg_gumroad is not None and avg_endpoint is not None:
            print(f"\n{metric_name} Metrics:")
            if isinstance(avg_gumroad, dict):
                for k in avg_gumroad.keys():
                    print(f"  k={k}: Gumroad: {avg_gumroad[k]:.4f}, "
                          f"Custom Endpoint: {avg_endpoint[k]:.4f}, "
                          f"Improvement: {(avg_endpoint[k] - avg_gumroad[k]) * 100:.2f}%")
            else:
                print(f"  Gumroad: {avg_gumroad:.4f}, "
                      f"Custom Endpoint: {avg_endpoint:.4f}, "
                      f"Improvement: {(avg_endpoint - avg_gumroad) * 100:.2f}%")

    # Conditionally print metrics based on calculation
    print_metric_group("nDCG@k", avg_gumroad_ndcg, avg_endpoint_ndcg)
    print_metric_group("Precision@k", avg_gumroad_precision, avg_endpoint_precision)
    print_metric_group("Weighted Precision@k", avg_gumroad_weighted_precision, avg_endpoint_weighted_precision)
    print_metric_group("Recall@k", avg_gumroad_recall, avg_endpoint_recall)
    print_metric_group("Mean Reciprocal Rank (MRR)", avg_gumroad_mrr, avg_endpoint_mrr)
    print_metric_group("Mean Average Precision (MAP)", avg_gumroad_map, avg_endpoint_map)
    print_metric_group("Success Rate@k", avg_gumroad_success_rate, avg_endpoint_success_rate)
    print_metric_group("Relevant Coverage@k", avg_gumroad_relevant_coverage, avg_endpoint_relevant_coverage)
    
    # Rank-related metrics
    if avg_gumroad_first_relevant_position is not None and avg_endpoint_first_relevant_position is not None:
        print("\nAverage Position of First Relevant Result:")
        print(f"  Gumroad: {avg_gumroad_first_relevant_position:.2f}, "
              f"Custom Endpoint: {avg_endpoint_first_relevant_position:.2f}, "
              f"Improvement: {avg_gumroad_first_relevant_position - avg_endpoint_first_relevant_position:.2f} positions")
    
    if avg_gumroad_avg_relevant_rank is not None and avg_endpoint_avg_relevant_rank is not None:
        print("\nAverage Rank of Relevant Results:")
        print(f"  Gumroad: {avg_gumroad_avg_relevant_rank:.2f}, "
              f"Custom Endpoint: {avg_endpoint_avg_relevant_rank:.2f}, "
              f"Improvement: {avg_gumroad_avg_relevant_rank - avg_endpoint_avg_relevant_rank:.2f} positions")


    # Visualization Generation
def generate_visualizations(metrics_summary, output_dir, selected_metrics):
    """
    Generate visualizations for the selected metrics
    
    Args:
        metrics_summary (dict): Dictionary of calculated metrics
        output_dir (str): Directory to save output files
        selected_metrics (list): List of metrics to visualize
    """
    import seaborn as sns
    import matplotlib.pyplot as plt
    import pandas as pd
    import os
    
    # Set theme for visualizations
    sns.set_theme(style="whitegrid")
    
    # Create a directory for visualizations if it doesn't exist
    viz_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)
    
    # Prepare visualization data for k-dependent metrics
    def prepare_k_metric_dataframe(metric_dict):
        """
        Prepare a dataframe for k-dependent metrics for visualization
        
        Args:
            metric_dict (dict): Dictionary of metrics for different k values
        
        Returns:
            pandas.DataFrame: Formatted dataframe for visualization
        """
        if not metric_dict:
            return None
        
        df_data = []
        for k, values in metric_dict.items():
            df_data.append({
                'k': int(k),
                'Search Engine': 'Gumroad',
                'Value': values['gumroad']
            })
            df_data.append({
                'k': int(k),
                'Search Engine': 'Custom Endpoint',
                'Value': values['endpoint']
            })
        
        return pd.DataFrame(df_data)
    
    # Create comprehensive dashboard of selected metrics
    def create_metric_plots(metrics_to_plot):
        """
        Create plots for selected metrics
        
        Args:
            metrics_to_plot (list): List of metrics to visualize
        """
        # Categorize metrics
        k_dependent_metrics = [
            m for m in metrics_to_plot 
            if m in ['ndcg', 'precision', 'weighted_precision', 'recall', 'success_rate', 'relevant_coverage']
        ]
        single_value_metrics = [m for m in metrics_to_plot if m in ['mrr', 'map']]
        
        # Calculate total number of plots
        num_metrics = len(k_dependent_metrics) + len(single_value_metrics)
        rows = (num_metrics + 1) // 2  # Ceiling division
        
        # Create figure with subplots
        fig, axes = plt.subplots(rows, 2, figsize=(16, 5*rows))
        fig.suptitle('Evaluation Metrics Comparison', fontsize=16)
        
        # Flatten axes for easier indexing
        axes = axes.flatten() if rows > 1 else axes
        
        # Plot k-dependent metrics
        for i, metric in enumerate(k_dependent_metrics):
            df_metric = prepare_k_metric_dataframe(metrics_summary.get(metric, {}))
            
            if df_metric is not None:
                ax = axes[i] if num_metrics > 1 else axes
                sns.lineplot(data=df_metric, x='k', y='Value', 
                             hue='Search Engine', marker='o', ax=ax)
                ax.set_title(f'{metric.replace("_", " ").title()}@k Comparison', fontsize=12)
                ax.set_xlabel('k')
                ax.set_ylabel(f'{metric.replace("_", " ").title()}@k')
        
        # Plot single-value metrics
        offset = len(k_dependent_metrics)
        for i, metric in enumerate(single_value_metrics):
            engines = ['Gumroad', 'Custom Endpoint']
            values = [
                metrics_summary.get(metric, {}).get('gumroad', 0), 
                metrics_summary.get(metric, {}).get('endpoint', 0)
            ]
            
            ax = axes[offset + i] if num_metrics > 1 else axes
            sns.barplot(x=engines, y=values, ax=ax)
            ax.set_title(f'{metric.upper()} Comparison', fontsize=12)
            ax.set_ylabel(metric.upper())
        
        # Remove any unused subplots
        if num_metrics < len(axes):
            for j in range(num_metrics, len(axes)):
                fig.delaxes(axes[j])
        
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'metrics_dashboard.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # Additional individual metric plots
    def create_individual_metric_plots():
        """
        Create individual plots for each metric
        """
        for metric in selected_metrics:
            plt.figure(figsize=(10, 6))
            
            if metric in ['ndcg', 'precision', 'weighted_precision', 'recall', 'success_rate', 'relevant_coverage']:
                df_metric = prepare_k_metric_dataframe(metrics_summary.get(metric, {}))
                
                if df_metric is not None:
                    sns.lineplot(data=df_metric, x='k', y='Value', 
                                 hue='Search Engine', marker='o')
                    plt.title(f'{metric.replace("_", " ").title()} Comparison', fontsize=16)
                    plt.xlabel('k', fontsize=14)
                    plt.ylabel(f'{metric.replace("_", " ").title()}@k', fontsize=14)
                    plt.grid(True, alpha=0.3)
                    plt.savefig(os.path.join(viz_dir, f'{metric}_comparison.png'), dpi=300, bbox_inches='tight')
                    plt.close()
            
            elif metric in ['mrr', 'map']:
                engines = ['Gumroad', 'Custom Endpoint']
                values = [
                    metrics_summary.get(metric, {}).get('gumroad', 0), 
                    metrics_summary.get(metric, {}).get('endpoint', 0)
                ]
                
                plt.figure(figsize=(8, 6))
                sns.barplot(x=engines, y=values)
                plt.title(f'{metric.upper()} Comparison', fontsize=16)
                plt.xlabel('Search Engine', fontsize=14)
                plt.ylabel(metric.upper(), fontsize=14)
                plt.grid(True, alpha=0.3, axis='y')
                plt.savefig(os.path.join(viz_dir, f'{metric}_comparison.png'), dpi=300, bbox_inches='tight')
                plt.close()
    
    # Generate visualizations
    if metrics_summary:
        print("\nGenerating visualizations...")
        create_metric_plots(selected_metrics)
        create_individual_metric_plots()
        print(f"\nVisualization complete! Graphs saved in {viz_dir}")

    # Prepare return dictionary with calculated metrics
    metrics_summary = {
        "processed_queries": len(processed_queries),
        "total_queries": len(expert_rankings)
    }
    
    # Add only calculated metrics to summary
    if avg_gumroad_ndcg: metrics_summary['ndcg'] = {str(k): {"gumroad": v, "endpoint": avg_endpoint_ndcg[k]} for k, v in avg_gumroad_ndcg.items()}
    if avg_gumroad_precision: metrics_summary['precision'] = {str(k): {"gumroad": v, "endpoint": avg_endpoint_precision[k]} for k, v in avg_gumroad_precision.items()}
    if avg_gumroad_weighted_precision: metrics_summary['weighted_precision'] = {str(k): {"gumroad": v, "endpoint": avg_endpoint_weighted_precision[k]} for k, v in avg_gumroad_weighted_precision.items()}
    if avg_gumroad_recall: metrics_summary['recall'] = {str(k): {"gumroad": v, "endpoint": avg_endpoint_recall[k]} for k, v in avg_gumroad_recall.items()}
    if avg_gumroad_mrr: metrics_summary['mrr'] = {"gumroad": avg_gumroad_mrr, "endpoint": avg_endpoint_mrr}
    if avg_gumroad_map: metrics_summary['map'] = {"gumroad": avg_gumroad_map, "endpoint": avg_endpoint_map}
    if avg_gumroad_success_rate: metrics_summary['success_rate'] = {str(k): {"gumroad": v, "endpoint": avg_endpoint_success_rate[k]} for k, v in avg_gumroad_success_rate.items()}
    if avg_gumroad_relevant_coverage: metrics_summary['relevant_coverage'] = {str(k): {"gumroad": v, "endpoint": avg_endpoint_relevant_coverage[k]} for k, v in avg_gumroad_relevant_coverage.items()}
    
    # Optional additional metrics
    if avg_gumroad_first_relevant_position is not None:
        metrics_summary['first_relevant_position'] = {
            "gumroad": avg_gumroad_first_relevant_position, 
            "endpoint": avg_endpoint_first_relevant_position
        }
    if avg_gumroad_avg_relevant_rank is not None:
        metrics_summary['avg_relevant_rank'] = {
            "gumroad": avg_gumroad_avg_relevant_rank, 
            "endpoint": avg_endpoint_avg_relevant_rank
        }


   # Call visualization generation if metrics were calculated
    if metrics_summary:
        generate_visualizations(metrics_summary, output_dir, selected_metrics)

    return metrics_summary

def calculate_average_precision(result_ids, expert_ids):
    """
    Calculate Average Precision (AP) for a single query.
    AP is the average of precision values at each relevant result position.
    
    Args:
        result_ids: List of result IDs from the search engine
        expert_ids: List of expert-ranked relevant IDs
        
    Returns:
        Average Precision value
    """
    relevant_set = set(expert_ids)
    if not relevant_set:
        return 0.0
    
    precision_values = []
    relevant_count = 0
    
    for i, result_id in enumerate(result_ids):
        position = i + 1  # 1-indexed position
        if result_id in relevant_set:
            relevant_count += 1
            precision_at_k = relevant_count / position
            precision_values.append(precision_at_k)
    
    if not precision_values:
        return 0.0
    
    return sum(precision_values) / len(relevant_set)  # Divide by total number of relevant items

if __name__ == "__main__":
    # Configuration
    base_dir = "search_eval_20250324_003120"
    gumroad_dir = os.path.join(base_dir, "gumroad_processed")
    endpoint_dir = os.path.join(base_dir, "endpoint_results")
    expert_dir = os.path.join(base_dir, "expert_rankings")
    output_dir = os.path.join(base_dir, "evaluation_results")
    
    # Set k values
    k_values = [1, 3, 5, 10]
    
    # Example of selecting specific metrics
    selected_metrics = ['ndcg', 'weighted_precision', 'rank_info']
    
    # Run the evaluation pipeline
    run_evaluation_pipeline(
        gumroad_dir, 
        endpoint_dir, 
        expert_dir, 
        output_dir, 
        k_values, 
        selected_metrics
    )