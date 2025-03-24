#!/usr/bin/env python3
import os
import sys
import argparse
import logging
from datetime import datetime

# Add the script directory to the Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("evaluation_pipeline.log"),
                        logging.StreamHandler(sys.stdout)
                    ])
logger = logging.getLogger(__name__)

# The file names use hyphens (-) rather than underscores (_)
try:
    # Use importlib to handle hyphenated file names
    import importlib.util
    
    # Function to import a module by file path
    def import_module_from_file(file_path, module_name):
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    
    # Import all required modules
    process_gumroad_data = import_module_from_file(
        os.path.join(script_dir, "process_gumroad_data.py"), "process_gumroad_data")

    query_search_endpoint = import_module_from_file(
        os.path.join(script_dir, "query_search_endpoint.py"), "query_search_endpoint")

    generate_expert_rankings = import_module_from_file(
        os.path.join(script_dir, "generate_expert_rankings.py"), "generate_expert_rankings")

    calculate_metrics = import_module_from_file(
        os.path.join(script_dir, "calculate_metrics.py"), "calculate_metrics")
    
    # Extract the functions we need
    get_gumroad_results = process_gumroad_data.get_gumroad_results
    save_gumroad_results = process_gumroad_data.save_gumroad_results
    process_queries_from_file = query_search_endpoint.process_queries_from_file
    process_gumroad_endpoint_pairs = generate_expert_rankings.process_gumroad_endpoint_pairs
    run_evaluation_pipeline = calculate_metrics.run_evaluation_pipeline
    
    logger.info("Successfully imported modules using dynamic import")
    
except Exception as e:
    logger.error(f"Failed to import modules: {e}")
    logger.error("Please check file names and make sure required modules are available")
    sys.exit(1)

def create_directory(dir_path):
    """Create directory if it doesn't exist."""
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        logger.info(f"Created directory: {dir_path}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate search engine performance")
    parser.add_argument("--gumroad-data", default="./gumroad_data", 
                        help="Directory containing raw Gumroad data")
    parser.add_argument("--endpoint-url", default="http://localhost:8000/two_phase_optimized",
                        help="URL of the search endpoint to evaluate")
    parser.add_argument("--api-key", default=os.environ.get("ANTHROPIC_API_KEY"),
                        help="Anthropic API key for expert rankings")
    parser.add_argument("--skip-gumroad", action="store_true",
                        help="Skip Gumroad data processing step")
    parser.add_argument("--skip-endpoint", action="store_true",
                        help="Skip endpoint querying step")
    parser.add_argument("--skip-expert", action="store_true",
                        help="Skip expert ranking generation step")
    parser.add_argument("--skip-metrics", action="store_true",
                        help="Skip metrics calculation step")
    parser.add_argument("--max-queries", type=int, default=50,
                        help="Maximum number of queries to process")
    parser.add_argument("--k-values", type=int, nargs='+', default=[5, 10, 20, 50],
                        help="k values to calculate metrics at")
    parser.add_argument("--continue", dest="continue_run", action="store_true",
                        help="Continue from a previous run")
    parser.add_argument("--output-dir", 
                        help="Output directory (defaults to timestamped directory, required with --continue)")
    parser.add_argument("--standard-metrics", action="store_true",
                        help="Use standard metrics (precision, recall, etc.) instead of focused metrics")
    
    args = parser.parse_args()
    
    # Set up directories
    if args.continue_run:
        if not args.output_dir:
            logger.error("--output-dir is required when using --continue")
            sys.exit(1)
        base_dir = args.output_dir
        if not os.path.exists(base_dir):
            logger.error(f"Output directory {base_dir} does not exist")
            sys.exit(1)
        logger.info(f"Continuing previous run in {base_dir}")
    else:
        if args.output_dir:
            base_dir = args.output_dir
            if os.path.exists(base_dir):
                logger.warning(f"Output directory {base_dir} already exists, files may be overwritten")
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_dir = f"search_eval_{timestamp}"
            
    gumroad_processed_dir = os.path.join(base_dir, "gumroad_processed")
    endpoint_results_dir = os.path.join(base_dir, "endpoint_results")
    expert_rankings_dir = os.path.join(base_dir, "expert_rankings")
    evaluation_results_dir = os.path.join(base_dir, "evaluation_results")
    
    # Create directories if they don't exist
    create_directory(base_dir)
    create_directory(gumroad_processed_dir)
    create_directory(endpoint_results_dir)
    create_directory(expert_rankings_dir)
    create_directory(evaluation_results_dir)
    
    # Set Anthropic API key if provided
    if args.api_key:
        os.environ["ANTHROPIC_API_KEY"] = args.api_key
        logger.info("Using provided Anthropic API key")
    elif os.environ.get("ANTHROPIC_API_KEY"):
        logger.info("Using Anthropic API key from environment")
    else:
        logger.warning("No Anthropic API key provided. Expert ranking step will fail.")
    
    # Step 1: Process Gumroad data
    if not args.skip_gumroad:
        logger.info("Step 1: Processing Gumroad data...")
        gumroad_results = get_gumroad_results(args.gumroad_data, args.max_queries)
        save_gumroad_results(gumroad_results, gumroad_processed_dir)
    else:
        logger.info("Skipping Gumroad data processing step")
    
    # Step 2: Query the search endpoint
    if not args.skip_endpoint:
        logger.info("Step 2: Querying search endpoint...")
        process_queries_from_file(gumroad_processed_dir, args.endpoint_url, endpoint_results_dir)
    else:
        logger.info("Skipping endpoint querying step")
    
    # Step 3: Generate expert rankings
    if not args.skip_expert:
        logger.info("Step 3: Generating expert rankings...")
        process_gumroad_endpoint_pairs(gumroad_processed_dir, endpoint_results_dir, expert_rankings_dir)
    else:
        logger.info("Skipping expert ranking generation step")
    
    # Step 4: Calculate evaluation metrics
    if not args.skip_metrics:
        logger.info("Step 4: Calculating evaluation metrics...")
        
        # Determine whether to use standard metrics or focused metrics (default is focused)
        use_standard_metrics = args.standard_metrics
        
        if use_standard_metrics:
            logger.info("Using standard metrics (including precision, recall, etc.)")
        else:
            logger.info("Using focused metrics (NDCG, weighted precision, position metrics)")
        
        # Run the evaluation pipeline
        metrics_summary = run_evaluation_pipeline(
            gumroad_processed_dir, 
            endpoint_results_dir, 
            expert_rankings_dir, 
            evaluation_results_dir,
            args.k_values,
            use_standard_metrics=use_standard_metrics
        )
        
        # Print summary
        logger.info("\n" + "="*50)
        logger.info("Evaluation Pipeline Complete")
        logger.info("="*50)
        logger.info(f"Processed {metrics_summary['processed_queries']} queries")
        logger.info("\nSummary of Metrics:")
        
        logger.info("\nnDCG@k Metrics:")
        for k in metrics_summary['ndcg']:
            logger.info(f"  k={k}: Gumroad: {metrics_summary['ndcg'][k]['gumroad']:.4f}, "
                      f"Custom Endpoint: {metrics_summary['ndcg'][k]['endpoint']:.4f}, "
                      f"Improvement: {(metrics_summary['ndcg'][k]['endpoint'] - metrics_summary['ndcg'][k]['gumroad']) * 100:.2f}%")
        
        logger.info("\nWeighted Precision@k Metrics:")
        for k in metrics_summary['weighted_precision']:
            logger.info(f"  k={k}: Gumroad: {metrics_summary['weighted_precision'][k]['gumroad']:.4f}, "
                      f"Custom Endpoint: {metrics_summary['weighted_precision'][k]['endpoint']:.4f}, "
                      f"Improvement: {(metrics_summary['weighted_precision'][k]['endpoint'] - metrics_summary['weighted_precision'][k]['gumroad']) * 100:.2f}%")
        
        # Position metrics
        if metrics_summary.get('first_relevant_position'):
            if metrics_summary['first_relevant_position']['gumroad'] != "N/A":
                logger.info("\nAverage Position of First Relevant Result:")
                logger.info(f"  Gumroad: {float(metrics_summary['first_relevant_position']['gumroad']):.2f}, "
                          f"Custom Endpoint: {float(metrics_summary['first_relevant_position']['endpoint']):.2f}, "
                          f"Improvement: {float(metrics_summary['first_relevant_position']['improvement']):.2f} positions")
        
        if metrics_summary.get('avg_relevant_rank'):
            if metrics_summary['avg_relevant_rank']['gumroad'] != "N/A":
                logger.info("\nAverage Rank of Relevant Results:")
                logger.info(f"  Gumroad: {float(metrics_summary['avg_relevant_rank']['gumroad']):.2f}, "
                          f"Custom Endpoint: {float(metrics_summary['avg_relevant_rank']['endpoint']):.2f}, "
                          f"Improvement: {float(metrics_summary['avg_relevant_rank']['improvement']):.2f} positions")
        
        logger.info("\nMean Reciprocal Rank (MRR):")
        logger.info(f"  Gumroad: {metrics_summary['mrr']['gumroad']:.4f}, "
                  f"Custom Endpoint: {metrics_summary['mrr']['endpoint']:.4f}, "
                  f"Improvement: {(metrics_summary['mrr']['endpoint'] - metrics_summary['mrr']['gumroad']) * 100:.2f}%")
        
        # Print standard metrics if they were calculated
        if use_standard_metrics:
            if 'precision' in metrics_summary:
                logger.info("\nPrecision@k Metrics:")
                for k in metrics_summary['precision']:
                    logger.info(f"  k={k}: Gumroad: {metrics_summary['precision'][k]['gumroad']:.4f}, "
                              f"Custom Endpoint: {metrics_summary['precision'][k]['endpoint']:.4f}, "
                              f"Improvement: {(metrics_summary['precision'][k]['endpoint'] - metrics_summary['precision'][k]['gumroad']) * 100:.2f}%")
            
            if 'recall' in metrics_summary:
                logger.info("\nRecall@k Metrics:")
                for k in metrics_summary['recall']:
                    logger.info(f"  k={k}: Gumroad: {metrics_summary['recall'][k]['gumroad']:.4f}, "
                              f"Custom Endpoint: {metrics_summary['recall'][k]['endpoint']:.4f}, "
                              f"Improvement: {(metrics_summary['recall'][k]['endpoint'] - metrics_summary['recall'][k]['gumroad']) * 100:.2f}%")
            
            if 'map' in metrics_summary:
                logger.info("\nMean Average Precision (MAP):")
                logger.info(f"  Gumroad: {metrics_summary['map']['gumroad']:.4f}, "
                          f"Custom Endpoint: {metrics_summary['map']['endpoint']:.4f}, "
                          f"Improvement: {(metrics_summary['map']['endpoint'] - metrics_summary['map']['gumroad']) * 100:.2f}%")
        
        logger.info(f"\nCheck {evaluation_results_dir} for full results and visualizations.")
    else:
        logger.info("Skipping metrics calculation step")

if __name__ == "__main__":
    main()