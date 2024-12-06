from advanced_ml_pipeline import AdvancedMLPipeline

# Main Execution
if __name__ == '__main__':
    # Initialize Pipeline
    pipeline = AdvancedMLPipeline('data/charging_pile.csv')
    
    # Run Complete Analysis
    results = pipeline.run_complete_analysis()
    
    # Print Detailed Results
    print("\n===== Preliminary Analysis =====")
    print("Descriptive Statistics:")
    print(results['preliminary_analysis']['descriptive_stats'])
    
    print("\n===== Model Performance =====")
    for metric, value in results['model_performance'].items():
        print(f"{metric}: {value}")