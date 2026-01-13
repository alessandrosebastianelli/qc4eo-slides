"""
Hyperparameter Tuning Script for Hybrid Quantum Classifier

This script provides an easy interface for tuning hyperparameters
using Optuna. Run this instead of main.py to find optimal hyperparameters.

Usage:
    # Tune everything (quantum + classical)
    python tune_hyperparameters.py
    
    # Tune only quantum hyperparameters
    python tune_hyperparameters.py --quantum-only
    
    # Tune only classical hyperparameters
    python tune_hyperparameters.py --classical-only
    
    # Quick test with fewer trials
    python tune_hyperparameters.py --quick
"""

import argparse
import sys
from config import Config
from hyperparameter_tuning import OptunaHybridTuner


def main():
    """Main function for hyperparameter tuning"""
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Hyperparameter tuning for Hybrid Quantum Classifier'
    )
    
    parser.add_argument(
        '--quantum-only',
        action='store_true',
        help='Tune only quantum hyperparameters'
    )
    
    parser.add_argument(
        '--classical-only',
        action='store_true',
        help='Tune only classical hyperparameters'
    )
    
    parser.add_argument(
        '--n-trials',
        type=int,
        default=50,
        help='Number of optimization trials (default: 50)'
    )
    
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Quick mode with only 10 trials'
    )
    
    parser.add_argument(
        '--timeout',
        type=int,
        default=None,
        help='Timeout in seconds (default: None)'
    )
    
    parser.add_argument(
        '--no-visualize',
        action='store_true',
        help='Skip visualization plots'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='tuning_results.json',
        help='Output file for results (default: tuning_results.json)'
    )
    
    args = parser.parse_args()
    
    # Determine what to tune
    tune_quantum = not args.classical_only
    tune_classical = not args.quantum_only
    
    # Determine number of trials
    n_trials = 10 if args.quick else args.n_trials
    
    # Print configuration
    print("\n" + "=" * 70)
    print("HYPERPARAMETER TUNING CONFIGURATION")
    print("=" * 70)
    print(f"Tune quantum parameters:   {tune_quantum}")
    print(f"Tune classical parameters: {tune_classical}")
    print(f"Number of trials:          {n_trials}")
    print(f"Timeout:                   {args.timeout or 'None'}")
    print(f"Generate visualizations:   {not args.no_visualize}")
    print(f"Output file:               {args.output}")
    print("=" * 70 + "\n")
    
    # Create base configuration
    base_config = Config()
    
    # Create tuner
    tuner = OptunaHybridTuner(
        base_config=base_config,
        tune_quantum=tune_quantum,
        tune_classical=tune_classical,
        n_trials=n_trials,
        timeout=args.timeout
    )
    
    # Run optimization
    print("Starting hyperparameter optimization...")
    print("This may take a while depending on the number of trials.\n")
    
    try:
        study = tuner.tune()
        
        # Save results
        tuner.save_results(study, filename=args.output)
        
        # Visualize
        if not args.no_visualize:
            from hyperparameter_tuning import visualize_study_results
            visualize_study_results(study)
        
        # Print best configuration
        print("\n" + "=" * 70)
        print("BEST CONFIGURATION FOUND")
        print("=" * 70)
        tuner.best_config.print_config()
        
        # Save best config to Python file for easy reuse
        best_config_file = 'best_config.py'
        with open(best_config_file, 'w') as f:
            f.write('"""Best configuration found by hyperparameter tuning"""\n\n')
            f.write('from config import Config\n\n')
            f.write('def get_best_config():\n')
            f.write('    """Return the best configuration from tuning"""\n')
            f.write('    config = Config()\n')
            
            if tune_quantum:
                f.write(f'    config.N_QUBITS = {tuner.best_config.N_QUBITS}\n')
                f.write(f'    config.N_LAYERS = {tuner.best_config.N_LAYERS}\n')
                f.write(f'    config.ENCODING = "{tuner.best_config.ENCODING}"\n')
                f.write(f'    config.CIRCUIT_TYPE = "{tuner.best_config.CIRCUIT_TYPE}"\n')
            
            if tune_classical:
                f.write(f'    config.LEARNING_RATE = {tuner.best_config.LEARNING_RATE}\n')
                f.write(f'    config.BATCH_SIZE = {tuner.best_config.BATCH_SIZE}\n')
                f.write(f'    config.DROPOUT_RATE = {tuner.best_config.DROPOUT_RATE}\n')
                f.write(f'    config.CNN_OUTPUT_DIM = {tuner.best_config.CNN_OUTPUT_DIM}\n')
            
            f.write('    return config\n')
        
        print(f"\n✓ Best configuration saved to: {best_config_file}")
        print(f"✓ To use it, run: from best_config import get_best_config")
        
        print("\n" + "=" * 70)
        print("NEXT STEPS")
        print("=" * 70)
        print("1. Review the results in tuning_results.json")
        print("2. Check visualization plots in tuning_plots/")
        print("3. Use the best configuration:")
        print("   >>> from best_config import get_best_config")
        print("   >>> config = get_best_config()")
        print("   >>> # Now train with this config")
        print("=" * 70 + "\n")
        
    except KeyboardInterrupt:
        print("\n\nOptimization interrupted by user!")
        print("Partial results may have been saved.")
        sys.exit(1)
    
    except Exception as e:
        print(f"\n\nError during optimization: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()




# """
# Simple example demonstrating Optuna hyperparameter tuning

# This script shows how to use the Optuna integration in different ways.
# """

# from hyperparameter_tuning import OptunaHybridTuner, run_study
# from config import Config


# def example_1_simple():
#     """
#     Example 1: Simplest usage - tune everything
#     """
#     print("\n" + "=" * 60)
#     print("EXAMPLE 1: Simple Tuning")
#     print("=" * 60)
    
#     # Just run the study
#     best_config, study = run_study(
#         tune_quantum=True,
#         tune_classical=True,
#         n_trials=10,  # Small number for demo
#         visualize=False,  # Skip plots for demo
#         save_results=True
#     )
    
#     print(f"\nBest accuracy found: {study.best_value:.4f}")
#     print("\nBest configuration:")
#     best_config.print_config()


# def example_2_quantum_only():
#     """
#     Example 2: Tune only quantum hyperparameters
#     """
#     print("\n" + "=" * 60)
#     print("EXAMPLE 2: Quantum-Only Tuning")
#     print("=" * 60)
    
#     # Create tuner
#     tuner = OptunaHybridTuner(
#         tune_quantum=True,
#         tune_classical=False,  # Keep classical fixed
#         n_trials=10
#     )
    
#     # Run optimization
#     study = tuner.tune()
    
#     print("\nBest quantum configuration:")
#     print(f"  Qubits:   {tuner.best_config.N_QUBITS}")
#     print(f"  Layers:   {tuner.best_config.N_LAYERS}")
#     print(f"  Encoding: {tuner.best_config.ENCODING}")
#     print(f"  Circuit:  {tuner.best_config.CIRCUIT_TYPE}")


# def example_3_two_stage():
#     """
#     Example 3: Two-stage tuning (quantum then classical)
#     """
#     print("\n" + "=" * 60)
#     print("EXAMPLE 3: Two-Stage Tuning")
#     print("=" * 60)
    
#     # Stage 1: Optimize quantum parameters
#     print("\nStage 1: Tuning quantum parameters...")
#     tuner1 = OptunaHybridTuner(
#         tune_quantum=True,
#         tune_classical=False,
#         n_trials=5
#     )
#     study1 = tuner1.tune()
    
#     print(f"\nStage 1 complete. Best accuracy: {study1.best_value:.4f}")
    
#     # Stage 2: Optimize classical parameters with best quantum config
#     print("\nStage 2: Tuning classical parameters...")
#     tuner2 = OptunaHybridTuner(
#         base_config=tuner1.best_config,  # Start from best quantum
#         tune_quantum=False,
#         tune_classical=True,
#         n_trials=5
#     )
#     study2 = tuner2.tune()
    
#     print(f"\nStage 2 complete. Best accuracy: {study2.best_value:.4f}")
#     print(f"Improvement: {(study2.best_value - study1.best_value):.4f}")


# def example_4_custom_ranges():
#     """
#     Example 4: Custom search ranges (advanced)
    
#     This requires modifying the OptunaHybridTuner.objective method,
#     but shows how you could customize the search space.
#     """
#     print("\n" + "=" * 60)
#     print("EXAMPLE 4: Custom Search Ranges (Conceptual)")
#     print("=" * 60)
    
#     print("""
#     To customize search ranges, modify optuna_tuner.py:
    
#     In the objective function:
    
#     # Custom quantum ranges
#     config.N_QUBITS = trial.suggest_int('n_qubits', 6, 8)  # Only 6 or 8
#     config.N_LAYERS = trial.suggest_int('n_layers', 3, 5)  # 3, 4, or 5
    
#     # Custom encoding choices
#     config.ENCODING = trial.suggest_categorical(
#         'encoding',
#         ['angle', 'iqp']  # Only these two
#     )
    
#     # Custom learning rate range
#     config.LEARNING_RATE = trial.suggest_float(
#         'learning_rate',
#         1e-4, 1e-3,  # Narrower range
#         log=True
#     )
#     """)


# def example_5_analyze_results():
#     """
#     Example 5: Analyze tuning results
#     """
#     print("\n" + "=" * 60)
#     print("EXAMPLE 5: Analyzing Results")
#     print("=" * 60)
    
#     # Run a small study
#     tuner = OptunaHybridTuner(
#         tune_quantum=True,
#         tune_classical=True,
#         n_trials=5
#     )
#     study = tuner.tune()
    
#     # Analyze
#     print("\n--- Trial Analysis ---")
#     print(f"Total trials: {len(study.trials)}")
#     print(f"Completed trials: {len([t for t in study.trials if t.state.name == 'COMPLETE'])}")
#     print(f"Pruned trials: {len([t for t in study.trials if t.state.name == 'PRUNED'])}")
    
#     print(f"\nBest trial:")
#     print(f"  Number: {study.best_trial.number}")
#     print(f"  Accuracy: {study.best_value:.4f}")
#     print(f"  Parameters:")
#     for key, value in study.best_params.items():
#         print(f"    {key}: {value}")
    
#     # Show top 3 trials
#     print(f"\nTop 3 trials:")
#     sorted_trials = sorted(
#         [t for t in study.trials if t.value is not None],
#         key=lambda t: t.value,
#         reverse=True
#     )
#     for i, trial in enumerate(sorted_trials[:3], 1):
#         print(f"\n  {i}. Trial #{trial.number}")
#         print(f"     Accuracy: {trial.value:.4f}")
#         print(f"     Key params: qubits={trial.params.get('n_qubits', 'N/A')}, "
#               f"lr={trial.params.get('learning_rate', 'N/A'):.6f}")


# def main():
#     """Run all examples (or comment out ones you don't want)"""
    
#     print("\n" + "=" * 70)
#     print("OPTUNA HYPERPARAMETER TUNING EXAMPLES")
#     print("=" * 70)
#     print("\nThese examples demonstrate different ways to use Optuna tuning.")
#     print("Note: Using small n_trials for demonstration. Increase for real use!")
#     print("=" * 70)
    
#     # Run examples (comment out ones you don't want to run)
    
#     # Uncomment the examples you want to run:
    
#     # example_1_simple()
#     # example_2_quantum_only()
#     # example_3_two_stage()
#     example_4_custom_ranges()  # Just prints info
#     # example_5_analyze_results()
    
#     print("\n" + "=" * 70)
#     print("EXAMPLES COMPLETE")
#     print("=" * 70)
#     print("\nFor real hyperparameter tuning, run:")
#     print("  python tune_hyperparameters.py")
#     print("\nSee HYPERPARAMETER_TUNING_GUIDE.md for detailed documentation.")
#     print("=" * 70 + "\n")


# if __name__ == "__main__":
#     main()
